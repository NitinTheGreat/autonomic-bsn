#!/usr/bin/env python3
"""Phase 3 failure-injection test suite. Makes NO LLM calls.

    python tests/test_injection.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.datasource import DataSource, Window  # noqa: E402
from datasets.dataset_replay_source import DatasetReplaySource  # noqa: E402
from injection.base import InjectionError, is_nan_triple  # noqa: E402
from injection.packet_loss import (  # noqa: E402
    LOSS_PARAMS,
    burst_stats,
    run_chain,
    transition_probs,
)
from injection.registry import (  # noqa: E402
    FAILURE_TYPES,
    SEVERITIES,
    STRATEGIES,
    make_injector,
)

SEED = 4242
N = 12

_fails: list[str] = []
_passes = 0


def check(name: str, ok: bool, extra: str = "") -> None:
    global _passes
    if ok:
        _passes += 1
        print("  PASS  %s%s" % (name, (" -- " + extra) if extra else ""))
    else:
        _fails.append(name)
        print("  FAIL  %s%s" % (name, (" -- " + extra) if extra else ""))


def base(dataset="pamap2"):
    if dataset == "pamap2":
        return DatasetReplaySource("pamap2", subjects=["subject101"],
                                   label_set="PAMAP2_8")
    return DatasetReplaySource("mhealth", subjects=["subject1"],
                               label_set="CANONICAL_6")


def take(src, n=N):
    out = []
    for i, w in enumerate(src.windows()):
        if i >= n:
            break
        out.append(w)
    return out


def frames_equal(a, b) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.t_sec != y.t_sec:
            return False
        for u, v in zip(x.accel_g, y.accel_g):
            if not (u == v or (math.isnan(u) and math.isnan(v))):
                return False
        if (x.gyro_dps is None) != (y.gyro_dps is None):
            return False
        if x.gyro_dps is not None:
            for u, v in zip(x.gyro_dps, y.gyro_dps):
                if not (u == v or (math.isnan(u) and math.isnan(v))):
                    return False
    return True


# --------------------------------------------------------------------------- #
def test_protocol_conformance():
    print("\n[1] every injector is itself a valid DataSource")
    for ft in FAILURE_TYPES:
        inj = make_injector(base(), ft, 2, "ankle", SEED)
        ok = (isinstance(inj, DataSource)
              and isinstance(inj.node_ids, list)
              and isinstance(inj.sampling_rate_hz, float))
        w = next(iter(inj.windows()))
        check("%s conforms to DataSource" % ft, ok and isinstance(w, Window))
    inj = make_injector(base(), "dropout", 2, "ankle", SEED)
    check("wrapped source keeps the base node_ids",
          inj.node_ids == base().node_ids)
    check("wrapped source keeps the base sampling rate",
          inj.sampling_rate_hz == base().sampling_rate_hz)


def test_untouched_bit_identical():
    print("\n[2] untouched nodes pass through BIT-IDENTICAL")
    clean = take(base())
    for ft in FAILURE_TYPES:
        for sev in (1, 4):
            got = take(make_injector(base(), ft, sev, "ankle", SEED))
            others = all(frames_equal(c.frames[n], g.frames[n])
                         for c, g in zip(clean, got)
                         for n in ("wrist", "chest"))
            check("%s sev%d leaves wrist+chest untouched" % (ft, sev), others)


def test_no_none_ever_written():
    print("\n[3] no injector EVER writes None where the base had a tuple")
    for dataset in ("pamap2", "mhealth"):
        clean = take(base(dataset))
        for ft in FAILURE_TYPES:
            for sev in SEVERITIES:
                got = take(make_injector(base(dataset), ft, sev, "ankle", SEED))
                bad = 0
                for c, g in zip(clean, got):
                    for a, b in zip(c.frames["ankle"], g.frames["ankle"]):
                        if a.accel_g is not None and b.accel_g is None:
                            bad += 1
                        if a.gyro_dps is not None and b.gyro_dps is None:
                            bad += 1
                if bad:
                    check("%s/%s sev%d writes no None" % (dataset, ft, sev),
                          False, "%d None writes" % bad)
        check("%s: no None written by any injector at any severity" % dataset,
              True)


def test_nan_not_zero():
    print("\n[4] blanked samples are NaN, not zero")
    got = take(make_injector(base(), "dropout", 4, "ankle", SEED))
    frames = got[0].frames["ankle"]
    check("dropout sev4 blanks the whole window with NaN",
          all(is_nan_triple(f.accel_g) for f in frames))
    check("blanked values are NOT zero",
          not any(f.accel_g == (0.0, 0.0, 0.0) for f in frames))
    check("gyro also blanked to NaN, still a tuple",
          all(f.gyro_dps is not None and is_nan_triple(f.gyro_dps)
              for f in frames))


def test_rate_degradation_no_nans():
    print("\n[5] rate_degradation introduces ZERO NaNs (stale, not absent)")
    for sev in SEVERITIES:
        got = take(make_injector(base(), "rate_degradation", sev, "ankle", SEED))
        nans = sum(1 for w in got for f in w.frames["ankle"]
                   if is_nan_triple(f.accel_g))
        check("rate_degradation sev%d emits 0 NaNs" % sev, nans == 0,
              "%d NaNs" % nans)
    # ...and it genuinely holds values, so unique values drop with severity.
    uniq = []
    for sev in SEVERITIES:
        w = take(make_injector(base(), "rate_degradation", sev, "ankle", SEED), 1)[0]
        uniq.append(len({f.accel_g[0] for f in w.frames["ankle"]}))
    check("held values decrease with severity", all(
        uniq[i] > uniq[i + 1] for i in range(len(uniq) - 1)), str(uniq))
    # packet loss, by contrast, DOES leave gaps -- the distinction Phase 4 needs
    pl = take(make_injector(base(), "packet_loss", 3, "ankle", SEED))
    pl_nans = sum(1 for w in pl for f in w.frames["ankle"]
                  if is_nan_triple(f.accel_g))
    check("packet_loss DOES introduce NaN gaps", pl_nans > 0,
          "%d NaNs -- distinguishable from stale-but-present" % pl_nans)


def test_determinism_and_seed_independence():
    print("\n[6] determinism and seed independence")
    a = take(make_injector(base(), "packet_loss", 3, "ankle", SEED))
    b = take(make_injector(base(), "packet_loss", 3, "ankle", SEED))
    check("same seed -> bit-identical output",
          all(frames_equal(x.frames["ankle"], y.frames["ankle"])
              for x, y in zip(a, b)))

    c = take(make_injector(base(), "packet_loss", 3, "ankle", SEED + 1))
    check("different seed -> different output",
          any(not frames_equal(x.frames["ankle"], z.frames["ankle"])
              for x, z in zip(a, c)))

    # Call-order independence: interleaving two injectors must not change
    # either one's output. Global numpy state would break this.
    i1 = make_injector(base(), "packet_loss", 3, "ankle", SEED)
    i2 = make_injector(base(), "packet_loss", 3, "ankle", SEED + 99)
    g1, g2 = i1.windows(), i2.windows()
    inter = []
    for _ in range(6):
        inter.append(next(g1))
        next(g2)
    check("interleaved generators do not affect each other",
          all(frames_equal(a[i].frames["ankle"], inter[i].frames["ankle"])
              for i in range(len(inter))),
          "own Generator per injector, never global numpy state")


def test_capability_guard():
    print("\n[7] gyro-targeting failure on MHEALTH chest RAISES")

    class GyroOnly(STRATEGIES["dropout"]):
        name = "gyro_probe"
        required_channels = ("accel", "gyro")

    STRATEGIES["gyro_probe"] = GyroOnly
    try:
        try:
            inj = make_injector(base("mhealth"), "gyro_probe", 2, "chest", SEED)
            next(iter(inj.windows()))
            check("raises rather than silently no-opping", False,
                  "no exception")
        except InjectionError as e:
            msg = str(e)
            check("raises rather than silently no-opping", True)
            check("message names the missing channel", "gyro" in msg)
            check("message explains why a no-op would be wrong",
                  "no-op" in msg and "severity axis" in msg)
        # the same failure on a node that HAS gyro must work
        inj = make_injector(base("mhealth"), "gyro_probe", 2, "ankle", SEED)
        next(iter(inj.windows()))
        check("same failure succeeds on a node with a gyro", True)
    finally:
        STRATEGIES.pop("gyro_probe", None)


def test_displacement_partial_on_chest():
    print("\n[8] displacement on MHEALTH chest rotates accel only")
    clean = take(base("mhealth"), 3)
    got = take(make_injector(base("mhealth"), "displacement", 3, "chest", SEED), 3)
    w = got[0]
    check("succeeds on a gyro-less node (partial application is correct here)",
          True)
    check("records gyro_rotated=false", w.meta["realised"]["gyro_rotated"] is False)
    check("chest gyro remains None, not a rotated tuple",
          all(f.gyro_dps is None for f in w.frames["chest"]))
    check("chest accel DID change",
          not frames_equal(clean[0].frames["chest"], w.frames["chest"]))
    # on a node WITH gyro, both must rotate
    w2 = take(make_injector(base("mhealth"), "displacement", 3, "ankle", SEED), 1)[0]
    check("on a gyro-bearing node, gyro_rotated=true",
          w2.meta["realised"]["gyro_rotated"] is True)
    check("ankle gyro values actually changed",
          not frames_equal(clean[0].frames["ankle"], w2.frames["ankle"]))


def test_clock_source_recorded():
    print("\n[9] clock_desync records the clock source per dataset")
    p = take(make_injector(base("pamap2"), "clock_desync", 2, "ankle", SEED), 2)[0]
    m = take(make_injector(base("mhealth"), "clock_desync", 2, "ankle", SEED), 2)[0]
    check("PAMAP2 recorded as measured",
          p.meta["realised"]["clock_source"] == "measured")
    check("MHEALTH recorded as derived",
          m.meta["realised"]["clock_source"] == "derived",
          "so Phase 6 cannot pool the two silently")
    check("offset actually applied to timestamps",
          abs(p.meta["realised"]["delta_ms"] - 200.0) < 1e-6)
    clean = take(base("pamap2"), 2)[0]
    shifted = [f.t_sec for f in p.frames["ankle"]]
    orig = [f.t_sec for f in clean.frames["ankle"]]
    check("every target timestamp moved by the requested delta",
          all(abs((s - o) - 0.2) < 1e-9 for s, o in zip(shifted, orig)))
    check("sensor VALUES are untouched by desync",
          all(a.accel_g == b.accel_g for a, b in
              zip(clean.frames["ankle"], p.frames["ankle"])))


def test_composition():
    print("\n[10] injectors compose")
    inner = make_injector(base(), "dropout", 1, "ankle", SEED)
    outer = make_injector(inner, "displacement", 2, "wrist", SEED + 5)
    w = next(iter(outer.windows()))
    stack = w.meta.get("injection_stack", [])
    check("wrapping an already-wrapped source works", isinstance(w, Window))
    check("both tags visible in the stack",
          "dropout:sev1" in stack and "displacement:sev2" in stack, str(stack))
    check("ankle carries the inner tag",
          w.frames["ankle"][-1].injected_failure == "dropout:sev1")
    check("wrist carries the outer tag",
          w.frames["wrist"][0].injected_failure == "displacement:sev2")
    check("chest untouched by both",
          all(f.injected_failure is None for f in w.frames["chest"]))


def test_monotonicity():
    print("\n[11] realised effect is monotonic in severity")
    keys = {"dropout": ("frac_nan", 1), "clock_desync": ("delta_ms", 1),
            "packet_loss": ("loss_rate", 1),
            "rate_degradation": ("realised_hz", -1),
            "displacement": ("realised_angle_deg", 1)}
    for ft, (key, sign) in keys.items():
        vals = []
        for sev in SEVERITIES:
            got = take(make_injector(base(), ft, sev, "ankle", SEED), 8)
            vals.append(float(np.mean([w.meta["realised"][key] for w in got])))
        mono = all((vals[i + 1] - vals[i]) * sign > 0
                   for i in range(len(vals) - 1))
        check("%s %s is monotonic" % (ft, key), mono,
              str([round(v, 4) for v in vals]))


def test_gilbert_elliott():
    print("\n[12] Gilbert-Elliott calibration")
    for sev, (L, B) in LOSS_PARAMS.items():
        p, r = transition_probs(L, B)
        check("sev%d stationary pi_Bad == L exactly" % sev,
              abs(p / (p + r) - L) < 1e-12, "pi=%.12f" % (p / (p + r)))
    for sev, (L, B) in LOSS_PARAMS.items():
        rates, bursts = [], []
        for c in range(20):
            lost = run_chain(10000, L, B, np.random.default_rng(900 + 10 * sev + c))
            st = burst_stats(lost)
            rates.append(st["loss_rate"])
            bursts.append(st["mean_burst"])
        mL, mB = float(np.mean(rates)), float(np.mean(bursts))
        check("sev%d realised L within 2%% abs" % sev, abs(mL - L) <= 0.02,
              "L %.4f vs %.2f" % (mL, L))
        check("sev%d realised B within 15%%" % sev,
              abs(mB - B) / B <= 0.15, "B %.3f vs %.1f" % (mB, B))


def test_bad_inputs():
    print("\n[13] invalid configuration fails loudly")
    for bad, label in [(("nope", 1, "ankle"), "unknown failure type"),
                       (("dropout", 9, "ankle"), "severity out of range"),
                       (("dropout", 1, "nose"), "unknown target node")]:
        try:
            make_injector(base(), *bad, SEED)
            check("%s raises" % label, False)
        except InjectionError:
            check("%s raises" % label, True)


def main() -> int:
    print("=" * 74)
    print("PHASE 3 INJECTION TEST SUITE (real data, no LLM calls)")
    print("=" * 74)
    test_protocol_conformance()
    test_untouched_bit_identical()
    test_no_none_ever_written()
    test_nan_not_zero()
    test_rate_degradation_no_nans()
    test_determinism_and_seed_independence()
    test_capability_guard()
    test_displacement_partial_on_chest()
    test_clock_source_recorded()
    test_composition()
    test_monotonicity()
    test_gilbert_elliott()
    test_bad_inputs()

    print("\n" + "=" * 74)
    if _fails:
        print("%d PASSED, %d FAILED" % (_passes, len(_fails)))
        for f in _fails:
            print("   FAILED: %s" % f)
        return 1
    print("ALL %d ASSERTIONS PASSED" % _passes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
