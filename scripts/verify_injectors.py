#!/usr/bin/env python3
"""Verify every injector before any severity is trusted.

Writes results/phase3/injector_verification.json (mirrored into frontend/).
Makes NO LLM calls.

Checks, per (failure_type, severity, dataset):
  * untouched nodes are BIT-IDENTICAL to the base source
  * the target node measurably differs
  * the realised effect magnitude increases monotonically with severity
  * no injector ever emits None where the base had a tuple
  * rate_degradation emits zero NaNs
  * determinism: same seed -> bit-identical output, twice
  * Gilbert-Elliott calibration

A miscalibrated severity axis would silently distort every Phase 6 curve, so
mismatches fail loudly rather than being noted and passed over.

Usage
-----
    python scripts/verify_injectors.py
    python scripts/verify_injectors.py --dataset pamap2 --windows 40
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_replay_source import DatasetReplaySource  # noqa: E402
from injection.base import InjectionError, is_nan_triple  # noqa: E402
from injection.packet_loss import (  # noqa: E402
    LOSS_PARAMS,
    burst_stats,
    run_chain,
    transition_probs,
)
from injection.registry import (  # noqa: E402
    DESCRIPTIONS,
    FAILURE_TYPES,
    SEVERITIES,
    make_injector,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "pamap2": {"label_set": "PAMAP2_8", "subject": "subject101",
               "target": "ankle"},
    "mhealth": {"label_set": "CANONICAL_6", "subject": "subject1",
                "target": "ankle"},
}

SEED = 20260829

# Which realised metric represents "effect magnitude" for monotonicity.
EFFECT_KEY = {
    "dropout": "frac_nan",
    "clock_desync": "delta_ms",
    "packet_loss": "loss_rate",
    "rate_degradation": "target_hz",      # DEcreases with severity -- see below
    "displacement": "realised_angle_deg",
}
# rate_degradation's rate falls as severity rises; every other effect grows.
EFFECT_DECREASES = {"rate_degradation"}

# Failures whose per-window effect is drawn at random. A low-severity draw can
# legitimately leave a short window untouched, so these are held to an
# aggregate criterion rather than a per-window one.
STOCHASTIC_TYPES = {"packet_loss"}

# Gilbert-Elliott calibration. The spec suggested 10,000 steps; that is
# statistically underpowered for severity 4 (L=0.5, B=8), whose bursty
# autocorrelation gives a single 10k chain sd~0.013 -- roughly 10% of seeds
# land outside a +/-2% absolute band by chance alone. We therefore average
# N_CHAINS independent 10k chains and assert on the mean, while also recording
# the per-chain spread so the underpowering is visible rather than hidden.
GE_STEPS = 10_000
GE_CHAINS = 20
GE_TOL_L = 0.02          # absolute
GE_TOL_B_PCT = 15.0      # percent


def frames_equal(a, b) -> bool:
    """Bit-identical comparison, treating NaN as equal to NaN."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.node_id != y.node_id or x.t_sec != y.t_sec:
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


def count_nans(frames) -> int:
    return sum(1 for f in frames if is_nan_triple(f.accel_g))


def verify_gilbert_elliott() -> dict:
    print("=" * 78)
    print("GILBERT-ELLIOTT CALIBRATION")
    print("=" * 78)
    print("  %d chains x %d steps per severity" % (GE_CHAINS, GE_STEPS))
    print("  %-4s %-10s %-11s %-9s %-10s %-11s %-8s %s"
          % ("sev", "target L", "realised L", "dL", "target B", "realised B",
             "dB%", "verdict"))
    rows, all_ok = [], True
    for sev in SEVERITIES:
        L, B = LOSS_PARAMS[sev]
        rates, bursts = [], []
        for c in range(GE_CHAINS):
            lost = run_chain(GE_STEPS, L, B, np.random.default_rng(SEED + 1000 * sev + c))
            st = burst_stats(lost)
            rates.append(st["loss_rate"])
            bursts.append(st["mean_burst"])
        mL, mB = float(np.mean(rates)), float(np.mean(bursts))
        dL, dB = abs(mL - L), abs(mB - B) / B * 100.0
        ok = dL <= GE_TOL_L and dB <= GE_TOL_B_PCT
        all_ok &= ok
        p, r = transition_probs(L, B)
        # How often a SINGLE 10k chain would pass -- the underpowering, made
        # visible rather than hidden.
        single_pass = float(np.mean(np.abs(np.array(rates) - L) <= GE_TOL_L))
        rows.append({
            "severity": sev, "target_loss_rate": L, "target_mean_burst": B,
            "p_good_to_bad": p, "r_bad_to_good": r,
            "theoretical_pi_bad": p / (p + r),
            "realised_loss_rate": mL, "realised_mean_burst": mB,
            "abs_error_L": dL, "pct_error_B": dB,
            "loss_rate_sd_across_chains": float(np.std(rates)),
            "single_chain_pass_fraction": single_pass,
            "within_tolerance": ok,
        })
        print("  %-4d %-10.4f %-11.4f %-9.4f %-10.1f %-11.3f %-8.2f %s"
              % (sev, L, mL, dL, B, mB, dB, "OK" if ok else "*** FAIL ***"))
    print("\n  formula check pi_Bad == L: %s"
          % all(abs(r["theoretical_pi_bad"] - r["target_loss_rate"]) < 1e-12
                for r in rows))
    return {"steps_per_chain": GE_STEPS, "n_chains": GE_CHAINS,
            "tolerance_L_abs": GE_TOL_L, "tolerance_B_pct": GE_TOL_B_PCT,
            "all_within_tolerance": all_ok, "per_severity": rows,
            "note": ("Averaged over %d independent chains. A single %d-step "
                     "chain is underpowered at severity 4 (L=0.5, B=8): burst "
                     "autocorrelation gives sd~0.013, so ~10%% of seeds miss a "
                     "+/-2%% band by chance. The model itself is exact -- "
                     "pi_Bad == L to 1e-12." % (GE_CHAINS, GE_STEPS))}


def verify_dataset(dataset: str, n_windows: int) -> dict:
    spec = DATASETS[dataset]
    target = spec["target"]
    print("\n" + "=" * 78)
    print("INJECTOR VERIFICATION -- %s (target node: %s)"
          % (dataset.upper(), target))
    print("=" * 78)

    def base():
        return DatasetReplaySource(dataset, subjects=[spec["subject"]],
                                   label_set=spec["label_set"])

    clean = []
    for i, w in enumerate(base().windows()):
        if i >= n_windows:
            break
        clean.append(w)
    print("  baseline: %d windows\n" % len(clean))

    results = {}
    for ftype in FAILURE_TYPES:
        print("  %-18s %s" % (ftype, DESCRIPTIONS[ftype]))
        per_sev, effects, ok_all = [], [], True

        for sev in SEVERITIES:
            inj = make_injector(base(), ftype, sev, target, SEED)
            got = []
            for i, w in enumerate(inj.windows()):
                if i >= n_windows:
                    break
                got.append(w)

            untouched_ok = True
            target_differs = 0
            none_written = 0
            nan_total = 0
            realised = []

            for cw, iw in zip(clean, got):
                for node in cw.frames:
                    if node == target:
                        if not frames_equal(cw.frames[node], iw.frames[node]):
                            target_differs += 1
                        # None must NEVER appear where the base had a tuple.
                        for a, b in zip(cw.frames[node], iw.frames[node]):
                            if a.accel_g is not None and b.accel_g is None:
                                none_written += 1
                            if a.gyro_dps is not None and b.gyro_dps is None:
                                none_written += 1
                        nan_total += count_nans(iw.frames[node])
                    elif not frames_equal(cw.frames[node], iw.frames[node]):
                        untouched_ok = False
                realised.append(iw.meta["realised"])

            key = EFFECT_KEY[ftype]
            vals = [r[key] for r in realised if r.get(key) is not None]
            mean_effect = float(np.mean(vals)) if vals else 0.0
            effects.append(mean_effect)

            # determinism: same seed, run twice -> bit-identical
            inj2 = make_injector(base(), ftype, sev, target, SEED)
            again = []
            for i, w in enumerate(inj2.windows()):
                if i >= n_windows:
                    break
                again.append(w)
            deterministic = all(frames_equal(a.frames[target], b.frames[target])
                                for a, b in zip(got, again))

            # "The target node measurably differs" needs qualifying for
            # STOCHASTIC failures. At packet_loss severity 1 (L=0.05, B=3) a
            # 128-sample MHEALTH window has roughly a 10% chance of drawing no
            # loss at all -- (1-p)^128 with p=0.0175 -- so demanding that
            # EVERY window differ would fail on correct behaviour. For those we
            # require a clear majority to differ and the aggregate effect to be
            # non-zero; deterministic failures must still change every window.
            diff_frac = target_differs / len(got) if got else 0.0
            if ftype in STOCHASTIC_TYPES:
                differs_ok = diff_frac >= 0.5 and mean_effect > 0.0
            else:
                differs_ok = target_differs == len(got)

            checks = {
                "untouched_nodes_bit_identical": untouched_ok,
                "target_node_differs": differs_ok,
                "no_none_written": none_written == 0,
                "deterministic_same_seed": deterministic,
            }
            if ftype == "rate_degradation":
                checks["zero_nans_introduced"] = nan_total == 0

            ok = all(checks.values())
            ok_all &= ok
            per_sev.append({
                "severity": sev,
                "requested": inj.strategy.params(sev),
                "mean_realised_effect": mean_effect,
                "effect_key": key,
                "n_windows": len(got),
                "windows_modified_fraction": diff_frac,
                "stochastic": ftype in STOCHASTIC_TYPES,
                "total_nans_in_target": nan_total,
                "checks": checks,
                "passed": ok,
                "sample_realised": realised[0] if realised else {},
            })
            print("     sev%d  %-22s = %-10.4f  %s"
                  % (sev, key, mean_effect,
                     "OK" if ok else "*** " +
                     ",".join(k for k, v in checks.items() if not v) + " ***"))

        # Monotonicity of the REALISED effect.
        if ftype in EFFECT_DECREASES:
            mono = all(effects[i] > effects[i + 1] for i in range(len(effects) - 1))
            direction = "decreasing"
        else:
            mono = all(effects[i] < effects[i + 1] for i in range(len(effects) - 1))
            direction = "increasing"
        ok_all &= mono
        print("     monotonic (%s): %s  %s\n"
              % (direction, mono, [round(e, 4) for e in effects]))

        results[ftype] = {
            "description": DESCRIPTIONS[ftype],
            "effect_key": EFFECT_KEY[ftype],
            "expected_direction": direction,
            "monotonic": mono,
            "mean_effects_by_severity": effects,
            "per_severity": per_sev,
            "passed": ok_all,
        }

    return {"dataset": dataset, "target_node": target,
            "n_windows_checked": len(clean),
            "subject": spec["subject"], "label_set": spec["label_set"],
            "failures": results,
            "all_passed": all(v["passed"] for v in results.values())}


def verify_capability_guard() -> dict:
    """A gyro-targeting failure on MHEALTH's chest must RAISE, not no-op."""
    print("\n" + "=" * 78)
    print("CAPABILITY GUARD -- gyro-targeting failure on a node with no gyro")
    print("=" * 78)
    base = DatasetReplaySource("mhealth", subjects=["subject1"],
                               label_set="CANONICAL_6")
    from injection.registry import STRATEGIES

    class GyroOnly(STRATEGIES["dropout"]):        # a gyro-requiring failure
        name = "gyro_only_probe"
        required_channels = ("accel", "gyro")

    STRATEGIES["gyro_only_probe"] = GyroOnly
    out = {}
    try:
        inj = make_injector(base, "gyro_only_probe", 2, "chest", SEED)
        next(iter(inj.windows()))
        out = {"raised": False,
               "verdict": "*** DID NOT RAISE -- silent no-op is a bug ***"}
        print("  *** DID NOT RAISE ***")
    except InjectionError as e:
        out = {"raised": True, "message": str(e).splitlines()[0]}
        print("  raised as required: %s" % str(e).splitlines()[0])
    finally:
        STRATEGIES.pop("gyro_only_probe", None)

    # displacement on the same node must SUCCEED, rotating accel only.
    inj = make_injector(
        DatasetReplaySource("mhealth", subjects=["subject1"],
                            label_set="CANONICAL_6"),
        "displacement", 3, "chest", SEED)
    w = next(iter(inj.windows()))
    out["displacement_on_chest_ok"] = True
    out["gyro_rotated"] = w.meta["realised"]["gyro_rotated"]
    out["chest_gyro_still_none"] = all(f.gyro_dps is None
                                       for f in w.frames["chest"])
    print("  displacement on the same node succeeds, gyro_rotated=%s, "
          "chest gyro still None=%s"
          % (out["gyro_rotated"], out["chest_gyro_still_none"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS) + ["all"],
                    default="all")
    ap.add_argument("--windows", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ge = verify_gilbert_elliott()
    targets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    per_dataset = {d: verify_dataset(d, args.windows) for d in targets}
    guard = verify_capability_guard()

    all_ok = (ge["all_within_tolerance"]
              and all(v["all_passed"] for v in per_dataset.values())
              and guard.get("raised") is True)

    payload = {
        "generated_by": "scripts/verify_injectors.py",
        "phase": 3,
        "seed": SEED,
        "failure_types": FAILURE_TYPES,
        "descriptions": DESCRIPTIONS,
        "gilbert_elliott": ge,
        "datasets": per_dataset,
        "capability_guard": guard,
        "all_passed": all_ok,
        "invariants": {
            "none_never_written": ("No injector may write None. None is "
                                   "reserved for structurally absent hardware; "
                                   "a dead node still HAS its sensors."),
            "nan_not_zero": ("Blanked samples are NaN, not zero: zero is a "
                             "real stationary reading a model may believe."),
            "rate_degradation_no_gaps": ("Rate degradation holds stale values "
                                         "and introduces no NaNs; packet loss "
                                         "introduces NaN gaps. Phase 4 must "
                                         "tell these apart."),
        },
    }
    out_path = args.out or os.path.join(REPO_ROOT, "results", "phase3",
                                        "injector_verification.json")
    _write(out_path, payload)

    print("\n" + "=" * 78)
    print("RESULT: %s" % ("ALL CHECKS PASSED" if all_ok else "FAILURES PRESENT"))
    print("=" * 78)
    print("wrote %s" % out_path)
    return 0 if all_ok else 1


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    mirror = os.path.join(REPO_ROOT, "frontend", "results", "phase3",
                          os.path.basename(path))
    os.makedirs(os.path.dirname(mirror), exist_ok=True)
    with open(mirror, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
