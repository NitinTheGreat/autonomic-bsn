#!/usr/bin/env python3
"""Phase 4 health-monitor test suite. Makes NO LLM calls.

    python tests/test_health.py
"""

from __future__ import annotations

import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.datasource import NodeFrame, Window  # noqa: E402
from datasets.dataset_replay_source import DatasetReplaySource  # noqa: E402
from health.diagnose import (  # noqa: E402
    DIAGNOSES,
    TRUST_WEIGHT,
    diagnose_node,
    diagnose_window,
    expected_observable_angle,
    state_for,
)
from health.signals import window_signals  # noqa: E402
from health.window_view import (  # noqa: E402
    PRESERVED_META_KEYS,
    STRIPPED_META_KEYS,
    BlindWindow,
    GroundTruthAccessError,
    blind,
)
from injection.registry import make_injector  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 77

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


def base(ds="pamap2"):
    if ds == "pamap2":
        return DatasetReplaySource("pamap2", subjects=["subject101"],
                                   label_set="PAMAP2_8")
    return DatasetReplaySource("mhealth", subjects=["subject1"],
                               label_set="CANONICAL_6")


def first(src, n=1):
    out = []
    for i, w in enumerate(src.windows()):
        if i >= n:
            break
        out.append(w)
    return out if n > 1 else out[0]


# --------------------------------------------------------------------------- #
def test_blind_window():
    print("\n[1] BlindWindow enforces the blind mechanically")
    inj = make_injector(base(), "dropout", 3, "ankle", SEED)
    w = first(inj)
    bw = blind(w)

    try:
        _ = bw.frames["ankle"][0].injected_failure
        check("BlindFrame RAISES on injected_failure", False, "returned a value")
    except GroundTruthAccessError as e:
        check("BlindFrame RAISES on injected_failure", True)
        check("error explains the circularity", "answer key" in str(e))

    check("raw Window DOES carry injected_failure (control)",
          w.frames["ankle"][-1].injected_failure == "dropout:sev3")

    for k in STRIPPED_META_KEYS:
        if k in (w.meta or {}):
            try:
                bw.meta[k]
                check("meta key %r stripped" % k, False, "still present")
            except KeyError:
                check("meta key %r stripped" % k, True)
    check("all ground-truth meta keys gone",
          not (set(bw.meta) & STRIPPED_META_KEYS), str(sorted(bw.meta)))

    try:
        _ = bw.label
        check("BlindWindow RAISES on label", False)
    except GroundTruthAccessError:
        check("BlindWindow RAISES on label (true activity is inferred, "
              "never given)", True)

    fm = bw.frames["ankle"][0].meta
    check("channels_present PRESERVED", "channels_present" in fm)
    check("node cannot be penalised for a sensor it lacks -- channels visible",
          isinstance(fm["channels_present"], list) and fm["channels_present"])

    mb = blind(first(base("mhealth")))
    cm = mb.frames["chest"][0].meta
    check("timestamp_derived PRESERVED (MHEALTH)",
          cm.get("timestamp_derived") is True)
    check("dataset_name / subject_id preserved",
          "dataset_name" in cm and "subject_id" in cm)


def test_raw_window_rejected():
    print("\n[2] signals/diagnose accept ONLY a BlindWindow")
    w = first(base())
    for fn, label in ((window_signals, "window_signals"),
                      (diagnose_window, "diagnose_window")):
        try:
            fn(w)
            check("%s rejects a raw Window" % label, False, "accepted it")
        except TypeError as e:
            check("%s rejects a raw Window" % label, True)
            check("%s error explains why" % label,
                  "ground-truth channel" in str(e))
    check("both accept a BlindWindow",
          isinstance(window_signals(blind(w)), dict)
          and isinstance(diagnose_window(blind(w)), dict))


def test_grep_blind():
    print("\n[3] grep: ground-truth identifiers absent from monitor code")
    banned = ["injected_failure", "failure_type", "severity", "realised"]
    for path in ("health/signals.py", "health/diagnose.py"):
        src = open(os.path.join(REPO_ROOT, path), encoding="utf-8").read()
        hits = [b for b in banned if b in src]
        check("%s contains none of %s" % (path, banned), not hits,
              ("found %s" % hits) if hits else "clean")


def test_nan_not_zero_imputed():
    print("\n[4] NaN-aware features are NOT zero-imputed")
    n = 100
    frames = []
    for i in range(n):
        # A constant-offset signal: zero-imputing the NaN half would inflate
        # variance enormously, so the two answers cannot coincide by accident.
        val = (2.0, 0.0, 0.0) if i < 50 else (float("nan"),) * 3
        frames.append(NodeFrame(node_id="ankle", t_sec=i * 0.01,
                                accel_g=val, gyro_dps=(0.0, 0.0, 0.0),
                                source="dataset",
                                meta={"channels_present": ["accel", "gyro"],
                                      "sampling_rate_hz": 100.0}))
    w = Window(start_sec=0.0, end_sec=1.0, frames={"ankle": frames},
               label="test", meta={"sampling_rate_hz": 100.0})
    sig = window_signals(blind(w))["nodes"]["ankle"]

    zero_imputed = np.array([[2.0, 0, 0]] * 50 + [[0.0, 0, 0]] * 50)
    var_if_imputed = float(np.var(zero_imputed[:, 0]))
    check("variance computed over valid samples only",
          abs(sig["axis_variance"][0] - 0.0) < 1e-12,
          "got %.6f; zero-imputed would be %.6f"
          % (sig["axis_variance"][0], var_if_imputed))
    check("zero-imputation would have changed the answer",
          var_if_imputed > 0.9)
    check("nan_fraction reported correctly",
          abs(sig["nan_fraction"] - 0.5) < 1e-9)
    check("mean magnitude uses valid samples only",
          abs(sig["mean_accel_magnitude_g"] - 2.0) < 1e-9)


def test_stale_vs_lossy():
    print("\n[5] rate_degradation and packet_loss are not confusable")
    rd = diagnose_window(blind(first(
        make_injector(base(), "rate_degradation", 4, "ankle", SEED))))["nodes"]["ankle"]
    pl = diagnose_window(blind(first(
        make_injector(base(), "packet_loss", 3, "ankle", SEED))))["nodes"]["ankle"]

    check("rate_degradation produces ZERO NaN",
          rd["signals"]["nan_fraction"] == 0.0)
    check("rate_degradation has a high repeat fraction",
          rd["signals"]["repeat_fraction"] > 0.5,
          "%.3f" % rd["signals"]["repeat_fraction"])
    check("rate_degradation diagnosed correctly",
          rd["diagnosis"] == "rate_degradation", rd["diagnosis"])

    check("packet_loss DOES produce NaN", pl["signals"]["nan_fraction"] > 0)
    check("packet_loss has a normal repeat fraction",
          pl["signals"]["repeat_fraction"] < 0.2,
          "%.3f" % pl["signals"]["repeat_fraction"])
    check("packet_loss diagnosed correctly",
          pl["diagnosis"] == "packet_loss", pl["diagnosis"])
    check("the two diagnoses differ", rd["diagnosis"] != pl["diagnosis"])


def test_dropout_vs_packet_loss():
    print("\n[6] terminal NaN (dropout) vs bursty NaN (packet_loss)")
    do = diagnose_window(blind(first(
        make_injector(base(), "dropout", 3, "ankle", SEED))))["nodes"]["ankle"]
    pl = diagnose_window(blind(first(
        make_injector(base(), "packet_loss", 2, "ankle", SEED))))["nodes"]["ankle"]
    check("dropout is terminal", do["signals"]["nan_terminal"] is True)
    check("dropout diagnosed as dropout", do["diagnosis"] == "dropout")
    check("dropout evidence cites the terminal run",
          "terminal" in do["evidence"]["rule"])
    check("packet_loss diagnosed as packet_loss", pl["diagnosis"] == "packet_loss")
    check("packet_loss evidence cites burstiness",
          "bursty" in pl["evidence"]["rule"] or
          "not terminal" in pl["evidence"]["rule"])


def test_mhealth_chest_gyro():
    print("\n[7] MHEALTH chest: gyro unavailable, NOT penalised")
    out = diagnose_window(blind(first(base("mhealth"))))
    chest = out["nodes"]["chest"]
    ankle = out["nodes"]["ankle"]
    g = chest["signals"]["gyro"]
    check("chest gyro reported unavailable", g["available"] is False)
    check("gyro stats are 'unavailable', not a default number",
          g["mean_abs_dps"] == "unavailable" and g["variance"] == "unavailable")
    check("reason recorded", "no gyroscope" in g.get("reason", ""))
    check("chest NOT penalised for the missing sensor",
          chest["state"] == "HEALTHY", "state=%s min=%.3f"
          % (chest["state"], chest["min_sub_score"]))
    check("ankle DOES have gyro (control)",
          ankle["signals"]["gyro"]["available"] is True)
    check("chest channels_present is accel+ecg",
          chest["channels_present"] == ["accel", "ecg"])


def test_min_aggregation():
    print("\n[8] MIN aggregation -- one bad sub-score drives the node down")
    sig = {"n_frames": 100, "nan_fraction": 0.0, "unique_value_ratio": 1.0,
           "max_repeat_run": 1, "repeat_fraction": 0.0,
           "cross_node_offset_ms": 0.0, "gravity_vector": None,
           "sub_scores": {"completeness": 0.0, "staleness": 1.0,
                          "temporal": 1.0, "sanity": 1.0}}
    v = diagnose_node(sig, "ankle", None)
    check("min drives the state, not the mean", v["min_sub_score"] == 0.0)
    check("node is OFFLINE despite 3 perfect sub-scores",
          v["state"] == "OFFLINE", "arithmetic mean would be 0.75 -> HEALTHY")
    check("trust weight is 0.0", v["trust_weight"] == 0.0)
    check("geometric mean recorded alongside",
          v["geometric_mean_sub_score"] < 0.01)
    check("aggregation is documented in the output",
          "conjunction" in v["aggregation"])
    check("state bands map as specified",
          state_for(0.95) == "HEALTHY" and state_for(0.7) == "DEGRADED"
          and state_for(0.4) == "UNRELIABLE" and state_for(0.1) == "OFFLINE")
    check("trust weights as specified",
          TRUST_WEIGHT == {"HEALTHY": 1.0, "DEGRADED": 0.6,
                           "UNRELIABLE": 0.25, "OFFLINE": 0.0})


def test_evidence_always_returned():
    print("\n[9] every verdict carries its evidence")
    for ft in ("dropout", "packet_loss", "rate_degradation", "clock_desync",
               "displacement"):
        v = diagnose_window(blind(first(
            make_injector(base(), ft, 3, "ankle", SEED))))["nodes"]["ankle"]
        ok = (isinstance(v["evidence"], dict) and v["evidence"].get("rule")
              and len(v["evidence"]) >= 2)
        check("%s verdict includes named evidence" % ft, bool(ok),
              v["evidence"].get("rule", "")[:52])
    clean = diagnose_window(blind(first(base())))["nodes"]["wrist"]
    check("healthy verdict also carries evidence",
          clean["evidence"].get("rule") == "no rule fired")
    check("diagnosis is always one of the 6 classes",
          clean["diagnosis"] in DIAGNOSES)


def test_alignment_geometry():
    print("\n[10] alignment-conditioned displacement threshold")
    # phi = 90 deg (gravity perpendicular to axis) -> full theta observable
    check("phi=90 gives the full angle",
          abs(expected_observable_angle(15.0, 90.0) - 15.0) < 1e-9)
    # phi = 0 (gravity along the axis) -> nothing observable
    check("phi=0 gives ~0 observable", expected_observable_angle(15.0, 0.0) < 1e-6)
    mid = expected_observable_angle(15.0, 30.0)
    check("intermediate alignment is between the two", 0 < mid < 15.0,
          "%.2f deg at phi=30" % mid)
    check("matches Phase 3's measured chest/standing case",
          abs(expected_observable_angle(15.0, math.degrees(math.acos(0.988)))
              - 2.3) < 0.6,
          "predicted %.2f deg vs 2.33 measured"
          % expected_observable_angle(15.0, math.degrees(math.acos(0.988))))

    out = diagnose_window(blind(first(
        make_injector(base(), "displacement", 4, "ankle", SEED))),
        {"ankle": {"vector": [0.0, -1.0, 0.0], "clean_angle_p95": 5.0}})
    d = out["nodes"]["ankle"]["displacement"]
    check("threshold scales with observability, not a constant",
          d["threshold_deg"] is not None and d["expected_min_detectable_deg"] > 0)
    check("alignment estimated from the window itself",
          0.0 <= d["estimated_alignment"] <= 1.0)
    check("natural spread folded into the threshold",
          d["threshold_deg"] >= 5.0)


def test_no_gyro_no_penalty_scores():
    print("\n[11] cross-node offset is 'not applicable' with one node")
    w = first(base())
    single = Window(start_sec=w.start_sec, end_sec=w.end_sec,
                    frames={"ankle": w.frames["ankle"]}, label=w.label,
                    meta=dict(w.meta or {}))
    sig = window_signals(blind(single))["nodes"]["ankle"]
    check("single-node window reports not_applicable",
          sig["cross_node_offset_ms"] == "not_applicable")
    check("synchrony sub-score omitted, not defaulted to a penalty",
          "synchrony" not in sig["sub_scores"])
    check("hardware-only clock metrics reported unavailable",
          sig["clock_offset_ms"] == "unavailable"
          and sig["skew_ppm"] == "unavailable")


def main() -> int:
    print("=" * 74)
    print("PHASE 4 HEALTH MONITOR TEST SUITE (real data, no LLM calls)")
    print("=" * 74)
    test_blind_window()
    test_raw_window_rejected()
    test_grep_blind()
    test_nan_not_zero_imputed()
    test_stale_vs_lossy()
    test_dropout_vs_packet_loss()
    test_mhealth_chest_gyro()
    test_min_aggregation()
    test_evidence_always_returned()
    test_alignment_geometry()
    test_no_gyro_no_penalty_scores()

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
