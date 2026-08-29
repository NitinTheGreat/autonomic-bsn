#!/usr/bin/env python3
"""PROBE: does a TEMPORAL baseline rescue displacement detection?

Phase 4 section 4 found displacement undetectable from a POPULATION reference
gravity vector, because natural posture swing exceeds the rotation's observable
effect. The stated fix was a temporal baseline: compare a node against its own
recent history instead. This script tests whether that actually works.

Bounded experiment. It does NOT change default monitor behaviour and does NOT
alter any published Phase 4 number. Results land in
results/phase4/temporal_baseline_probe.json and are reported alongside.

Makes NO LLM calls.

The constraint that shapes the whole experiment
-----------------------------------------------
Displacement is PERSISTENT. Once the rolling window fills with post-rotation
data the reference rotates with it and the delta collapses to zero. So the
detectable signal is the TRANSIENT at onset, not the sustained state. Recall is
therefore measured in the first K windows after onset, not pooled across the
whole injected span -- pooled recall is reported too, and the gap between them
is the finding.

An exact shortcut, verified before use
--------------------------------------
Rotation is linear, so mean(R @ a_i) == R @ mean(a_i): the injected window's
mean gravity is exactly R applied to the clean window's mean gravity. That lets
the whole (N x K x severity) sweep run off one pass of precomputed clean
gravity vectors instead of re-running the injector thousands of times.
`--verify-shortcut` checks it against the real injector.

Usage
-----
    python scripts/probe_temporal_baseline.py
    python scripts/probe_temporal_baseline.py --verify-shortcut
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict, deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_replay_source import DatasetReplaySource  # noqa: E402
from health.diagnose import (  # noqa: E402
    NODE_ROTATION_AXIS,
    SMALLEST_TARGET_THETA_DEG,
    TH_DISPLACEMENT_FLOOR_DEG,
    TH_DISPLACEMENT_FRACTION,
    angle_between,
    expected_observable_angle,
)
from health.window_view import blind  # noqa: E402
from injection.displacement import THETA_DEG, rotation_matrix  # noqa: E402
from injection.registry import make_injector  # noqa: E402
from scripts.run_detection_eval import SPLIT  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WINDOW_COUNTS = (5, 10, 20)        # rolling baseline length N
ONSET_HORIZONS = (1, 2, 5)         # recall measured in the first K windows
SEVERITIES = (1, 2, 3, 4)
CLEAN_PREFIX = 25                  # windows before onset, to fill the baseline
SEED = 20260830


# --------------------------------------------------------------------------- #
# gravity extraction
# --------------------------------------------------------------------------- #
def window_gravity(w, node: str):
    """Unit mean-gravity vector for one node, over non-NaN samples."""
    frames = blind(w).frames[node]
    a = np.array([f.accel_g for f in frames], dtype=float)
    a = a[~np.isnan(a).any(axis=1)]
    if not len(a):
        return None
    g = np.mean(a, axis=0)
    n = np.linalg.norm(g)
    return (g / n) if n > 1e-9 else None


def collect(dataset: str, subject: str, label_set: str, nodes, limit=None):
    """Per (activity, node) ordered list of clean unit gravity vectors."""
    src = DatasetReplaySource(dataset, subjects=[subject], label_set=label_set)
    out: dict = defaultdict(lambda: defaultdict(list))
    stream: list = []
    for i, w in enumerate(src.windows()):
        if limit and i >= limit:
            break
        gs = {n: window_gravity(w, n) for n in nodes}
        stream.append({"label": w.label, "gravity": gs})
        for n in nodes:
            if gs[n] is not None:
                out[w.label][n].append(gs[n])
    return out, stream


# --------------------------------------------------------------------------- #
# rolling baseline
# --------------------------------------------------------------------------- #
def rolling_deltas(vectors, n_window: int):
    """Angle of each vector against the mean of the PRECEDING n_window vectors.

    Causal: the current window never contributes to its own reference. Entries
    before the buffer fills are None.
    """
    buf: deque = deque(maxlen=n_window)
    out = []
    for v in vectors:
        if len(buf) < n_window or v is None:
            out.append(None)
        else:
            ref = np.mean(np.array(buf), axis=0)
            rn = np.linalg.norm(ref)
            out.append(angle_between(v, ref / rn) if rn > 1e-9 else None)
        if v is not None:
            buf.append(v)
    return out


def threshold_for(node: str, gravity, natural_spread):
    """Same threshold structure the monitor uses, with a temporal spread term."""
    axis = np.zeros(3)
    axis[NODE_ROTATION_AXIS.get(node, 0)] = 1.0
    alignment = abs(float(np.dot(gravity, axis)))
    phi = math.degrees(math.acos(max(-1.0, min(1.0, alignment))))
    detectable = expected_observable_angle(SMALLEST_TARGET_THETA_DEG, phi)
    th = max(TH_DISPLACEMENT_FLOOR_DEG, TH_DISPLACEMENT_FRACTION * detectable)
    if natural_spread is not None:
        th = max(th, natural_spread)
    return th, alignment


def verify_shortcut(dataset: str, spec: dict, node: str) -> dict:
    """mean(R @ a) == R @ mean(a) -- confirm before relying on it."""
    src = DatasetReplaySource(dataset, subjects=[spec["eval_subject"]],
                              label_set=spec["label_set"])
    clean = []
    for i, w in enumerate(src.windows()):
        if i >= 6:
            break
        clean.append(w)
    inj = make_injector(
        DatasetReplaySource(dataset, subjects=[spec["eval_subject"]],
                            label_set=spec["label_set"]),
        "displacement", 4, node, SEED)
    errs = []
    for i, iw in enumerate(inj.windows()):
        if i >= 6:
            break
        g_clean = window_gravity(clean[i], node)
        g_inj = window_gravity(iw, node)
        if g_clean is None or g_inj is None:
            continue
        R = rotation_matrix(NODE_ROTATION_AXIS[node], THETA_DEG[4])
        pred = R @ g_clean
        pred = pred / np.linalg.norm(pred)
        errs.append(float(np.degrees(np.arccos(
            np.clip(np.dot(pred, g_inj), -1, 1)))))
    return {"n": len(errs), "max_error_deg": max(errs) if errs else None,
            "exact": bool(errs and max(errs) < 1e-6)}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--verify-shortcut", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results: dict = {}
    print("=" * 78)
    print("PROBE: TEMPORAL BASELINE FOR DISPLACEMENT")
    print("=" * 78)

    for dataset in ("pamap2", "mhealth"):
        spec = SPLIT[dataset]
        node = spec["target"]
        nodes = [node]
        print("\n%s -- node %s (calib %s, eval %s)"
              % (dataset.upper(), node, spec["calibration_subject"],
                 spec["eval_subject"]))

        if args.verify_shortcut:
            v = verify_shortcut(dataset, spec, node)
            print("  shortcut check: max error %.2e deg over %d windows -> %s"
                  % (v["max_error_deg"], v["n"],
                     "EXACT" if v["exact"] else "*** NOT EXACT ***"))
            results.setdefault(dataset, {})["shortcut_check"] = v

        # ---- calibration: temporal spread on CLEAN held-out data --------- #
        cal_by_act, cal_stream = collect(dataset, spec["calibration_subject"],
                                         spec["label_set"], nodes, args.windows)
        cal_vecs = [s["gravity"][node] for s in cal_stream]
        spreads = {}
        for N in WINDOW_COUNTS:
            d = [x for x in rolling_deltas(cal_vecs, N) if x is not None]
            spreads[N] = float(np.percentile(d, 95)) if d else None
        print("  temporal spread on clean held-out data (p95 of rolling delta):")
        for N in WINDOW_COUNTS:
            print("     N=%-3d %.2f deg" % (N, spreads[N]))

        # ---- evaluation data --------------------------------------------- #
        ev_by_act, ev_stream = collect(dataset, spec["eval_subject"],
                                       spec["label_set"], nodes, args.windows)
        ev_vecs = [s["gravity"][node] for s in ev_stream]

        ds_res: dict = results.setdefault(dataset, {})
        ds_res.update({
            "node": node,
            "calibration_subject": spec["calibration_subject"],
            "eval_subject": spec["eval_subject"],
            "temporal_spread_p95_deg": spreads,
            "n_eval_windows": len(ev_stream),
        })

        # ---- FPR on clean data, two ways --------------------------------- #
        fpr: dict = {}
        for N in WINDOW_COUNTS:
            # (a) natural ordered stream, which crosses activity boundaries
            deltas = rolling_deltas(ev_vecs, N)
            fp = tot = 0
            for v, d in zip(ev_vecs, deltas):
                if d is None or v is None:
                    continue
                th, _ = threshold_for(node, v, spreads[N])
                tot += 1
                fp += int(d > th)
            # (b) within-activity streams, which never cross a boundary
            fp_wa = tot_wa = 0
            for act, per_node in ev_by_act.items():
                vecs = per_node[node]
                for v, d in zip(vecs, rolling_deltas(vecs, N)):
                    if d is None or v is None:
                        continue
                    th, _ = threshold_for(node, v, spreads[N])
                    tot_wa += 1
                    fp_wa += int(d > th)
            fpr[N] = {
                "mixed_stream": {"n": tot, "fp": fp,
                                 "rate": fp / tot if tot else 0.0},
                "within_activity": {"n": tot_wa, "fp": fp_wa,
                                    "rate": fp_wa / tot_wa if tot_wa else 0.0},
            }
            print("  N=%-3d clean FPR: mixed stream %.4f | within-activity %.4f"
                  % (N, fpr[N]["mixed_stream"]["rate"],
                     fpr[N]["within_activity"]["rate"]))
        ds_res["false_positive_rate"] = fpr

        # ---- recall: onset transient vs sustained ------------------------ #
        recall: dict = {}
        for act in sorted(ev_by_act):
            vecs = ev_by_act[act][node]
            if len(vecs) < CLEAN_PREFIX + max(ONSET_HORIZONS) + 2:
                continue
            for sev in SEVERITIES:
                R = rotation_matrix(NODE_ROTATION_AXIS[node], THETA_DEG[sev])
                # onset at CLEAN_PREFIX: everything after is rotated
                observed = []
                for i, v in enumerate(vecs):
                    if v is None:
                        observed.append(None)
                    elif i < CLEAN_PREFIX:
                        observed.append(v)
                    else:
                        r = R @ v
                        observed.append(r / np.linalg.norm(r))
                for N in WINDOW_COUNTS:
                    deltas = rolling_deltas(observed, N)
                    hits = []
                    for i in range(CLEAN_PREFIX, len(observed)):
                        d, v = deltas[i], observed[i]
                        if d is None or v is None:
                            hits.append(None)
                            continue
                        th, _ = threshold_for(node, v, spreads[N])
                        hits.append(d > th)
                    valid = [h for h in hits if h is not None]
                    key = "%s/sev%d/N%d" % (act, sev, N)
                    entry = {"pooled_recall": (sum(valid) / len(valid))
                             if valid else 0.0, "n_pooled": len(valid)}
                    for K in ONSET_HORIZONS:
                        w = [h for h in hits[:K] if h is not None]
                        entry["recall_at_K%d" % K] = ((sum(w) / len(w))
                                                      if w else 0.0)
                    # how fast the transient dies
                    entry["mean_delta_first"] = float(np.mean(
                        [d for d in [deltas[i] for i in
                                     range(CLEAN_PREFIX,
                                           min(CLEAN_PREFIX + 2, len(deltas)))]
                         if d is not None]) or 0.0) if deltas else 0.0
                    tail = [deltas[i] for i in
                            range(CLEAN_PREFIX + N + 2, len(deltas))]
                    tail = [d for d in tail if d is not None]
                    entry["mean_delta_sustained"] = (float(np.mean(tail))
                                                     if tail else None)
                    recall[key] = entry
        ds_res["recall"] = recall

        print("\n  recall at onset vs pooled (node=%s):" % node)
        print("     %-26s %8s %8s %8s %9s %11s"
              % ("activity/sev/N", "K=1", "K=2", "K=5", "pooled",
                 "delta t0->tail"))
        for key in sorted(recall):
            e = recall[key]
            sus = e["mean_delta_sustained"]
            print("     %-26s %8.2f %8.2f %8.2f %9.2f %5.1f -> %.1f deg"
                  % (key, e["recall_at_K1"], e["recall_at_K2"],
                     e["recall_at_K5"], e["pooled_recall"],
                     e["mean_delta_first"], sus if sus is not None else float("nan")))

    # ---- verdict ---------------------------------------------------------- #
    best_onset, best_pooled, worst_fpr = 0.0, 0.0, 0.0
    for ds in results.values():
        for e in ds.get("recall", {}).values():
            best_onset = max(best_onset, e["recall_at_K1"])
            best_pooled = max(best_pooled, e["pooled_recall"])
        for f in ds.get("false_positive_rate", {}).values():
            worst_fpr = max(worst_fpr, f["mixed_stream"]["rate"])

    payload = {
        "generated_by": "scripts/probe_temporal_baseline.py",
        "phase": "4-probe",
        "status": "PROBE ONLY -- default monitor behaviour unchanged; no Phase "
                  "4 headline number is affected",
        "seed": SEED,
        "rolling_window_lengths": list(WINDOW_COUNTS),
        "onset_horizons": list(ONSET_HORIZONS),
        "severities": list(SEVERITIES),
        "clean_prefix_windows": CLEAN_PREFIX,
        "phase4_population_baseline_reference": {
            "displacement_f1": 0.595,
            "displacement_recall": 0.458,
            "overall_fpr": 0.0667,
            "note": "all 12 Phase 4 false positives were displacement",
        },
        "datasets": results,
        "summary": {
            "best_onset_recall_K1": best_onset,
            "best_pooled_recall": best_pooled,
            "worst_clean_fpr_mixed_stream": worst_fpr,
        },
    }
    out_path = args.out or os.path.join(REPO_ROOT, "results", "phase4",
                                        "temporal_baseline_probe.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    mirror = os.path.join(REPO_ROOT, "frontend", "results", "phase4",
                          os.path.basename(out_path))
    os.makedirs(os.path.dirname(mirror), exist_ok=True)
    with open(mirror, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("  best onset recall (K=1)      : %.3f" % best_onset)
    print("  best pooled recall           : %.3f" % best_pooled)
    print("  worst clean FPR (mixed)      : %.4f" % worst_fpr)
    print("  Phase 4 population baseline  : recall 0.458, FPR 0.0667")
    print("\nwrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
