#!/usr/bin/env python3
"""Run the health monitor across clean and injected windows -> Phase 4 metrics.

Writes results/phase4/detection_metrics.json (mirrored into frontend/).
Makes NO LLM calls.

Subject split
-------------
The per-(node, dataset) reference gravity vectors used by the displacement
check are calibrated on a HELD-OUT subject, disjoint from every subject
evaluated. That reference is calibration data about a node's typical mounting;
it says nothing about whether the window under judgement was injected.

The monitor only ever sees BlindWindows. Ground truth is read solely by
health.score_detection, after the fact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_replay_source import DatasetReplaySource  # noqa: E402
from health.diagnose import (  # noqa: E402
    DIAGNOSES,
    STATE_BOUNDS,
    TRUST_WEIGHT,
    diagnose_window,
)
from health.score_detection import DetectionTally, detection_latency  # noqa: E402
from health.window_view import blind  # noqa: E402
from injection.registry import DESCRIPTIONS, FAILURE_TYPES, make_injector  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260830

SPLIT = {
    "pamap2": {"label_set": "PAMAP2_8",
               "calibration_subject": "subject102",
               "eval_subject": "subject101",
               "target": "ankle"},
    "mhealth": {"label_set": "CANONICAL_6",
                "calibration_subject": "subject2",
                "eval_subject": "subject1",
                "target": "ankle"},
}
SEVERITIES = (1, 2, 3, 4)


def calibrate_references(dataset: str, subject: str, label_set: str,
                         max_windows: int) -> dict:
    """Mean gravity direction per node, from CLEAN held-out windows."""
    src = DatasetReplaySource(dataset, subjects=[subject], label_set=label_set)
    acc: dict = defaultdict(list)
    for i, w in enumerate(src.windows()):
        if i >= max_windows:
            break
        bw = blind(w)
        for node, frames in bw.frames.items():
            a = np.array([f.accel_g for f in frames], dtype=float)
            a = a[~np.isnan(a).any(axis=1)]
            if len(a):
                acc[node].append(np.mean(a, axis=0))
    refs = {}
    for node, vecs in acc.items():
        arr = np.array(vecs)
        v = np.mean(arr, axis=0)
        n = np.linalg.norm(v)
        if n <= 1e-9:
            continue
        ref = v / n
        # Natural spread of the gravity direction across CLEAN held-out
        # windows. A body-worn node swings with posture, and that swing is
        # typically far larger than any sensor rotation -- without this term
        # the displacement rule fires on nearly every clean window.
        angles = []
        for row in arr:
            rn = np.linalg.norm(row)
            if rn > 1e-9:
                angles.append(math.degrees(math.acos(
                    float(np.clip(np.dot(row / rn, ref), -1.0, 1.0)))))
        refs[node] = {
            "vector": ref.tolist(),
            "n_calibration_windows": len(angles),
            "clean_angle_mean": float(np.mean(angles)) if angles else None,
            "clean_angle_p95": float(np.percentile(angles, 95)) if angles else None,
            "clean_angle_max": float(np.max(angles)) if angles else None,
        }
    return refs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=int, default=40,
                    help="evaluation windows per condition")
    ap.add_argument("--calib-windows", type=int, default=60)
    ap.add_argument("--dataset", choices=sorted(SPLIT) + ["all"], default="all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    datasets = sorted(SPLIT) if args.dataset == "all" else [args.dataset]
    tally = DetectionTally()
    per_dataset_meta = {}
    latencies: dict = defaultdict(list)

    for dataset in datasets:
        spec = SPLIT[dataset]
        target = spec["target"]
        print("=" * 78)
        print("HEALTH MONITOR EVAL -- %s" % dataset.upper())
        print("=" * 78)
        print("  calibration subject : %s (held out)" % spec["calibration_subject"])
        print("  evaluation subject  : %s" % spec["eval_subject"])
        print("  target node         : %s\n" % target)

        refs = calibrate_references(dataset, spec["calibration_subject"],
                                    spec["label_set"], args.calib_windows)
        print("  reference gravity vectors (and natural clean spread):")
        for n, r in refs.items():
            v = r["vector"]
            print("     %-7s [%+.3f %+.3f %+.3f]  clean angle: mean %.1f deg, "
                  "p95 %.1f deg, max %.1f deg"
                  % (n, v[0], v[1], v[2], r["clean_angle_mean"],
                     r["clean_angle_p95"], r["clean_angle_max"]))

        def base():
            return DatasetReplaySource(dataset, subjects=[spec["eval_subject"]],
                                       label_set=spec["label_set"])

        # ---- clean windows: false-positive rate ------------------------- #
        n_clean = 0
        for i, w in enumerate(base().windows()):
            if i >= args.windows:
                break
            out = diagnose_window(blind(w), refs)
            tally.add(out, w, dataset)
            n_clean += 1
        print("\n  clean windows scored: %d" % n_clean)

        # ---- injected conditions ---------------------------------------- #
        for ft in FAILURE_TYPES:
            line = []
            for sev in SEVERITIES:
                inj = make_injector(base(), ft, sev, target, SEED)
                outs, n = [], 0
                for i, w in enumerate(inj.windows()):
                    if i >= args.windows:
                        break
                    out = diagnose_window(blind(w), refs)
                    tally.add(out, w, dataset)
                    outs.append(out)
                    n += 1
                    onset = (w.meta.get("realised", {}) or {}).get("onset_sec")
                    if onset is not None:
                        lat = detection_latency([out], onset, target)
                        if lat and lat["latency_sec"] is not None:
                            latencies["%s/sev%d" % (ft, sev)].append(
                                lat["latency_sec"])
                sub = [r for r in tally.rows[-n * 3:]
                       if r["is_target"] and r["true_failure"] == ft]
                rec = (sum(1 for r in sub if r["flagged"]) / len(sub)
                       if sub else 0.0)
                line.append("sev%d %.2f" % (sev, rec))
            print("  %-18s recall: %s" % (ft, "  ".join(line)))

        per_dataset_meta[dataset] = {
            "calibration_subject": spec["calibration_subject"],
            "eval_subject": spec["eval_subject"],
            "target_node": target,
            "label_set": spec["label_set"],
            "reference_gravity": refs,
            "n_clean_windows": n_clean,
        }

    summary = tally.summary()

    # ---- report ---------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("DETECTION F1 BY FAILURE TYPE (pooled across severities)")
    print("=" * 78)
    for ft, d in summary["detection_by_failure"].items():
        a = d["all"]
        print("  %-18s P %.3f  R %.3f  F1 %.3f   (tp %d fp %d fn %d)"
              % (ft, a["precision"], a["recall"], a["f1"], a["tp"], a["fp"],
                 a["fn"]))

    fpr = summary["false_positives"]
    print("\n" + "=" * 78)
    print("FALSE POSITIVES ON CLEAN WINDOWS")
    print("=" * 78)
    print("  %d of %d node-windows flagged -> FPR %.4f"
          % (fpr["false_positives"], fpr["n_clean_node_windows"],
             fpr["false_positive_rate"]))
    if fpr["misdiagnosed_as"]:
        print("  misdiagnosed as: %s" % fpr["misdiagnosed_as"])

    cm = summary["diagnosis_confusion"]
    print("\n" + "=" * 78)
    print("DIAGNOSIS CONFUSION (rows = truth, cols = predicted)")
    print("=" * 78)
    hdr = "".join("%9s" % d[:8] for d in cm["labels"])
    print("%-20s%s" % ("", hdr))
    for i, d in enumerate(cm["labels"]):
        print("%-20s%s" % (d[:19], "".join("%9d" % c for c in cm["matrix"][i])))
    print("\n  overall diagnosis accuracy: %.4f (n=%d)"
          % (cm["overall_accuracy"], cm["n"]))

    print("\n" + "=" * 78)
    print("DISPLACEMENT BY (NODE, ACTIVITY)")
    print("=" * 78)
    disp = summary["displacement_by_node_activity"]
    for k in sorted(disp):
        e = disp[k]
        print("  %-26s n=%3d  recall %.2f  dx %.2f  realised %s  align %s%s"
              % (k, e["n"], e["recall"], e["diagnosis_accuracy"],
                 ("%.2f deg" % e["mean_realised_deg"]) if e["mean_realised_deg"] is not None else "-",
                 ("%.3f" % e["mean_alignment"]) if e["mean_alignment"] is not None else "-",
                 "   <- UNDETECTABLE" if e["undetectable"] else ""))

    payload = {
        "generated_by": "scripts/run_detection_eval.py",
        "phase": 4,
        "seed": SEED,
        "windows_per_condition": args.windows,
        "diagnosis_classes": list(DIAGNOSES),
        "states": {"bounds": STATE_BOUNDS, "trust_weights": TRUST_WEIGHT},
        "failure_descriptions": DESCRIPTIONS,
        "splits": per_dataset_meta,
        "detection_latency_sec": {k: {"n": len(v), "mean": float(np.mean(v)),
                                      "max": float(np.max(v))}
                                  for k, v in latencies.items()},
        **summary,
        "blind_note": ("The monitor only ever received BlindWindows: "
                       "injected_failure, the true activity label and all "
                       "injection metadata are stripped and raise on access. "
                       "Ground truth is read only here, after the fact."),
    }
    out_path = args.out or os.path.join(REPO_ROOT, "results", "phase4",
                                        "detection_metrics.json")
    _write(out_path, payload)
    print("\nwrote %s" % out_path)
    return 0


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    mirror = os.path.join(REPO_ROOT, "frontend", "results", "phase4",
                          os.path.basename(path))
    os.makedirs(os.path.dirname(mirror), exist_ok=True)
    with open(mirror, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
