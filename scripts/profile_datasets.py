#!/usr/bin/env python3
"""Profile both datasets through the DataSource contract -> the paper's Table 1.

Writes results/phase2/dataset_stats.json (mirrored into frontend/results/, since
http.server cannot serve above its own root).

Per dataset: subject count, sampling rate, node placements, per-node channel
availability, total raw rows, total duration, window count at 2.56 s / 50 %
overlap, and windows-per-class counts. Includes the PAMAP2 reference window
count and whether the +/-1 % assertion against Phase 1's verified 9,909 passed.

Makes NO LLM calls.

Usage
-----
    python scripts/profile_datasets.py
    python scripts/profile_datasets.py --dataset mhealth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import mhealth_loader, pamap2_loader  # noqa: E402
from datasets.dataset_replay_source import (  # noqa: E402
    DEFAULT_OVERLAP,
    DEFAULT_WINDOW_SEC,
    KNOWN_EMPTY_SUBJECTS,
    PAMAP2_REFERENCE_WINDOWS,
    REFERENCE_TOLERANCE,
    DatasetReplaySource,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPECS = {
    "pamap2": {
        "loader": pamap2_loader,
        "label_set": "PAMAP2_8",
        "node_placements": {
            "wrist": "dominant wrist (dataset calls it 'hand')",
            "chest": "chest",
            "ankle": "dominant ankle",
        },
        "node_channels": {n: ["accel", "gyro", "mag"]
                          for n in pamap2_loader.NODE_IDS},
        "timestamp": "measured (column 0)",
        "extra_channels": {"heart_rate": "~9 Hz, sparse, never interpolated"},
    },
    "mhealth": {
        "loader": mhealth_loader,
        "label_set": "CANONICAL_6",
        "node_placements": {
            "wrist": "right lower arm",
            "chest": "chest",
            "ankle": "left ankle",
        },
        "node_channels": mhealth_loader.NODE_CHANNELS,
        "timestamp": "DERIVED from row index at 50 Hz (no time column exists)",
        "extra_channels": {"ecg": "2-lead ECG on the chest node, physiological "
                                  "only -- never a motion input"},
    },
}


def profile(dataset: str) -> dict:
    spec = SPECS[dataset]
    loader = spec["loader"]
    data_dir = loader.default_dir(REPO_ROOT)
    subjects = loader.available_subjects(data_dir)

    print("=" * 78)
    print("PROFILING %s  (label set: %s)" % (dataset.upper(), spec["label_set"]))
    print("=" * 78)
    if not subjects:
        print("  NO FILES FOUND at %s -- see data/raw/README.md" % data_dir)
        return {"dataset": dataset, "available": False, "source_dir": data_dir}

    src = DatasetReplaySource(dataset, label_set=spec["label_set"],
                              data_dir=data_dir)

    total_rows = 0
    total_seconds = 0.0
    per_subject: dict[str, dict] = {}
    class_counts: Counter = Counter()
    total_windows = 0

    for s in subjects:
        t0 = time.time()
        df = loader.load_subject(data_dir, s)
        if df is None:
            continue
        rows = len(df)
        dur = float(df["timestamp"].max() - df["timestamp"].min())
        counts: Counter = Counter()
        n_win = 0
        for w in src.windows_from_frame(df, DEFAULT_WINDOW_SEC, DEFAULT_OVERLAP):
            counts[w.label] += 1
            n_win += 1
        total_rows += rows
        total_seconds += dur
        total_windows += n_win
        class_counts.update(counts)
        per_subject[s] = {
            "rows": rows,
            "duration_min": dur / 60.0,
            "windows": n_win,
            "windows_per_class": dict(sorted(counts.items())),
        }
        note = ""
        if n_win == 0 and s in KNOWN_EMPTY_SUBJECTS.get(dataset, set()):
            note = "  (expected: too little data to window)"
        print("  %-12s %8d rows  %6.1f min  %5d windows%s"
              % (s, rows, dur / 60.0, n_win, note))
        del df

    out = {
        "dataset": dataset,
        "available": True,
        "source_dir": data_dir,
        "label_set": spec["label_set"],
        "classes": src.classes,
        "n_classes": len(src.classes),
        "n_subjects": len(per_subject),
        "subjects": sorted(per_subject),
        "sampling_rate_hz": src.sampling_rate_hz,
        "node_ids": src.node_ids,
        "node_placements": spec["node_placements"],
        "node_channels": spec["node_channels"],
        "timestamp_source": spec["timestamp"],
        "extra_channels": spec["extra_channels"],
        "total_raw_rows": total_rows,
        "total_duration_min": total_seconds / 60.0,
        "total_duration_hours": total_seconds / 3600.0,
        "window_sec": DEFAULT_WINDOW_SEC,
        "overlap": DEFAULT_OVERLAP,
        "total_windows": total_windows,
        "windows_per_class": dict(sorted(class_counts.items())),
        "per_subject": per_subject,
        "known_empty_subjects": sorted(KNOWN_EMPTY_SUBJECTS.get(dataset, [])),
    }

    print("\n  totals: %d rows, %.1f h, %d windows across %d subjects"
          % (total_rows, out["total_duration_hours"], total_windows,
             len(per_subject)))
    print("  windows per class:")
    for k, v in out["windows_per_class"].items():
        print("     %-20s %6d  %s" % (k, v, "#" * int(v / 120)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(SPECS) + ["all"], default="all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    targets = sorted(SPECS) if args.dataset == "all" else [args.dataset]
    profiles = {d: profile(d) for d in targets}

    ref = None
    if "pamap2" in profiles and profiles["pamap2"].get("available"):
        actual = profiles["pamap2"]["total_windows"]
        lo = PAMAP2_REFERENCE_WINDOWS * (1 - REFERENCE_TOLERANCE)
        hi = PAMAP2_REFERENCE_WINDOWS * (1 + REFERENCE_TOLERANCE)
        ref = {
            "actual": actual,
            "reference": PAMAP2_REFERENCE_WINDOWS,
            "source": "Phase 1 verified figure",
            "tolerance_pct": REFERENCE_TOLERANCE * 100,
            "delta": actual - PAMAP2_REFERENCE_WINDOWS,
            "delta_pct": 100.0 * (actual - PAMAP2_REFERENCE_WINDOWS)
            / PAMAP2_REFERENCE_WINDOWS,
            "assertion_passed": lo <= actual <= hi,
        }
        print("\n" + "=" * 78)
        print("REFERENCE CHECK -- PAMAP2 window count vs Phase 1")
        print("=" * 78)
        print("  actual %d  reference %d  delta %+d (%+.3f%%)  within +/-1%%: %s"
              % (ref["actual"], ref["reference"], ref["delta"],
                 ref["delta_pct"], ref["assertion_passed"]))
        if not ref["assertion_passed"]:
            print("  *** DIVERGED from Phase 1's verified behaviour -- "
                  "investigate, do not accept.")

    payload = {
        "generated_by": "scripts/profile_datasets.py",
        "phase": 2,
        "window_sec": DEFAULT_WINDOW_SEC,
        "overlap": DEFAULT_OVERLAP,
        "datasets": profiles,
        "pamap2_reference_check": ref,
        "schema_note": (
            "Node channel availability is NOT uniform across datasets. "
            "MHEALTH's chest node carries accel + ECG only -- no gyroscope, no "
            "magnetometer -- while every PAMAP2 node is a full 9-axis IMU. "
            "Feature extractors must branch on channels_present and must never "
            "zero-fill an absent channel."),
    }
    out_path = args.out or os.path.join(REPO_ROOT, "results", "phase2",
                                        "dataset_stats.json")
    _write(out_path, payload)
    print("\nwrote %s" % out_path)
    return 0 if (ref is None or ref["assertion_passed"]) else 1


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    mirror = os.path.join(REPO_ROOT, "frontend", "results", "phase2",
                          os.path.basename(path))
    os.makedirs(os.path.dirname(mirror), exist_ok=True)
    with open(mirror, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
