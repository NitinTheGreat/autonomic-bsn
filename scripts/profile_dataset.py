#!/usr/bin/env python3
"""Phase 1 -- profile the PAMAP2 dataset for the dashboard.

Produces results/phase1/dataset_profile.json: what is actually in the data,
so the Phase 1 page can show the dataset rather than describe it. Needs no
LLM and no API key -- run it as soon as the data is downloaded.

Reuses the verified column map and windowing from check_baseline_accuracy.py,
so the window counts shown here are the same ones the accuracy run samples
from.

Usage
-----
    python scripts/profile_dataset.py
    python scripts/profile_dataset.py --trace-seconds 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _llm_client import REPO_ROOT, load_config  # noqa: E402
from check_baseline_accuracy import (  # noqa: E402
    AXES,
    NODE_COLS,
    ORDERED_IDS,
    PAMAP2_ALL_ACTIVITIES,
    TARGET_ACTIVITIES,
    load_subject,
    resolve_data_dir,
    segment_and_window,
    verify_activity_ids,
)

SUBJECTS = list(range(101, 110))


def signal_traces(df, activity_id: int, seconds: float, hz: int) -> dict | None:
    """A short contiguous raw-signal excerpt, for sparklines on the dashboard."""
    seg = df[df["activityID"] == activity_id]
    if seg.empty:
        return None
    feat_cols = [c for c in df.columns
                 if c not in ("timestamp", "activityID", "subject")]
    seg = seg.dropna(subset=feat_cols)
    n = int(seconds * hz)
    if len(seg) < n:
        return None
    # Take from the middle of the segment: the start is often still settling.
    mid = len(seg) // 2
    block = seg.iloc[mid - n // 2: mid + n // 2]
    # Downsample to keep the JSON small; the shape is what matters visually.
    step = max(1, len(block) // 128)
    out = {}
    for node in NODE_COLS:
        out[node] = {a: [round(float(v), 3) for v in
                         block["%s_%s" % (node, a)].to_numpy()[::step]]
                     for a in AXES}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--trace-seconds", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    hz = int(cfg["baseline"]["sampling_rate_hz"])
    data_dir = resolve_data_dir(cfg, args.data_dir)
    out_path = args.out or os.path.join(
        REPO_ROOT, "results", "phase1", "dataset_profile.json")

    print("=" * 78)
    print("PAMAP2 DATASET PROFILE")
    print("=" * 78)
    print("source: %s\n" % data_dir)

    frames, per_subject, traces = {}, {}, {}
    total_rows = 0
    nan_rows = 0

    for s in SUBJECTS:
        df = load_subject(data_dir, s)
        if df is None:
            print("  subject%d.dat  MISSING" % s)
            continue
        frames[s] = df
        feat_cols = [c for c in df.columns
                     if c not in ("timestamp", "activityID", "subject")]
        n_nan = int(df[feat_cols].isna().any(axis=1).sum())
        counts = Counter(df["activityID"].astype(int).tolist())
        dur = float(df["timestamp"].max() - df["timestamp"].min())
        per_subject[s] = {
            "rows": len(df),
            "duration_s": dur,
            "nan_rows": n_nan,
            "nan_pct": 100.0 * n_nan / max(1, len(df)),
            "activity_counts": {str(k): int(v) for k, v in sorted(counts.items())},
            "target_rows": int(sum(v for k, v in counts.items()
                                   if k in TARGET_ACTIVITIES)),
        }
        total_rows += len(df)
        nan_rows += n_nan
        print("  subject%d  %7d rows  %6.1f min  NaN %5.2f%%  %2d activities"
              % (s, len(df), dur / 60, per_subject[s]["nan_pct"], len(counts)))

    if not frames:
        raise SystemExit("FATAL: no subject files loaded from %s" % data_dir)

    # Reuse the exact label-map verification the accuracy script runs.
    print()
    label_report = verify_activity_ids(frames)

    # Window counts per class, using the same windowing as the accuracy run.
    print("Windowing all subjects (same parameters as the accuracy check)...")
    window_counts: Counter = Counter()
    windows_by_subject: dict[int, int] = {}
    for s, df in frames.items():
        w = segment_and_window(df, cfg)
        windows_by_subject[s] = len(w)
        window_counts.update(x["activity_id"] for x in w)
        print("   subject%d -> %5d windows" % (s, len(w)))

    # Raw traces, one per target activity, from the first subject that has it.
    print("\nExtracting %.1fs raw traces per activity..." % args.trace_seconds)
    for aid in ORDERED_IDS:
        for s, df in frames.items():
            t = signal_traces(df, aid, args.trace_seconds, hz)
            if t:
                traces[TARGET_ACTIVITIES[aid]] = {"subject": s, "channels": t}
                break

    global_counts: Counter = Counter()
    for s in per_subject:
        for k, v in per_subject[s]["activity_counts"].items():
            global_counts[int(k)] += v

    payload = {
        "source_dir": data_dir,
        "subjects": sorted(frames),
        "n_subjects": len(frames),
        "total_rows": total_rows,
        "total_duration_hours": sum(v["duration_s"] for v in per_subject.values()) / 3600,
        "nan_rows": nan_rows,
        "nan_pct": 100.0 * nan_rows / max(1, total_rows),
        "sampling_rate_hz": hz,
        "window_seconds": cfg["baseline"]["window_seconds"],
        "overlap": cfg["baseline"]["overlap"],
        "features_per_window": 45,
        "column_map": {"timestamp": 0, "activityID": 1, **NODE_COLS},
        "activity_names": {str(k): v for k, v in PAMAP2_ALL_ACTIVITIES.items()},
        "target_activities": {str(k): v for k, v in TARGET_ACTIVITIES.items()},
        "global_activity_counts": {str(k): int(v)
                                   for k, v in sorted(global_counts.items())},
        "per_subject": {str(k): v for k, v in per_subject.items()},
        "window_counts": {TARGET_ACTIVITIES[k]: int(v)
                          for k, v in sorted(window_counts.items())},
        "windows_by_subject": {str(k): v for k, v in windows_by_subject.items()},
        "total_windows": int(sum(window_counts.values())),
        "traces": traces,
        "trace_seconds": args.trace_seconds,
        "label_verification": label_report,
    }
    _write(out_path, payload)

    print("\n" + "=" * 78)
    print("total: %d rows, %.1f hours, %d windows across %d subjects"
          % (total_rows, payload["total_duration_hours"],
             payload["total_windows"], len(frames)))
    print("windows per class:")
    for name, n in payload["window_counts"].items():
        print("   %-20s %6d  %s" % (name, n, "#" * int(n / 200)))
    print("=" * 78)
    print("wrote %s" % out_path)
    return 0


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    mirror = os.path.join(REPO_ROOT, "frontend", "results", "phase1",
                          os.path.basename(path))
    os.makedirs(os.path.dirname(mirror), exist_ok=True)
    with open(mirror, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
