#!/usr/bin/env python3
"""Export a SMALL set of representative windows for the frontend explorer.

Not every window -- one clean window per activity from one representative
subject per dataset. Enough for the explorer to render real waveforms without
shipping the dataset into the repo.

Writes:
    frontend/data/phase2_samples/<dataset>_<subject>_<activity>.json
    frontend/data/phase2_samples/manifest.json

The manifest is what populates the page's dropdowns, so the UI reflects real
availability rather than a hardcoded list -- if an activity failed to export,
it simply is not offered.

Each sample carries per-node accel x/y/z traces, gyro where the node has one,
the label, and channels_present. A node without a gyroscope exports
`"gyro": null` -- never zeros -- so the frontend can show an explicit
"no gyroscope on this node" notice.

Makes NO LLM calls.

Usage
-----
    python scripts/export_phase2_samples.py
    python scripts/export_phase2_samples.py --subject-pamap2 subject105
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import mhealth_loader, pamap2_loader  # noqa: E402
from datasets.dataset_replay_source import DatasetReplaySource  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "frontend", "data", "phase2_samples")

SPECS = {
    "pamap2": {"loader": pamap2_loader, "label_set": "PAMAP2_8",
               "default_subject": "subject101"},
    "mhealth": {"loader": mhealth_loader, "label_set": "CANONICAL_6",
                "default_subject": "subject1"},
}


def _clean(v: float):
    """JSON has no NaN. A dropped sample exports as null, never as 0."""
    return None if (v is None or math.isnan(v)) else round(float(v), 5)


def window_to_payload(w, dataset: str, subject: str) -> dict:
    nodes = {}
    for node_id, frames in w.frames.items():
        if not frames:
            continue
        f0 = frames[0]
        has_gyro = f0.gyro_dps is not None
        nodes[node_id] = {
            "channels_present": list(f0.meta.get("channels_present", [])),
            "has_gyro": has_gyro,
            "t": [round(f.t_sec - w.start_sec, 4) for f in frames],
            "accel_g": {
                "x": [_clean(f.accel_g[0]) for f in frames],
                "y": [_clean(f.accel_g[1]) for f in frames],
                "z": [_clean(f.accel_g[2]) for f in frames],
            },
            # null, not zeros -- absence must stay visible downstream
            "gyro_dps": None if not has_gyro else {
                "x": [_clean(f.gyro_dps[0]) for f in frames],
                "y": [_clean(f.gyro_dps[1]) for f in frames],
                "z": [_clean(f.gyro_dps[2]) for f in frames],
            },
        }
    return {
        "dataset": dataset,
        "subject": subject,
        "activity": w.label,
        "start_sec": round(w.start_sec, 4),
        "end_sec": round(w.end_sec, 4),
        "n_samples": len(next(iter(w.frames.values()))),
        "units": {"accel": "g", "gyro": "deg/s"},
        "nodes": nodes,
    }


def export_dataset(dataset: str, subject: str | None) -> list[dict]:
    spec = SPECS[dataset]
    loader = spec["loader"]
    data_dir = loader.default_dir(REPO_ROOT)
    subject = subject or spec["default_subject"]

    src = DatasetReplaySource(dataset, subjects=[subject],
                              label_set=spec["label_set"], data_dir=data_dir)

    print("  %s / %s -> looking for one window per activity"
          % (dataset, subject))

    # Pick the MEDIAN-MOTION window per class, not an arbitrary positional one.
    #
    # Taking the middle of the list looked reasonable but is not: for
    # descending_stairs on subject101 it landed on the single quietest window
    # of 113 (ankle sd 0.0076 against a class median of 0.711), which renders
    # as a near-flat line and misrepresents the activity. Ranking by motion
    # energy and taking the median guarantees a representative trace.
    by_label: dict[str, list] = {}
    for w in src.windows():
        by_label.setdefault(w.label, []).append(w)

    entries = []
    for label in src.classes:
        got = by_label.get(label)
        if not got:
            print("     %-20s no windows -- not exported" % label)
            continue

        def motion(win) -> float:
            """Mean per-axis std of ankle accel -- a simple activity-energy proxy."""
            vals = []
            for ax in range(3):
                col = [f.accel_g[ax] for f in win.frames["ankle"]
                       if not math.isnan(f.accel_g[ax])]
                if len(col) > 1:
                    m = sum(col) / len(col)
                    vals.append((sum((c - m) ** 2 for c in col) / len(col)) ** 0.5)
            return sum(vals) / len(vals) if vals else 0.0

        ranked = sorted(got, key=motion)
        w = ranked[len(ranked) // 2]
        payload = window_to_payload(w, dataset, subject)
        fname = "%s_%s_%s.json" % (dataset, subject, label)
        with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        size_kb = os.path.getsize(os.path.join(OUT_DIR, fname)) / 1024
        nodes_wo_gyro = [n for n, d in payload["nodes"].items()
                         if not d["has_gyro"]]
        entries.append({
            "dataset": dataset, "subject": subject, "activity": label,
            "file": fname, "n_samples": payload["n_samples"],
            "nodes": sorted(payload["nodes"]),
            "nodes_without_gyro": nodes_wo_gyro,
        })
        print("     %-20s %4d samples  %5.1f KB%s"
              % (label, payload["n_samples"], size_kb,
                 ("  (no gyro: %s)" % ",".join(nodes_wo_gyro))
                 if nodes_wo_gyro else ""))
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject-pamap2", default=None)
    ap.add_argument("--subject-mhealth", default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print("EXPORTING PHASE 2 SAMPLE WINDOWS")
    print("=" * 78)
    print("out: %s\n" % OUT_DIR)

    entries = []
    entries += export_dataset("pamap2", args.subject_pamap2)
    entries += export_dataset("mhealth", args.subject_mhealth)

    datasets: dict[str, dict] = {}
    for e in entries:
        d = datasets.setdefault(e["dataset"], {"subjects": {}})
        s = d["subjects"].setdefault(e["subject"], {"activities": []})
        s["activities"].append({"activity": e["activity"], "file": e["file"],
                                "n_samples": e["n_samples"],
                                "nodes": e["nodes"],
                                "nodes_without_gyro": e["nodes_without_gyro"]})

    manifest = {
        "generated_by": "scripts/export_phase2_samples.py",
        "phase": 2,
        "units": {"accel": "g", "gyro": "deg/s"},
        "note": ("A node with no gyroscope exports gyro_dps: null -- never "
                 "zeros. MHEALTH's chest node has accel + ECG only."),
        "datasets": datasets,
        "n_samples_exported": len(entries),
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    total_kb = sum(os.path.getsize(os.path.join(OUT_DIR, f))
                   for f in os.listdir(OUT_DIR)) / 1024
    print("\n%d samples exported, %.1f KB total" % (len(entries), total_kb))
    print("wrote %s" % os.path.join(OUT_DIR, "manifest.json"))
    return 0 if entries else 1


if __name__ == "__main__":
    sys.exit(main())
