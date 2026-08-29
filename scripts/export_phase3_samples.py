#!/usr/bin/env python3
"""Export PAIRED {clean, injected} traces for the Phase 3 frontend.

For a handful of representative combinations only -- not the whole sweep.
Each file carries the clean trace, the injected trace, the requested and
REALISED effect magnitudes, and the ground-truth annotation.

Sample selection reuses Phase 2's motion-energy ranking: the positional middle
of a class was found there to pick the quietest window (ankle sd 0.0076 against
a class median of 0.711), which renders as a near-flat line and misrepresents
the activity.

NaN is exported as JSON null so the frontend can draw a real GAP. It must never
be coerced to 0 -- a dropout drawn as a flat line at zero would visually
contradict the exact distinction this phase exists to preserve.

Makes NO LLM calls.

Usage
-----
    python scripts/export_phase3_samples.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_replay_source import DatasetReplaySource  # noqa: E402
from injection.registry import DESCRIPTIONS, FAILURE_TYPES, make_injector  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "frontend", "data", "phase3_samples")

SEED = 20260829

# Kept deliberately small: a few activities that span the motion range.
PLANS = {
    "pamap2": {"label_set": "PAMAP2_8", "subject": "subject101",
               "activities": ["walking", "running", "sitting"],
               "target": "ankle"},
    "mhealth": {"label_set": "CANONICAL_6", "subject": "subject1",
                "activities": ["walking", "running"],
                "target": "ankle"},
}
SEVERITIES = (1, 2, 3, 4)


def jnum(v):
    """NaN -> null so the frontend can render a gap. Never 0."""
    return None if (v is None or math.isnan(v)) else round(float(v), 5)


def node_payload(frames) -> dict:
    has_gyro = frames[0].gyro_dps is not None
    return {
        "has_gyro": has_gyro,
        "channels_present": list(frames[0].meta.get("channels_present", [])),
        "t": [round(f.t_sec, 4) for f in frames],
        "accel_g": {ax: [jnum(f.accel_g[i]) for f in frames]
                    for i, ax in enumerate(("x", "y", "z"))},
        "gyro_dps": None if not has_gyro else {
            ax: [jnum(f.gyro_dps[i]) for f in frames]
            for i, ax in enumerate(("x", "y", "z"))},
    }


class _SingleWindowSource:
    """A DataSource that replays exactly one already-materialised Window.

    Lets us wrap an injector around a chosen window without re-parsing the
    subject file for every (failure x severity) combination -- the naive
    version reloaded ~376k rows 100 times and did not finish.

    It satisfies the same Protocol as DatasetReplaySource, which is the point
    of the contract: the injector cannot tell the difference.
    """

    def __init__(self, window, node_ids, sampling_rate_hz):
        self._window = window
        self.node_ids = list(node_ids)
        self.sampling_rate_hz = float(sampling_rate_hz)

    def windows(self, window_sec: float = 2.56, overlap: float = 0.5):
        yield copy.deepcopy(self._window)


def motion_energy(w, node: str) -> float:
    """Mean per-axis std of the node's accel -- Phase 2's ranking."""
    vals = []
    for ax in range(3):
        col = [f.accel_g[ax] for f in w.frames[node]
               if not math.isnan(f.accel_g[ax])]
        if len(col) > 1:
            m = sum(col) / len(col)
            vals.append((sum((c - m) ** 2 for c in col) / len(col)) ** 0.5)
    return sum(vals) / len(vals) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 78)
    print("EXPORTING PHASE 3 PAIRED SAMPLES")
    print("=" * 78)
    print("out: %s\n" % args.out_dir)

    entries = []
    for dataset, plan in PLANS.items():
        target = plan["target"]

        # Parse the subject ONCE, then reuse the materialised windows.
        src = DatasetReplaySource(dataset, subjects=[plan["subject"]],
                                  label_set=plan["label_set"])
        wanted = set(plan["activities"])
        by_label: dict[str, list] = {}
        for idx, w in enumerate(src.windows()):
            if w.label in wanted:
                by_label.setdefault(w.label, []).append((idx, w))

        for activity in plan["activities"]:
            pool = by_label.get(activity)
            if not pool:
                print("  %s/%s: no windows -- skipped" % (dataset, activity))
                continue
            ranked = sorted(pool, key=lambda p: motion_energy(p[1], target))
            _, clean_w = ranked[len(ranked) // 2]
            one = _SingleWindowSource(clean_w, src.node_ids,
                                      src.sampling_rate_hz)

            for ftype in FAILURE_TYPES:
                for sev in SEVERITIES:
                    inj = make_injector(one, ftype, sev, target, SEED)
                    injected_w = next(iter(inj.windows()), None)
                    if injected_w is None:
                        continue

                    payload = {
                        "dataset": dataset,
                        "subject": plan["subject"],
                        "activity": activity,
                        "failure_type": ftype,
                        "failure_description": DESCRIPTIONS[ftype],
                        "severity": sev,
                        "target_node": target,
                        "seed": SEED,
                        "units": {"accel": "g", "gyro": "deg/s"},
                        "window": {"start_sec": round(clean_w.start_sec, 4),
                                   "end_sec": round(clean_w.end_sec, 4)},
                        # ground truth -- for display and for Phase 4 scoring,
                        # never for the monitor to read
                        "ground_truth": {
                            "tag": injected_w.meta["tag"],
                            "requested": injected_w.meta["requested"],
                            "realised": injected_w.meta["realised"],
                        },
                        # Untouched nodes are bit-identical to `clean` by
                        # construction (asserted in tests), so storing them
                        # twice proves nothing and tripled the payload. Only
                        # the target node's after-trace is kept; the frontend
                        # falls back to `clean` for every other node.
                        "clean": {n: node_payload(clean_w.frames[n])
                                  for n in clean_w.frames},
                        "injected": {target: node_payload(
                            injected_w.frames[target])},
                        "injected_nodes": [target],
                    }
                    fname = "%s_%s_%s_sev%d.json" % (dataset, activity, ftype, sev)
                    with open(os.path.join(args.out_dir, fname), "w",
                              encoding="utf-8") as fh:
                        json.dump(payload, fh, separators=(",", ":"))
                    entries.append({
                        "dataset": dataset, "activity": activity,
                        "failure_type": ftype, "severity": sev,
                        "target_node": target, "file": fname,
                        "realised": injected_w.meta["realised"],
                    })
            print("  %-8s %-10s -> %d files (%d failures x %d severities)"
                  % (dataset, activity, len(FAILURE_TYPES) * len(SEVERITIES),
                     len(FAILURE_TYPES), len(SEVERITIES)))

    # Manifest drives the dropdowns from real availability.
    tree: dict = {}
    for e in entries:
        d = tree.setdefault(e["dataset"], {"activities": {}})
        a = d["activities"].setdefault(e["activity"],
                                       {"target_node": e["target_node"],
                                        "failures": {}})
        f = a["failures"].setdefault(e["failure_type"], {"severities": []})
        f["severities"].append({"severity": e["severity"], "file": e["file"]})

    manifest = {
        "generated_by": "scripts/export_phase3_samples.py",
        "phase": 3,
        "seed": SEED,
        "failure_types": FAILURE_TYPES,
        "descriptions": DESCRIPTIONS,
        "units": {"accel": "g", "gyro": "deg/s"},
        "note": ("NaN is exported as null so it renders as a GAP. It is never "
                 "coerced to 0: zero is a real stationary reading, and a "
                 "dropout drawn flat at zero would contradict the distinction "
                 "this phase preserves."),
        "datasets": tree,
        "n_files": len(entries),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    total_kb = sum(os.path.getsize(os.path.join(args.out_dir, f))
                   for f in os.listdir(args.out_dir)) / 1024
    print("\n%d paired samples, %.1f KB total" % (len(entries), total_kb))
    print("wrote %s" % os.path.join(args.out_dir, "manifest.json"))
    return 0 if entries else 1


if __name__ == "__main__":
    sys.exit(main())
