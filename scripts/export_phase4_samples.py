#!/usr/bin/env python3
"""Export monitor verdicts for the Phase 3/4 frontend. NO LLM calls.

Produces frontend/data/phase4_samples/verdicts.json, keyed by the SAME
filenames the Phase 3 exporter produced. That lets phase3_injection.html show
the waveform and the monitor's verdict side by side without re-running Python
-- the single most convincing view in the demo.

The monitor sees only BlindWindows here, exactly as in the evaluation.
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
from health.diagnose import diagnose_window  # noqa: E402
from health.window_view import blind  # noqa: E402
from injection.registry import FAILURE_TYPES, make_injector  # noqa: E402
from scripts.export_phase3_samples import (  # noqa: E402
    PLANS,
    SEED,
    _SingleWindowSource,
    motion_energy,
)
from scripts.run_detection_eval import SPLIT, calibrate_references  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "frontend", "data", "phase4_samples")
SEVERITIES = (1, 2, 3, 4)


def slim(verdict: dict) -> dict:
    """Everything the UI needs, without the full signal dump."""
    s = verdict["signals"]
    return {
        "node": verdict["node"],
        "diagnosis": verdict["diagnosis"],
        "state": verdict["state"],
        "trust_weight": verdict["trust_weight"],
        "min_sub_score": verdict["min_sub_score"],
        "geometric_mean_sub_score": verdict["geometric_mean_sub_score"],
        "sub_scores": verdict["sub_scores"],
        "evidence": verdict["evidence"],
        "channels_present": verdict["channels_present"],
        "gyro_available": verdict["gyro_available"],
        "key_signals": {
            "nan_fraction": s.get("nan_fraction"),
            "nan_terminal": s.get("nan_terminal"),
            "nan_run_max": s.get("nan_run_max"),
            "repeat_fraction": s.get("repeat_fraction"),
            "unique_value_ratio": s.get("unique_value_ratio"),
            "max_repeat_run": s.get("max_repeat_run"),
            "effective_unique_rate_hz": s.get("effective_unique_rate_hz"),
            "cross_node_offset_ms": s.get("cross_node_offset_ms"),
            "mean_accel_magnitude_g": s.get("mean_accel_magnitude_g"),
        },
        "displacement": {
            k: verdict["displacement"].get(k)
            for k in ("available", "angle_deg", "threshold_deg",
                      "estimated_alignment", "observability",
                      "natural_spread_deg", "swamped_by_natural_variation")
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 78)
    print("EXPORTING PHASE 4 MONITOR VERDICTS")
    print("=" * 78)

    verdicts: dict = {}
    refs_used: dict = {}

    for dataset, plan in PLANS.items():
        target = plan["target"]
        spec = SPLIT[dataset]
        refs = calibrate_references(dataset, spec["calibration_subject"],
                                    spec["label_set"], 60)
        refs_used[dataset] = {
            "calibration_subject": spec["calibration_subject"],
            "natural_spread_p95_deg": {n: r["clean_angle_p95"]
                                       for n, r in refs.items()},
        }

        src = DatasetReplaySource(dataset, subjects=[plan["subject"]],
                                  label_set=plan["label_set"])
        wanted = set(plan["activities"])
        by_label: dict = {}
        for w in src.windows():
            if w.label in wanted:
                by_label.setdefault(w.label, []).append(w)

        for activity in plan["activities"]:
            pool = by_label.get(activity)
            if not pool:
                continue
            ranked = sorted(pool, key=lambda x: motion_energy(x, target))
            clean_w = ranked[len(ranked) // 2]
            one = _SingleWindowSource(clean_w, src.node_ids,
                                      src.sampling_rate_hz)

            # the clean baseline verdict for this window
            base_out = diagnose_window(blind(clean_w), refs)
            verdicts["%s_%s_CLEAN.json" % (dataset, activity)] = {
                "dataset": dataset, "activity": activity, "failure": None,
                "nodes": {n: slim(v) for n, v in base_out["nodes"].items()},
                "network_state": base_out["network_state"],
                "flagged_nodes": base_out["flagged_nodes"],
            }

            for ft in FAILURE_TYPES:
                for sev in SEVERITIES:
                    inj = make_injector(one, ft, sev, target, SEED)
                    w = next(iter(inj.windows()), None)
                    if w is None:
                        continue
                    out = diagnose_window(blind(w), refs)
                    key = "%s_%s_%s_sev%d.json" % (dataset, activity, ft, sev)
                    verdicts[key] = {
                        "dataset": dataset, "activity": activity,
                        "failure": ft, "severity": sev, "target_node": target,
                        "nodes": {n: slim(v) for n, v in out["nodes"].items()},
                        "network_state": out["network_state"],
                        "flagged_nodes": out["flagged_nodes"],
                        # ground truth, for DISPLAY ONLY -- the monitor never
                        # saw it
                        "truth": {"tag": w.meta.get("tag"),
                                  "realised": w.meta.get("realised", {})},
                    }
            print("  %-8s %-10s -> %d verdicts"
                  % (dataset, activity, len(FAILURE_TYPES) * len(SEVERITIES) + 1))

    payload = {
        "generated_by": "scripts/export_phase4_samples.py",
        "phase": 4,
        "seed": SEED,
        "calibration": refs_used,
        "note": ("Verdicts are keyed by the Phase 3 sample filenames. The "
                 "monitor received only BlindWindows; the 'truth' block is "
                 "carried for display and was never visible to it."),
        "verdicts": verdicts,
    }
    path = os.path.join(args.out_dir, "verdicts.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print("\n%d verdicts, %.1f KB" % (len(verdicts),
                                      os.path.getsize(path) / 1024))
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
