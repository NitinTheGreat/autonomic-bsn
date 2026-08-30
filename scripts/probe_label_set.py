#!/usr/bin/env python3
"""PROBE: does dropping the two stair classes rescue Gate 2?

Gate 2 failed at 0.4722 against a 0.65 threshold. check_baseline_accuracy.py's
own data-driven diagnosis ranked fix (c) first -- drop to the 6-class set --
because 100 % of the top confusions involve stairs. This measures that instead
of assuming it.

It reuses check_baseline_accuracy.py's pipeline unchanged, overriding only the
label set, so the comparison is like-for-like: same windows, same prompt
structure, same few-shot policy, same subjects, same seed.

Does NOT modify the Phase 1 script or its published result.

Usage
-----
    python scripts/probe_label_set.py --n-windows 96
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import check_baseline_accuracy as B  # noqa: E402
from _llm_client import REPO_ROOT, describe_backend, load_config  # noqa: E402

# The 6-class set: PAMAP2_8 minus the two stair classes.
SIX = {1: "lying", 2: "sitting", 3: "standing", 4: "walking",
       5: "running", 6: "cycling"}


def apply_label_set(mapping: dict) -> None:
    """Repoint check_baseline_accuracy's module-level label constants."""
    ids = sorted(mapping, key=lambda i: list(mapping).index(i))
    letters = [chr(ord("A") + i) for i in range(len(ids))]
    B.TARGET_ACTIVITIES = dict(mapping)
    B.ORDERED_IDS = ids
    B.LETTERS = letters
    B.LETTER_TO_ID = dict(zip(letters, ids))
    B.ID_TO_LETTER = {v: k for k, v in B.LETTER_TO_ID.items()}
    B.LEGEND = "\n".join("%s = %s" % (l, mapping[i])
                         for l, i in zip(letters, ids))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-windows", type=int, default=96)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    out_path = args.out or os.path.join(REPO_ROOT, "results", "phase1",
                                        "label_set_probe.json")
    tmp = os.path.join(REPO_ROOT, "results", "phase1", "_probe_tmp.json")

    print("=" * 78)
    print("PROBE: 6-CLASS SET (stairs dropped) vs the failed 8-class Gate 2")
    print("=" * 78)
    print("  %s" % describe_backend(cfg, (args.backend or cfg["backend"])))
    print("  8-class Gate 2 result: 0.4722 (FAIL, threshold 0.65)\n")

    apply_label_set(SIX)
    argv = ["check_baseline_accuracy.py", "--n-windows", str(args.n_windows),
            "--out", tmp]
    if args.backend:
        argv += ["--backend", args.backend]
    old_argv = sys.argv
    sys.argv = argv
    try:
        B.main()
    finally:
        sys.argv = old_argv

    with open(tmp, "r", encoding="utf-8") as fh:
        six = json.load(fh)

    eight_path = os.path.join(REPO_ROOT, "results", "phase1",
                              "baseline_accuracy.json")
    eight = None
    if os.path.isfile(eight_path):
        with open(eight_path, "r", encoding="utf-8") as fh:
            eight = json.load(fh)

    payload = {
        "generated_by": "scripts/probe_label_set.py",
        "status": ("PROBE ONLY -- Phase 1's published 8-class Gate 2 result is "
                   "unchanged"),
        "eight_class": {
            "classes": 8,
            "overall_accuracy": eight["overall_accuracy"] if eight else None,
            "pass": eight["pass"] if eight else None,
            "per_class_accuracy": eight["per_class_accuracy"] if eight else None,
        },
        "six_class": {
            "classes": 6,
            "overall_accuracy": six["overall_accuracy"],
            "pass": six["overall_accuracy"] >= 0.65,
            "per_class_accuracy": six["per_class_accuracy"],
            "confusion_matrix": six["confusion_matrix"],
            "confusion_matrix_labels": six["confusion_matrix_labels"],
            "n_windows": six["n_windows"],
            "most_confused_pairs": six["most_confused_pairs"],
        },
        "threshold": 0.65,
        "chance_8": 1 / 8,
        "chance_6": 1 / 6,
        "model": six.get("model"),
        "backend": six.get("backend"),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    mirror = os.path.join(REPO_ROOT, "frontend", "results", "phase1",
                          os.path.basename(out_path))
    os.makedirs(os.path.dirname(mirror), exist_ok=True)
    with open(mirror, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    for p in (tmp, os.path.join(REPO_ROOT, "frontend", "results", "phase1",
                                os.path.basename(tmp))):
        if os.path.isfile(p):
            os.remove(p)

    print("\n" + "=" * 78)
    print("HEAD-TO-HEAD")
    print("=" * 78)
    e = payload["eight_class"]["overall_accuracy"]
    s = payload["six_class"]["overall_accuracy"]
    print("  8-class (chance 0.125): %.4f  %s"
          % (e, "PASS" if e and e >= 0.65 else "FAIL"))
    print("  6-class (chance 0.167): %.4f  %s"
          % (s, "PASS" if s >= 0.65 else "FAIL"))
    print("  delta                 : %+.4f" % (s - (e or 0)))
    print("\nwrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
