#!/usr/bin/env python3
"""Cross-check both datasets' activity IDs against core/labels.py.

Uses the SAME tiered warning scheme adopted in Phase 1:

    missing expected ID                  -> WARNING
    undocumented ID with real volume     -> WARNING
    documented-but-deliberately-excluded -> INFO (with its reason)

The naive "warn on any unexpected ID" rule is deliberately NOT reintroduced.
Both datasets legitimately contain activities outside our label sets -- PAMAP2
ids 7/16/17/24, MHEALTH ids 5/6/7/8/10/12 -- and warning on those every run
would train the reader to ignore warnings, which is worse than not warning.

Usage
-----
    python scripts/verify_labels.py
    python scripts/verify_labels.py --dataset pamap2
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.labels import (  # noqa: E402
    ALL_ACTIVITIES,
    EXCLUDED_IDS,
    ID_MAPS,
    NULL_IDS,
)
from datasets import mhealth_loader, pamap2_loader  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOADERS = {"pamap2": pamap2_loader, "mhealth": mhealth_loader}

UNDOCUMENTED_WARN_ROWS = 1000


def verify(dataset: str) -> dict:
    loader = LOADERS[dataset]
    data_dir = loader.default_dir(REPO_ROOT)
    subjects = loader.available_subjects(data_dir)

    print("=" * 78)
    print("LABEL VERIFICATION -- %s" % dataset.upper())
    print("=" * 78)
    print("source   : %s" % data_dir)
    print("subjects : %d" % len(subjects))
    if not subjects:
        print("\n*** NO FILES FOUND -- see data/raw/README.md")
        return {"dataset": dataset, "ok": False,
                "warnings": ["no subject files found in %s" % data_dir]}

    id_map = ID_MAPS[dataset]
    all_acts = ALL_ACTIVITIES[dataset]
    excluded = EXCLUDED_IDS[dataset]
    null_id = NULL_IDS[dataset]

    per_subject: dict[str, dict[str, int]] = {}
    global_counts: Counter = Counter()

    for s in subjects:
        df = loader.load_subject(data_dir, s)
        if df is None:
            continue
        counts = Counter(df["activityID"].astype(int).tolist())
        per_subject[s] = {str(k): int(v) for k, v in sorted(counts.items())}
        global_counts.update(counts)

        print("\n%s  (%d rows)" % (s, len(df)))
        for aid in sorted(counts):
            nm = all_acts.get(aid, "!! UNKNOWN ID !!")
            if aid in id_map:
                tag = "  <- mapped to '%s'" % id_map[aid]
            elif aid == null_id:
                tag = "  (null/transient, dropped)"
            elif aid in excluded:
                tag = "  (excluded by design)"
            else:
                tag = "  *** UNDOCUMENTED ***"
            print("    id %-3d %-26s %8d rows%s"
                  % (aid, nm, counts[aid], tag))

    # ---- tiered verification --------------------------------------------- #
    warnings: list[str] = []
    info: list[str] = []

    missing = [i for i in sorted(id_map) if global_counts.get(i, 0) == 0]
    if missing:
        warnings.append(
            "MISSING expected activity IDs %s (%s) -- the label map may be "
            "wrong or the wrong files were loaded."
            % (missing, [id_map[i] for i in missing]))

    thin = [i for i in sorted(id_map) if 0 < global_counts.get(i, 0) < 500]
    if thin:
        warnings.append(
            "Expected activity IDs %s have <500 rows total -- too thin to "
            "window reliably." % thin)

    for aid, n in sorted(global_counts.items()):
        if aid in id_map or aid == null_id:
            continue
        if aid in excluded:
            name, reason = excluded[aid]
            info.append("id %d (%s): %d rows -- excluded by design.\n"
                        "        %s" % (aid, name, n, reason))
        elif aid in all_acts:
            info.append("id %d (%s): %d rows -- documented activity outside "
                        "our label sets." % (aid, all_acts[aid], n))
        elif n >= UNDOCUMENTED_WARN_ROWS:
            warnings.append(
                "UNDOCUMENTED activity ID %d with %d rows -- not in the %s "
                "activity list at all. Label map is probably wrong."
                % (aid, n, dataset.upper()))

    print("\n" + "-" * 78)
    if info:
        print("INFO (expected, not a problem):")
        for m in info:
            print("   - " + m)

    print("\nMAPPED CLASSES:")
    for aid in sorted(id_map):
        print("   id %-3d %-20s %9d rows"
              % (aid, id_map[aid], global_counts.get(aid, 0)))

    if warnings:
        print("\n" + "!" * 78)
        print("!! LABEL-MAP WARNING -- do not silently proceed")
        print("!" * 78)
        for m in warnings:
            print("   *** " + m)
    else:
        print("\nOK: all mapped activity IDs present; no undocumented IDs.")
    print("-" * 78 + "\n")

    return {"dataset": dataset, "ok": not warnings, "warnings": warnings,
            "info": info, "per_subject": per_subject,
            "global_counts": {str(k): int(v)
                              for k, v in sorted(global_counts.items())}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(LOADERS) + ["all"],
                    default="all")
    args = ap.parse_args()

    targets = sorted(LOADERS) if args.dataset == "all" else [args.dataset]
    results = [verify(d) for d in targets]

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in results:
        print("  %-10s %s" % (r["dataset"], "OK" if r["ok"] else "WARNINGS"))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
