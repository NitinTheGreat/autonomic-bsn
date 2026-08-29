#!/usr/bin/env python3
"""Verify the MHEALTH dataset is present and well-formed. VERIFICATION ONLY.

This is deliberately NOT a parser. No DataSource, no windowing, no feature
extraction -- all of that belongs to Phase 2. The job here is to answer, before
Phase 2 starts, whether the files are complete, correctly shaped, and carry the
activity labels we expect.

It reuses the tiered label-map warning scheme adopted for PAMAP2 in Phase 1:
    missing expected ID        -> prominent WARNING
    undocumented ID            -> prominent WARNING
    documented-but-excluded ID -> INFO
The naive "warn on any unexpected ID" rule is deliberately NOT reintroduced:
MHEALTH legitimately contains activities outside our 6-class set, and warning
on those every run would train the reader to ignore warnings.

Usage
-----
    python scripts/verify_mhealth.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _llm_client import REPO_ROOT  # noqa: E402

SAMPLING_RATE_HZ = 50           # MHEALTH is 50 Hz (PAMAP2 is 100 Hz)
N_COLUMNS = 24
N_SUBJECTS = 10
COL_LABEL = 23

# ----------------------------------------------------------------------------
# VERIFIED 24-column layout. Unlike PAMAP2, the nodes are NOT uniform:
# the chest node carries accelerometer + ECG only -- no gyro, no magnetometer.
# ----------------------------------------------------------------------------
COLUMN_MAP = {
    "chest_accel":  [0, 1, 2],
    "ecg":          [3, 4],          # lead 1, lead 2 -- no PAMAP2 equivalent
    "ankle_accel":  [5, 6, 7],
    "ankle_gyro":   [8, 9, 10],
    "ankle_mag":    [11, 12, 13],
    "wrist_accel":  [14, 15, 16],
    "wrist_gyro":   [17, 18, 19],
    "wrist_mag":    [20, 21, 22],
    "activity_label": [23],
}

NODE_CHANNELS = {
    "chest": {"accel": [0, 1, 2], "ecg": [3, 4]},          # NO gyro, NO mag
    "ankle": {"accel": [5, 6, 7], "gyro": [8, 9, 10], "mag": [11, 12, 13]},
    "wrist": {"accel": [14, 15, 16], "gyro": [17, 18, 19], "mag": [20, 21, 22]},
}

# Our 6-class canonical set, chosen to align with the PAMAP2 classes.
TARGET_ACTIVITIES = {
    1: "standing", 2: "sitting", 3: "lying",
    4: "walking", 9: "cycling", 11: "running",
}

# Full documented MHEALTH activity set.
MHEALTH_ALL_ACTIVITIES = {
    0: "null_transient",
    1: "standing", 2: "sitting", 3: "lying", 4: "walking",
    5: "climbing_stairs", 6: "waist_bends_forward", 7: "frontal_elevation_arms",
    8: "knees_bending_crouching", 9: "cycling", 10: "jogging", 11: "running",
    12: "jump_front_back",
}

# Notes surfaced in the report because they change what Phase 2 can compare.
EXCLUSION_NOTES = {
    5: ("MHEALTH id 5 is 'climbing stairs' with NO ascending/descending "
        "distinction, unlike PAMAP2's separate 12/13. It cannot be mapped onto "
        "the PAMAP2 stair classes and is excluded from the canonical set."),
    10: ("MHEALTH id 10 'jogging' is deliberately excluded to avoid conflating "
         "it with running (11); the two are distinct labels here."),
}


def resolve_dir(override: str | None) -> str:
    candidates = [override] if override else [
        os.path.join(REPO_ROOT, "data", "raw", "mhealth"),
        os.path.join(REPO_ROOT, "data", "raw", "MHEALTHDATASET"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.normpath(c)
    raise SystemExit(
        "FATAL: MHEALTH directory not found. Tried:\n  " +
        "\n  ".join(str(c) for c in candidates) +
        "\n\nDownload it (see data/raw/README.md):\n"
        "  curl -L -o mhealth.zip "
        "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip\n"
        "  unzip mhealth.zip && mv MHEALTHDATASET mhealth\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = resolve_dir(args.data_dir)
    out_path = args.out or os.path.join(
        REPO_ROOT, "results", "phase1", "mhealth_profile.json")

    print("=" * 78)
    print("MHEALTH DATASET VERIFICATION  (verification only -- no parser)")
    print("=" * 78)
    print("source: %s" % data_dir)
    print("expected: %d subjects, %d columns, %d Hz\n"
          % (N_SUBJECTS, N_COLUMNS, SAMPLING_RATE_HZ))

    problems: list[str] = []
    per_subject: dict[str, dict] = {}
    global_counts: Counter = Counter()

    # ---- 1 & 2: files exist, 24 columns each -------------------------------
    print("%-22s %8s %8s %10s %8s" % ("FILE", "ROWS", "COLS", "DURATION", "LABELS"))
    print("-" * 78)

    for i in range(1, N_SUBJECTS + 1):
        name = "mHealth_subject%d.log" % i
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            problems.append("MISSING FILE: %s" % name)
            print("%-22s %s" % (name, "*** MISSING ***"))
            continue

        # Tab-separated in the published files; \s+ tolerates either.
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        ncols = df.shape[1]
        if ncols != N_COLUMNS:
            problems.append("%s has %d columns, expected %d"
                            % (name, ncols, N_COLUMNS))

        # 3: rows and duration at 50 Hz
        rows = len(df)
        duration_s = rows / SAMPLING_RATE_HZ

        # 4: activity labels with counts
        counts = Counter(df[COL_LABEL].astype(int).tolist()) if ncols > COL_LABEL \
            else Counter()
        global_counts.update(counts)

        per_subject["subject%d" % i] = {
            "file": name,
            "rows": rows,
            "columns": ncols,
            "duration_s": duration_s,
            "duration_min": duration_s / 60,
            "activity_counts": {str(k): int(v) for k, v in sorted(counts.items())},
            "target_rows": int(sum(v for k, v in counts.items()
                                   if k in TARGET_ACTIVITIES)),
        }
        print("%-22s %8d %8d %8.1f m %8d"
              % (name, rows, ncols, duration_s / 60, len(counts)))

    if not per_subject:
        raise SystemExit("FATAL: no MHEALTH files could be read from " + data_dir)

    # ---- 4: per-subject label breakdown ------------------------------------
    print("\n" + "=" * 78)
    print("ACTIVITY LABELS PER SUBJECT (column %d)" % COL_LABEL)
    print("=" * 78)
    for sid, rec in per_subject.items():
        print("\n%s  (%d rows)" % (sid, rec["rows"]))
        for aid_s, n in rec["activity_counts"].items():
            aid = int(aid_s)
            nm = MHEALTH_ALL_ACTIVITIES.get(aid, "!! UNKNOWN ID !!")
            tag = "  <- target" if aid in TARGET_ACTIVITIES else ""
            print("    id %-3d %-26s %7d rows%s" % (aid, nm, n, tag))

    # ---- 5: tiered label verification --------------------------------------
    print("\n" + "=" * 78)
    print("LABEL-MAP VERIFICATION (same tiered scheme as PAMAP2)")
    print("=" * 78)

    warnings: list[str] = []
    info: list[str] = []

    missing = [i for i in sorted(TARGET_ACTIVITIES)
               if global_counts.get(i, 0) == 0]
    if missing:
        warnings.append(
            "MISSING expected activity IDs %s (%s) -- label map may be wrong."
            % (missing, [TARGET_ACTIVITIES[i] for i in missing]))

    thin = [i for i in sorted(TARGET_ACTIVITIES)
            if 0 < global_counts.get(i, 0) < 500]
    if thin:
        warnings.append(
            "Expected activity IDs %s have <500 rows total -- too thin." % thin)

    for aid, n in sorted(global_counts.items()):
        if aid in TARGET_ACTIVITIES or aid == 0:
            continue
        if aid in MHEALTH_ALL_ACTIVITIES:
            note = EXCLUSION_NOTES.get(aid)
            info.append("id %d (%s): %d rows -- documented MHEALTH activity, "
                        "excluded from the 6-class set by design.%s"
                        % (aid, MHEALTH_ALL_ACTIVITIES[aid], n,
                           ("\n        " + note) if note else ""))
        elif n >= 1000:
            warnings.append(
                "UNDOCUMENTED activity ID %d with %d rows -- not in the MHEALTH "
                "activity list at all. Label map is probably wrong." % (aid, n))

    if info:
        print("\nINFO (expected, not a problem):")
        for m in info:
            print("   - " + m)

    print("\nCANONICAL 6-CLASS SET:")
    for aid in sorted(TARGET_ACTIVITIES):
        print("   id %-3d %-12s %8d rows"
              % (aid, TARGET_ACTIVITIES[aid], global_counts.get(aid, 0)))

    if warnings or problems:
        print("\n" + "!" * 78)
        print("!! PROBLEMS FOUND -- do not silently proceed")
        print("!" * 78)
        for m in problems + warnings:
            print("   *** " + m)
    else:
        print("\nOK: all 10 files present at %d columns; all 6 expected "
              "activity IDs present; no undocumented IDs." % N_COLUMNS)

    # ---- schema warning that Phase 2 must honour ---------------------------
    print("\n" + "=" * 78)
    print("SCHEMA NOTE FOR PHASE 2 -- non-uniform nodes")
    print("=" * 78)
    print("  chest : accel (0,1,2) + ECG (3,4)      -- NO gyro, NO magnetometer")
    print("  ankle : accel + gyro + mag  (5..13)    -- full 9-axis")
    print("  wrist : accel + gyro + mag  (14..22)   -- full 9-axis")
    print()
    print("  Unlike PAMAP2, whose chest node is a full 9-axis IMU, MHEALTH's")
    print("  chest node has no rotational or magnetic channels. Phase 2's")
    print("  feature extractor MUST handle per-node channel availability")
    print("  rather than assume a uniform schema, and MUST NOT zero-fill the")
    print("  missing channels -- a zero gyro reading is not the same as an")
    print("  absent sensor, and conflating them would corrupt the degradation")
    print("  study this project exists to run.")

    payload = {
        "dataset": "MHEALTH",
        "source_dir": data_dir,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "expected_columns": N_COLUMNS,
        "n_subjects_found": len(per_subject),
        "all_files_present": len(per_subject) == N_SUBJECTS,
        "all_files_24_columns": all(r["columns"] == N_COLUMNS
                                    for r in per_subject.values()),
        "total_rows": sum(r["rows"] for r in per_subject.values()),
        "total_duration_hours": sum(r["duration_s"] for r in per_subject.values()) / 3600,
        "column_map": COLUMN_MAP,
        "node_channels": NODE_CHANNELS,
        "chest_has_gyro": False,
        "chest_has_magnetometer": False,
        "schema_note": ("MHEALTH's chest node carries accel + ECG only -- no "
                        "gyro, no magnetometer -- unlike PAMAP2's full 9-axis "
                        "chest IMU. Phase 2 must handle per-node channel "
                        "availability and must not zero-fill missing channels."),
        "activity_names": {str(k): v for k, v in MHEALTH_ALL_ACTIVITIES.items()},
        "target_activities": {str(k): v for k, v in TARGET_ACTIVITIES.items()},
        "exclusion_notes": {str(k): v for k, v in EXCLUSION_NOTES.items()},
        "global_activity_counts": {str(k): int(v)
                                   for k, v in sorted(global_counts.items())},
        "per_subject": per_subject,
        "label_verification": {"warnings": warnings, "info": info,
                               "problems": problems,
                               "ok": not (warnings or problems)},
    }
    _write(out_path, payload)

    print("\n" + "=" * 78)
    print("total: %d rows, %.1f hours across %d subjects"
          % (payload["total_rows"], payload["total_duration_hours"],
             len(per_subject)))
    print("wrote %s" % out_path)
    print("=" * 78)
    return 0 if payload["label_verification"]["ok"] else 1


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
