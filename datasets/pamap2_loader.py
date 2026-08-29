"""PAMAP2 loader — the full 54-column map, verified in Phase 1.

Layout
------
    col 0        timestamp (s), ~100 Hz
    col 1        activityID
    col 2        heart rate (bpm), ~9 Hz and therefore SPARSE (NaN on most
                 rows). Carried into meta as-is; deliberately NOT interpolated,
                 because a fabricated heart-rate series would be a fabricated
                 physiological signal.
    cols 3-19    IMU hand  (mapped to canonical node "wrist")
    cols 20-36   IMU chest
    cols 37-53   IMU ankle

Each 17-column IMU block, relative to its start:
    +0            temperature (C)
    +1,+2,+3      accel, +/-16 g scale (m/s^2)   <- USED
    +4,+5,+6      accel, +/-6 g scale (m/s^2)    <- NOT used, but see below
    +7,+8,+9      gyroscope (rad/s)
    +10,+11,+12   magnetometer (uT)
    +13..+16      orientation -- INVALID per the dataset's own readme; never use

Why accel16 and not accel6 -- MEASURED, not assumed
---------------------------------------------------
The usual claim is that the +/-6 g channel "saturates during vigorous motion".
Measured on subject101 running (wrist, 21,007 rows) that is OVERSTATED: accel6
reports up to 6.33 g and only 0.10 % of its samples sit at or beyond the +/-6 g
rail, so it does not hard-clip and tracks accel16 closely.

The real reason to prefer accel16 is narrower but still valid: peak
accelerations genuinely exceed the 6 g rail (6.75 g observed on the same run),
so accel6 cannot represent the extremes. Those peaks are exactly the samples
that separate running from walking, and they matter more once degradation
compresses the signal. accel16 is the safe choice; the difference is small, not
dramatic.

Unit conversion (applied here so datasets and hardware agree downstream)
------------------------------------------------------------------------
    accel_g  = accel16_m_s2 / 9.80665
    gyro_dps = gyro_rad_s * 180 / pi

The ESP32 nodes planned for Phase 9 report g and deg/s natively, which is why
we normalise to those units rather than to SI: the hardware source then needs
no conversion layer and every consumer sees one unit system.

NaN policy
----------
Rows with NaN in any of the nine ACCELEROMETER columns are dropped, matching
Phase 1's verified behaviour (this is what reproduces the 9,909-window
reference figure). Gyroscope NaN is retained as NaN inside the tuple: the node
HAS a gyroscope, it merely dropped a sample. That is a different fact from
MHEALTH's chest node, which has no gyroscope at all and reports None.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from core.datasource import STANDARD_GRAVITY

DATASET_NAME = "pamap2"
SAMPLING_RATE_HZ = 100.0
N_COLUMNS = 54

COL_TIMESTAMP = 0
COL_ACTIVITY = 1
COL_HEART_RATE = 2

# IMU block starts. PAMAP2's "hand" IMU is worn on the dominant wrist; we
# normalise the name to the canonical "wrist".
IMU_BLOCK_START = {"wrist": 3, "chest": 20, "ankle": 37}
IMU_BLOCK_WIDTH = 17

# Offsets within a block.
OFF_TEMPERATURE = 0
OFF_ACCEL16 = (1, 2, 3)
OFF_ACCEL6 = (4, 5, 6)          # unused: saturates in vigorous motion
OFF_GYRO = (7, 8, 9)
OFF_MAG = (10, 11, 12)
OFF_ORIENTATION = (13, 14, 15, 16)   # invalid in this collection

NODE_IDS = ["wrist", "chest", "ankle"]
CHANNELS_PRESENT = ["accel", "gyro", "mag"]     # every PAMAP2 node is 9-axis
AXES = ("x", "y", "z")

RAD_TO_DEG = 180.0 / math.pi


def _cols(node: str, offsets) -> list[int]:
    b = IMU_BLOCK_START[node]
    return [b + o for o in offsets]


ACCEL_COLS = {n: _cols(n, OFF_ACCEL16) for n in NODE_IDS}
GYRO_COLS = {n: _cols(n, OFF_GYRO) for n in NODE_IDS}
MAG_COLS = {n: _cols(n, OFF_MAG) for n in NODE_IDS}


def default_dir(repo_root: str) -> str:
    return os.path.join(repo_root, "data", "raw", "pamap2", "Protocol")


def subject_path(data_dir: str, subject: str | int) -> str:
    s = str(subject)
    if not s.startswith("subject"):
        s = "subject" + s
    return os.path.join(data_dir, s + ".dat")


def available_subjects(data_dir: str) -> list[str]:
    if not os.path.isdir(data_dir):
        return []
    out = []
    for f in sorted(os.listdir(data_dir)):
        if f.startswith("subject") and f.endswith(".dat"):
            out.append(f[:-4])
    return sorted(out)


def load_subject(data_dir: str, subject: str | int) -> pd.DataFrame | None:
    """Load one subject into a tidy frame with canonical column names.

    Returns columns:
        timestamp, activityID, heart_rate,
        {node}_acc_{x,y,z}   (g)
        {node}_gyr_{x,y,z}   (deg/s)
        {node}_mag_{x,y,z}   (uT, carried but unused by the windower)
    """
    path = subject_path(data_dir, subject)
    if not os.path.isfile(path):
        return None

    usecols = [COL_TIMESTAMP, COL_ACTIVITY, COL_HEART_RATE]
    names = ["timestamp", "activityID", "heart_rate"]
    for n in NODE_IDS:
        usecols += ACCEL_COLS[n] + GYRO_COLS[n] + MAG_COLS[n]
        names += ["%s_acc_%s" % (n, a) for a in AXES]
        names += ["%s_gyr_%s" % (n, a) for a in AXES]
        names += ["%s_mag_%s" % (n, a) for a in AXES]

    order = sorted(range(len(usecols)), key=lambda i: usecols[i])
    sorted_cols = [usecols[i] for i in order]
    sorted_names = [names[i] for i in order]

    # PAMAP2 is single-space separated -> the fast C parser applies.
    try:
        df = pd.read_csv(path, sep=" ", header=None, usecols=sorted_cols,
                         names=sorted_names, na_values=["NaN"])
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", header=None, usecols=sorted_cols,
                         names=sorted_names, na_values=["NaN"], engine="python")

    # --- unit conversion, once, here -------------------------------------- #
    for n in NODE_IDS:
        for a in AXES:
            df["%s_acc_%s" % (n, a)] = df["%s_acc_%s" % (n, a)] / STANDARD_GRAVITY
            df["%s_gyr_%s" % (n, a)] = df["%s_gyr_%s" % (n, a)] * RAD_TO_DEG

    sid = str(subject)
    df["subject_id"] = sid if sid.startswith("subject") else "subject" + sid
    return df


def accel_columns() -> list[str]:
    """The nine accelerometer columns the windower requires to be non-NaN.

    Matches Phase 1 exactly -- gyro/mag NaN does not drop a row.
    """
    return ["%s_acc_%s" % (n, a) for n in NODE_IDS for a in AXES]


def frame_meta(subject_id: str, heart_rate) -> dict:
    hr = None
    if heart_rate is not None and not (isinstance(heart_rate, float)
                                       and np.isnan(heart_rate)):
        hr = float(heart_rate)
    return {
        "subject_id": subject_id,
        "dataset_name": DATASET_NAME,
        "channels_present": list(CHANNELS_PRESENT),
        "heart_rate": hr,          # sparse (~9 Hz); None on most rows, never
                                   # interpolated
    }
