"""MHEALTH loader — verified 24-column layout, 50 Hz.

Layout (0-indexed)
------------------
    0,1,2     chest accel x,y,z        (m/s^2)
    3,4       ECG lead 1, lead 2       (mV)
    5,6,7     ankle accel x,y,z        (m/s^2)
    8,9,10    ankle gyro x,y,z         (deg/s)
    11,12,13  ankle magnetometer x,y,z
    14,15,16  wrist accel x,y,z        (m/s^2)
    17,18,19  wrist gyro x,y,z         (deg/s)
    20,21,22  wrist magnetometer x,y,z
    23        activity label

NON-UNIFORM NODES -- the defining property of this dataset
----------------------------------------------------------
    chest : accel + ECG            NO gyroscope, NO magnetometer
    ankle : accel + gyro + mag     full 9-axis
    wrist : accel + gyro + mag     full 9-axis

The chest node therefore reports `gyro_dps=None` and
`channels_present=["accel","ecg"]`. It is never zero-filled: a zero gyro
reading means "not rotating", while an absent gyro means "no sensor". See
core/datasource.py.

ECG is carried in the chest node's meta. It is a physiological channel, NOT a
motion channel, and must never be fed to a motion feature extractor.

NO TIMESTAMP COLUMN
-------------------
MHEALTH ships no time column. We synthesise t_sec from the row index at 50 Hz
(t = i / 50.0) and record `timestamp_derived: True` in every frame's meta.

This matters concretely for Phase 3's clock-desync injector: it must know it is
perturbing a *synthetic, perfectly regular* clock rather than a recorded one.
Desync applied to a synthetic clock cannot reproduce real jitter statistics, so
results from MHEALTH desync experiments are not comparable to PAMAP2's, whose
timestamps were actually measured.

Units
-----
Accelerations are already m/s^2 and are converted to g here (/ 9.80665).
Gyroscope values are already deg/s and need no conversion -- unlike PAMAP2,
which stores rad/s.
"""

from __future__ import annotations

import os

import pandas as pd

from core.datasource import STANDARD_GRAVITY

DATASET_NAME = "mhealth"
SAMPLING_RATE_HZ = 50.0
N_COLUMNS = 24

COL_LABEL = 23
COL_ECG = [3, 4]

NODE_IDS = ["wrist", "chest", "ankle"]
AXES = ("x", "y", "z")

ACCEL_COLS = {"chest": [0, 1, 2], "ankle": [5, 6, 7], "wrist": [14, 15, 16]}
GYRO_COLS = {"chest": None, "ankle": [8, 9, 10], "wrist": [17, 18, 19]}
MAG_COLS = {"chest": None, "ankle": [11, 12, 13], "wrist": [20, 21, 22]}

# Declared availability per node -- the source of truth downstream branches on.
NODE_CHANNELS = {
    "chest": ["accel", "ecg"],          # NO gyro, NO mag
    "ankle": ["accel", "gyro", "mag"],
    "wrist": ["accel", "gyro", "mag"],
}


def default_dir(repo_root: str) -> str:
    return os.path.join(repo_root, "data", "raw", "mhealth")


def subject_path(data_dir: str, subject: str | int) -> str:
    s = str(subject).replace("subject", "").replace("mHealth_", "")
    return os.path.join(data_dir, "mHealth_subject%s.log" % s)


def available_subjects(data_dir: str) -> list[str]:
    if not os.path.isdir(data_dir):
        return []
    out = []
    for f in os.listdir(data_dir):
        if f.startswith("mHealth_subject") and f.endswith(".log"):
            out.append("subject" + f[len("mHealth_subject"):-len(".log")])
    return sorted(out, key=lambda s: int(s.replace("subject", "")))


def load_subject(data_dir: str, subject: str | int) -> pd.DataFrame | None:
    """Load one subject with canonical column names and a DERIVED timestamp."""
    path = subject_path(data_dir, subject)
    if not os.path.isfile(path):
        return None

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != N_COLUMNS:
        raise ValueError("%s has %d columns, expected %d"
                         % (path, df.shape[1], N_COLUMNS))

    out = pd.DataFrame()
    # Synthetic clock: no timestamp column exists in this dataset.
    out["timestamp"] = df.index.to_numpy() / SAMPLING_RATE_HZ
    out["activityID"] = df[COL_LABEL].astype(int)
    out["ecg_lead1"] = df[COL_ECG[0]]
    out["ecg_lead2"] = df[COL_ECG[1]]

    for n in NODE_IDS:
        for i, a in enumerate(AXES):
            # m/s^2 -> g
            out["%s_acc_%s" % (n, a)] = df[ACCEL_COLS[n][i]] / STANDARD_GRAVITY
        if GYRO_COLS[n] is not None:
            for i, a in enumerate(AXES):
                # already deg/s -- no conversion
                out["%s_gyr_%s" % (n, a)] = df[GYRO_COLS[n][i]]
        if MAG_COLS[n] is not None:
            for i, a in enumerate(AXES):
                out["%s_mag_%s" % (n, a)] = df[MAG_COLS[n][i]]

    sid = str(subject)
    out["subject_id"] = sid if sid.startswith("subject") else "subject" + sid
    return out


def accel_columns() -> list[str]:
    return ["%s_acc_%s" % (n, a) for n in NODE_IDS for a in AXES]


def frame_meta(node_id: str, subject_id: str, ecg1=None, ecg2=None) -> dict:
    meta = {
        "subject_id": subject_id,
        "dataset_name": DATASET_NAME,
        "channels_present": list(NODE_CHANNELS[node_id]),
        # Phase 3's clock-desync injector MUST see this: the clock is synthetic.
        "timestamp_derived": True,
        "timestamp_source": "row_index / %.1f Hz" % SAMPLING_RATE_HZ,
    }
    if node_id == "chest":
        # Physiological channel, carried alongside motion -- never a motion input.
        meta["ecg"] = {
            "lead1": None if ecg1 is None else float(ecg1),
            "lead2": None if ecg2 is None else float(ecg2),
        }
    return meta
