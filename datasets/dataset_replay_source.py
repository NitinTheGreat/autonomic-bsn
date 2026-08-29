"""DatasetReplaySource — replays a recorded dataset through the DataSource contract.

This is the reference implementation of `core.datasource.DataSource`. Phase 9's
HardwareLiveSource must satisfy the same Protocol, so nothing downstream can
tell a replayed dataset from live nodes except by `NodeFrame.source`.

Windowing contract — carried forward from Phase 1 UNCHANGED
-----------------------------------------------------------
Deliberately not re-derived, because Phase 1's figures are the reference:

  * 2.56 s windows, 50 % overlap (step 1.28 s)
  * windows NEVER straddle an activity change or a recording gap -- they are
    cut only inside contiguous single-activity segments. A window spanning an
    activity change would carry a meaningless label; one spanning a gap would
    have fabricated statistics.
  * a window is kept only if it holds >= 60 % of the expected sample count
  * the canonical label comes from the dataset's ID map

NaN policy matches Phase 1: a row is dropped when any ACCELEROMETER channel is
NaN. Gyro NaN does not drop the row -- the node has a gyroscope, it merely
dropped a sample, and that tuple carries NaN rather than None. None is reserved
for "this node has no gyroscope at all".

REFERENCE CHECK
---------------
PAMAP2 across all 9 subjects at PAMAP2_8 must reproduce ~9,909 windows (Phase
1's verified figure). `check_reference_window_count()` asserts within +/-1 %. A
mismatch means this rebuilt parser diverged from verified behaviour and must be
investigated, not accepted.

subject109 yields 0 windows (only 8,477 rows across 2 activities). That is
expected and is NOT an error.
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from core.datasource import NodeFrame, Window
from core.labels import id_map_for, resolve_label_set
from datasets import mhealth_loader, pamap2_loader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOADERS = {"pamap2": pamap2_loader, "mhealth": mhealth_loader}

# Phase 1's verified figure, all 9 subjects, 8-class set.
PAMAP2_REFERENCE_WINDOWS = 9909
REFERENCE_TOLERANCE = 0.01          # +/- 1 %

DEFAULT_WINDOW_SEC = 2.56
DEFAULT_OVERLAP = 0.5
MIN_COVERAGE = 0.6                  # keep window if >= 60 % of expected samples
MAX_SAMPLE_GAP_S = 0.1              # larger gap -> segment break

# subject109 has only 8,477 rows across 2 activities -> 0 windows, expected.
KNOWN_EMPTY_SUBJECTS = {"pamap2": {"subject109"}}


class DatasetReplaySource:
    """Replays a recorded dataset as Windows of NodeFrames."""

    def __init__(self, dataset: str, subjects: Optional[list[str]] = None,
                 label_set: str = "PAMAP2_8",
                 data_dir: Optional[str] = None):
        self.dataset = dataset.lower()
        if self.dataset not in LOADERS:
            raise ValueError("Unknown dataset %r. Known: %s"
                             % (dataset, sorted(LOADERS)))
        self._loader = LOADERS[self.dataset]

        # Raises if this dataset cannot express the requested label set.
        self.label_set = label_set
        self.classes = resolve_label_set(self.dataset, label_set)
        self._id_map = id_map_for(self.dataset, label_set)

        self.data_dir = data_dir or self._loader.default_dir(REPO_ROOT)
        available = self._loader.available_subjects(self.data_dir)
        if not available:
            raise FileNotFoundError(
                "No %s subject files found in %s. See data/raw/README.md for "
                "download commands." % (self.dataset, self.data_dir))

        self.subjects = list(subjects) if subjects else available
        unknown = [s for s in self.subjects if s not in available]
        if unknown:
            raise ValueError("Unknown %s subjects %s. Available: %s"
                             % (self.dataset, unknown, available))

        # --- DataSource protocol attributes ---
        self.node_ids: list[str] = list(self._loader.NODE_IDS)
        self.sampling_rate_hz: float = float(self._loader.SAMPLING_RATE_HZ)

    # ----------------------------------------------------------------- API --
    def windows(self, window_sec: float = DEFAULT_WINDOW_SEC,
                overlap: float = DEFAULT_OVERLAP) -> Iterator[Window]:
        for subject in self.subjects:
            df = self._loader.load_subject(self.data_dir, subject)
            if df is None:
                continue
            yield from self._windows_for_subject(df, window_sec, overlap)

    def window_count(self, window_sec: float = DEFAULT_WINDOW_SEC,
                     overlap: float = DEFAULT_OVERLAP) -> int:
        return sum(1 for _ in self.windows(window_sec, overlap))

    def windows_from_frame(self, df: pd.DataFrame,
                           window_sec: float = DEFAULT_WINDOW_SEC,
                           overlap: float = DEFAULT_OVERLAP) -> Iterator[Window]:
        """Window an already-loaded frame.

        Lets callers that need raw stats too (profilers, exporters) load each
        subject once instead of paying the parse cost twice.
        """
        yield from self._windows_for_subject(df, window_sec, overlap)

    # ------------------------------------------------------------ internals --
    def _windows_for_subject(self, df: pd.DataFrame, window_sec: float,
                             overlap: float) -> Iterator[Window]:
        step_s = window_sec * (1.0 - overlap)
        expected_n = int(round(window_sec * self.sampling_rate_hz))
        min_n = int(round(expected_n * MIN_COVERAGE))

        accel_cols = self._loader.accel_columns()

        d = df[df["activityID"].isin(self._id_map)].copy()   # drops null id 0
        d = d.dropna(subset=accel_cols)                      # accel-NaN only
        if d.empty:
            return
        d = d.sort_values("timestamp")

        act = d["activityID"].to_numpy()
        ts = d["timestamp"].to_numpy()
        brk = np.empty(len(d), dtype=bool)
        brk[0] = True
        # Segment break on activity change OR recording gap.
        brk[1:] = (act[1:] != act[:-1]) | (np.diff(ts) > MAX_SAMPLE_GAP_S)
        d["_seg"] = np.cumsum(brk)

        for _, seg in d.groupby("_seg", sort=False):
            t = seg["timestamp"].to_numpy()
            if t[-1] - t[0] < window_sec:
                continue
            aid = int(seg["activityID"].iloc[0])
            label = self._id_map.get(aid)
            if label is None:
                continue
            start = t[0]
            while start + window_sec <= t[-1] + 1e-9:
                lo = np.searchsorted(t, start, "left")
                hi = np.searchsorted(t, start + window_sec, "left")
                if hi - lo >= min_n:
                    yield self._build_window(seg.iloc[lo:hi], float(start),
                                             float(start + window_sec), label)
                start += step_s

    def _build_window(self, block: pd.DataFrame, start_sec: float,
                      end_sec: float, label: str) -> Window:
        frames: dict[str, list[NodeFrame]] = {n: [] for n in self.node_ids}
        subject_id = str(block["subject_id"].iloc[0])
        times = block["timestamp"].to_numpy()

        is_pamap2 = self.dataset == "pamap2"
        hr = block["heart_rate"].to_numpy() if is_pamap2 else None
        ecg1 = block["ecg_lead1"].to_numpy() if not is_pamap2 else None
        ecg2 = block["ecg_lead2"].to_numpy() if not is_pamap2 else None

        acc = {n: block[["%s_acc_%s" % (n, a) for a in ("x", "y", "z")]]
               .to_numpy() for n in self.node_ids}
        gyr = {}
        for n in self.node_ids:
            cols = ["%s_gyr_%s" % (n, a) for a in ("x", "y", "z")]
            gyr[n] = block[cols].to_numpy() if cols[0] in block.columns else None

        for n in self.node_ids:
            a_n, g_n = acc[n], gyr[n]
            for i in range(len(block)):
                if is_pamap2:
                    meta = pamap2_loader.frame_meta(subject_id, hr[i])
                else:
                    meta = mhealth_loader.frame_meta(
                        n, subject_id,
                        None if ecg1 is None else ecg1[i],
                        None if ecg2 is None else ecg2[i])
                # None ONLY when the node has no gyroscope at all. A present
                # gyro that dropped a sample keeps a tuple containing NaN.
                gyro = None if g_n is None else (
                    float(g_n[i][0]), float(g_n[i][1]), float(g_n[i][2]))
                frames[n].append(NodeFrame(
                    node_id=n,
                    t_sec=float(times[i]),
                    accel_g=(float(a_n[i][0]), float(a_n[i][1]),
                             float(a_n[i][2])),
                    gyro_dps=gyro,
                    source="dataset",
                    injected_failure=None,   # Phase 3 writes this, never here
                    meta=meta,
                ))
        return Window(start_sec=start_sec, end_sec=end_sec,
                      frames=frames, label=label)


def check_reference_window_count(data_dir: Optional[str] = None) -> dict:
    """Assert PAMAP2/PAMAP2_8 reproduces Phase 1's verified 9,909 windows."""
    src = DatasetReplaySource("pamap2", label_set="PAMAP2_8", data_dir=data_dir)
    per_subject = {}
    total = 0
    for s in src.subjects:
        one = DatasetReplaySource("pamap2", subjects=[s], label_set="PAMAP2_8",
                                  data_dir=data_dir)
        n = one.window_count()
        per_subject[s] = n
        total += n

    lo = PAMAP2_REFERENCE_WINDOWS * (1 - REFERENCE_TOLERANCE)
    hi = PAMAP2_REFERENCE_WINDOWS * (1 + REFERENCE_TOLERANCE)
    within = lo <= total <= hi
    return {
        "actual": total,
        "reference": PAMAP2_REFERENCE_WINDOWS,
        "tolerance_pct": REFERENCE_TOLERANCE * 100,
        "delta": total - PAMAP2_REFERENCE_WINDOWS,
        "delta_pct": 100.0 * (total - PAMAP2_REFERENCE_WINDOWS)
        / PAMAP2_REFERENCE_WINDOWS,
        "within_tolerance": within,
        "per_subject": per_subject,
        "known_empty_subjects": sorted(KNOWN_EMPTY_SUBJECTS.get("pamap2", [])),
    }
