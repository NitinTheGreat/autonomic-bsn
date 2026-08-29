#!/usr/bin/env python3
"""Phase 2 DataSource test suite.

Runs against the REAL datasets on disk -- there are no mocks here, because the
things worth asserting (window counts, label integrity, absent channels) are
properties of the data as parsed, not of a fixture.

Makes NO LLM calls.

    python tests/test_datasource.py
    python tests/test_datasource.py --fast   # skip the full 9-subject count
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from core.datasource import (  # noqa: E402
    STANDARD_GRAVITY,
    DataSource,
    NodeFrame,
    Window,
    ms2_to_g,
    rad_s_to_dps,
)
from core.labels import LabelSetError, resolve_label_set  # noqa: E402
from datasets import mhealth_loader, pamap2_loader  # noqa: E402
from datasets.dataset_replay_source import (  # noqa: E402
    MAX_SAMPLE_GAP_S,
    PAMAP2_REFERENCE_WINDOWS,
    REFERENCE_TOLERANCE,
    DatasetReplaySource,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_fails: list[str] = []
_passes = 0


def check(name: str, ok: bool, extra: str = "") -> None:
    global _passes
    if ok:
        _passes += 1
        print("  PASS  %s%s" % (name, (" -- " + extra) if extra else ""))
    else:
        _fails.append(name)
        print("  FAIL  %s%s" % (name, (" -- " + extra) if extra else ""))


def take(it, n):
    out = []
    for i, x in enumerate(it):
        if i >= n:
            break
        out.append(x)
    return out


# --------------------------------------------------------------------------- #
def test_protocol_conformance():
    print("\n[1] DataSource protocol conformance")
    src = DatasetReplaySource("pamap2", subjects=["subject101"],
                              label_set="PAMAP2_8")
    check("DatasetReplaySource isinstance DataSource", isinstance(src, DataSource))
    check("exposes node_ids", isinstance(src.node_ids, list) and src.node_ids)
    check("exposes sampling_rate_hz",
          isinstance(src.sampling_rate_hz, float) and src.sampling_rate_hz > 0)
    check("windows() is callable and yields Window",
          isinstance(take(src.windows(), 1)[0], Window))

    w = take(src.windows(), 1)[0]
    f = w.frames["wrist"][0]
    check("yields NodeFrame objects", isinstance(f, NodeFrame))
    check("frames keyed by canonical node ids",
          sorted(w.frames) == ["ankle", "chest", "wrist"])
    check("NodeFrame.source == 'dataset'", f.source == "dataset")
    check("injected_failure stays None in Phase 2",
          all(fr.injected_failure is None
              for n in w.frames for fr in w.frames[n]),
          "Phase 3 writes it; the Phase 4 monitor must never read it")


def test_frame_counts():
    print("\n[2] frame count per window == rate * window_sec")
    for ds, subj, label_set, rate in [
        ("pamap2", "subject101", "PAMAP2_8", 100.0),
        ("mhealth", "subject1", "CANONICAL_6", 50.0),
    ]:
        src = DatasetReplaySource(ds, subjects=[subj], label_set=label_set)
        check("%s sampling_rate_hz == %.0f" % (ds, rate),
              src.sampling_rate_hz == rate)
        ws = take(src.windows(), 25)
        expected = rate * 2.56
        # >=60% coverage is the contract, so allow that band but no more.
        ok = all(0.6 * expected <= len(w.frames[n]) <= expected + 1
                 for w in ws for n in src.node_ids)
        counts = sorted({len(w.frames["wrist"]) for w in ws})
        check("%s frames within coverage band of %.0f" % (ds, expected), ok,
              "observed %s" % counts[:6])
        same = all(len({len(w.frames[n]) for n in src.node_ids}) == 1
                   for w in ws)
        check("%s all nodes have equal frame counts per window" % ds, same)


def test_no_straddle():
    print("\n[3] windows never straddle an activity change or a gap")
    for ds, subj, label_set in [("pamap2", "subject101", "PAMAP2_8"),
                                ("mhealth", "subject1", "CANONICAL_6")]:
        src = DatasetReplaySource(ds, subjects=[subj], label_set=label_set)
        ws = take(src.windows(), 200)

        check("%s every window carries exactly one label" % ds,
              all(w.label is not None for w in ws) and
              all(isinstance(w.label, str) for w in ws))

        # Activity change: rebuild each window's true labels from the raw frame
        # times and confirm they are homogeneous.
        loader = pamap2_loader if ds == "pamap2" else mhealth_loader
        df = loader.load_subject(loader.default_dir(REPO_ROOT), subj)
        ts = df["timestamp"].to_numpy()
        acts = df["activityID"].to_numpy()
        lookup = dict(zip(ts.round(4), acts))

        straddles = 0
        for w in ws:
            ids = {lookup.get(round(f.t_sec, 4))
                   for f in w.frames["wrist"]}
            ids.discard(None)
            if len(ids) > 1:
                straddles += 1
        check("%s no window spans two activityIDs" % ds, straddles == 0,
              "%d straddling of %d checked" % (straddles, len(ws)))

        # Recording gap: no intra-window time step exceeds the segment-break
        # threshold.
        bad = 0
        for w in ws:
            t = [f.t_sec for f in w.frames["wrist"]]
            if any(t[i + 1] - t[i] > MAX_SAMPLE_GAP_S for i in range(len(t) - 1)):
                bad += 1
        check("%s no window spans a recording gap (>%.2fs)" % (ds, MAX_SAMPLE_GAP_S),
              bad == 0, "%d gapped of %d checked" % (bad, len(ws)))

        # Frames must stay inside the declared window bounds.
        inside = all(w.start_sec - 1e-6 <= f.t_sec <= w.end_sec + 1e-6
                     for w in ws for f in w.frames["wrist"])
        check("%s frame times lie within [start_sec, end_sec]" % ds, inside)


def test_mhealth_absent_gyro():
    print("\n[4] MHEALTH chest: absent gyro is None, NEVER zero-filled")
    src = DatasetReplaySource("mhealth", subjects=["subject1"],
                              label_set="CANONICAL_6")
    ws = take(src.windows(), 40)
    chest = [f for w in ws for f in w.frames["chest"]]
    wrist = [f for w in ws for f in w.frames["wrist"]]
    ankle = [f for w in ws for f in w.frames["ankle"]]

    check("chest gyro_dps is None on every frame",
          all(f.gyro_dps is None for f in chest),
          "%d chest frames checked" % len(chest))
    # The distinction the whole project rests on.
    zero_tuples = sum(1 for f in chest if f.gyro_dps == (0.0, 0.0, 0.0))
    check("chest gyro is NOT a zero tuple", zero_tuples == 0,
          "a zero reading means 'not rotating'; None means 'no sensor'")
    check("chest channels_present == ['accel','ecg']",
          all(f.meta["channels_present"] == ["accel", "ecg"] for f in chest))
    check("chest carries ECG in meta",
          all("ecg" in f.meta for f in chest))
    check("wrist/ankle DO have gyro tuples",
          all(f.gyro_dps is not None and len(f.gyro_dps) == 3
              for f in wrist + ankle))
    check("wrist/ankle channels_present includes gyro",
          all("gyro" in f.meta["channels_present"] for f in wrist + ankle))
    check("MHEALTH timestamps flagged as derived",
          all(f.meta.get("timestamp_derived") is True for f in chest),
          "Phase 3's clock-desync injector must know the clock is synthetic")

    # PAMAP2, by contrast, has gyro on every node.
    p = DatasetReplaySource("pamap2", subjects=["subject101"],
                            label_set="PAMAP2_8")
    pw = take(p.windows(), 5)
    check("PAMAP2 chest DOES have a gyro tuple",
          all(f.gyro_dps is not None for w in pw for f in w.frames["chest"]))


def test_label_set_enforcement():
    print("\n[5] label-set enforcement")
    try:
        resolve_label_set("mhealth", "PAMAP2_8")
        check("PAMAP2_8 from MHEALTH raises", False, "no error raised")
    except LabelSetError as e:
        check("PAMAP2_8 from MHEALTH raises", True)
        check("error explains the stairs mismatch",
              "ascending" in str(e) and "climbing stairs" in str(e).lower())
        check("error warns against silently returning 6 classes",
              "silently" in str(e))

    try:
        DatasetReplaySource("mhealth", subjects=["subject1"],
                            label_set="PAMAP2_8")
        check("DatasetReplaySource refuses PAMAP2_8 for MHEALTH", False)
    except LabelSetError:
        check("DatasetReplaySource refuses PAMAP2_8 for MHEALTH", True)

    check("MHEALTH permits CANONICAL_6",
          len(resolve_label_set("mhealth", "CANONICAL_6")) == 6)
    check("PAMAP2 permits both",
          len(resolve_label_set("pamap2", "PAMAP2_8")) == 8 and
          len(resolve_label_set("pamap2", "CANONICAL_6")) == 6)


def test_unit_conversions():
    print("\n[6] unit conversions")
    check("9.80665 m/s^2 -> 1.0 g", abs(ms2_to_g(9.80665) - 1.0) < 1e-12)
    check("-19.6133 m/s^2 -> -2.0 g", abs(ms2_to_g(-19.6133) + 2.0) < 1e-9)
    check("pi rad/s -> 180 deg/s", abs(rad_s_to_dps(math.pi) - 180.0) < 1e-9)
    check("1 rad/s -> 57.29578 deg/s",
          abs(rad_s_to_dps(1.0) - 57.29577951308232) < 1e-9)

    # End-to-end against the real file: raw column 4 is wrist accel16 x.
    data_dir = pamap2_loader.default_dir(REPO_ROOT)
    path = pamap2_loader.subject_path(data_dir, "subject101")
    raw = pd.read_csv(path, sep=" ", header=None, nrows=200,
                      usecols=[4, 10], names=["a16x", "gx"], na_values=["NaN"])
    df = pamap2_loader.load_subject(data_dir, "subject101").head(200)

    i = raw["a16x"].first_valid_index()
    expected_g = raw["a16x"].iloc[i] / STANDARD_GRAVITY
    check("PAMAP2 accel16 col 4 -> wrist_acc_x in g",
          abs(df["wrist_acc_x"].iloc[i] - expected_g) < 1e-9,
          "raw %.5f m/s^2 -> %.5f g" % (raw["a16x"].iloc[i], expected_g))

    j = raw["gx"].first_valid_index()
    expected_dps = raw["gx"].iloc[j] * 180.0 / math.pi
    check("PAMAP2 gyro col 10 (rad/s) -> wrist_gyr_x in deg/s",
          abs(df["wrist_gyr_x"].iloc[j] - expected_dps) < 1e-9,
          "raw %.5f rad/s -> %.5f deg/s" % (raw["gx"].iloc[j], expected_dps))

    # MHEALTH gyro is ALREADY deg/s -- converting it again would be a bug.
    m_dir = mhealth_loader.default_dir(REPO_ROOT)
    m_raw = pd.read_csv(mhealth_loader.subject_path(m_dir, "subject1"),
                        sep=r"\s+", header=None, nrows=50, engine="python")
    m_df = mhealth_loader.load_subject(m_dir, "subject1").head(50)
    check("MHEALTH gyro passed through unconverted (already deg/s)",
          abs(m_df["ankle_gyr_x"].iloc[0] - m_raw[8].iloc[0]) < 1e-9)
    check("MHEALTH accel converted m/s^2 -> g",
          abs(m_df["chest_acc_x"].iloc[0]
              - m_raw[0].iloc[0] / STANDARD_GRAVITY) < 1e-9)


def test_subject109_empty():
    print("\n[7] subject109 yields 0 windows without raising")
    try:
        src = DatasetReplaySource("pamap2", subjects=["subject109"],
                                  label_set="PAMAP2_8")
        n = src.window_count()
        check("subject109 returns 0 windows, no exception", n == 0,
              "%d windows (8,477 rows across 2 activities)" % n)
    except Exception as e:
        check("subject109 returns 0 windows, no exception", False,
              "raised %s" % type(e).__name__)


def test_reference_window_count(fast: bool):
    print("\n[8] PAMAP2 reference window count vs Phase 1")
    if fast:
        print("  SKIPPED (--fast)")
        return
    from datasets.dataset_replay_source import check_reference_window_count
    r = check_reference_window_count()
    check("PAMAP2 reproduces Phase 1's %d windows within +/-%.0f%%"
          % (PAMAP2_REFERENCE_WINDOWS, REFERENCE_TOLERANCE * 100),
          r["within_tolerance"],
          "actual %d, delta %+d (%+.3f%%)"
          % (r["actual"], r["delta"], r["delta_pct"]))
    check("subject109 contributes 0 windows",
          r["per_subject"].get("subject109") == 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true",
                    help="skip the full 9-subject window count")
    args = ap.parse_args()

    print("=" * 74)
    print("PHASE 2 DATASOURCE TEST SUITE (real data, no LLM calls)")
    print("=" * 74)

    test_protocol_conformance()
    test_frame_counts()
    test_no_straddle()
    test_mhealth_absent_gyro()
    test_label_set_enforcement()
    test_unit_conversions()
    test_subject109_empty()
    test_reference_window_count(args.fast)

    print("\n" + "=" * 74)
    if _fails:
        print("%d PASSED, %d FAILED" % (_passes, len(_fails)))
        for f in _fails:
            print("   FAILED: %s" % f)
        return 1
    print("ALL %d ASSERTIONS PASSED" % _passes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
