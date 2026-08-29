#!/usr/bin/env python3
"""Phase 1, Step 2 -- is zero-shot LLM activity recognition good enough on CLEAN data?

We need healthy-network accuracy high enough to leave headroom for measuring
degradation. If the model is already near chance on pristine sensors, any later
"confidently wrong under degradation" result is unattributable.

PASS threshold: overall accuracy >= 0.65 over 8 classes (chance = 0.125).

DISPOSABLE PARSER WARNING
-------------------------
The PAMAP2 reading code here is deliberately throwaway. Only the *verified
column indices* carry forward. Phase 2 rebuilds this properly behind DataSource.

Usage
-----
    python scripts/check_baseline_accuracy.py
    python scripts/check_baseline_accuracy.py --n-windows 40   # quick smoke run
    python scripts/check_baseline_accuracy.py --verify-labels-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _llm_client import (  # noqa: E402
    REPO_ROOT,
    LogprobError,
    confidence_method_for,
    describe_backend,
    load_config,
    resolve_backends,
    score_labels,
)

# --------------------------------------------------------------------------- #
# PAMAP2 column map (VERIFIED -- this is the part that carries into Phase 2)
#
# Layout: col0 timestamp, col1 activityID, col2 heart-rate, then three 17-column
# IMU blocks at 3 (hand/wrist), 20 (chest), 37 (ankle). Within a block:
#   +0 temperature | +1..+3 accel 16g | +4..+6 accel 6g | +7..+9 gyro
#   +10..+12 magnetometer | +13..+16 orientation (invalid in this collection)
# So accel16 = block_start + 1 .. block_start + 3.
# --------------------------------------------------------------------------- #
COL_TIMESTAMP = 0
COL_ACTIVITY = 1
NODE_COLS = {
    "wrist": [4, 5, 6],       # IMU hand block @3  -> 4,5,6
    "chest": [21, 22, 23],    # IMU chest block @20 -> 21,22,23
    "ankle": [38, 39, 40],    # IMU ankle block @37 -> 38,39,40
}
AXES = ["x", "y", "z"]
STATS = ["mean", "std", "min", "max", "energy"]

# The 8 classes under test.
TARGET_ACTIVITIES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 12: "ascending_stairs", 13: "descending_stairs",
}

# Full documented PAMAP2 activity set. Used to tell "an ID we deliberately
# exclude" apart from "an ID that means our label map is WRONG". Without this
# distinction the Step-2.2 check would cry wolf on every correct dataset,
# because the Protocol files legitimately contain 7/16/17/24.
PAMAP2_ALL_ACTIVITIES = {
    0: "transient", 1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "nordic_walking", 9: "watching_tv",
    10: "computer_work", 11: "car_driving", 12: "ascending_stairs",
    13: "descending_stairs", 16: "vacuum_cleaning", 17: "ironing",
    18: "folding_laundry", 19: "house_cleaning", 20: "playing_soccer",
    24: "rope_jumping",
}

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
ORDERED_IDS = [1, 2, 3, 4, 5, 6, 12, 13]
LETTER_TO_ID = dict(zip(LETTERS, ORDERED_IDS))
ID_TO_LETTER = {v: k for k, v in LETTER_TO_ID.items()}

LEGEND = "\n".join(
    "%s = %s" % (l, TARGET_ACTIVITIES[i]) for l, i in zip(LETTERS, ORDERED_IDS))

PROMPT_TEMPLATE = """You are an expert at recognising human activity from body-worn inertial sensors.

Each window is 2.56 seconds of tri-axial accelerometer data (units m/s^2, gravity included) from three body-worn nodes: wrist, chest and ankle. For each node and axis you are given mean, standard deviation, minimum, maximum and energy (mean of squares).

Legend:
{legend}

{few_shot}Now classify this window.

{query}

Answer with exactly one letter from the legend.
Answer:"""

EXAMPLE_BLOCK = """{features}
Answer: {letter}

"""


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def resolve_data_dir(cfg: dict, override: str | None) -> str:
    candidates = [override] if override else cfg["data"]["pamap2_protocol_dirs"]
    tried = []
    for c in candidates:
        p = c if os.path.isabs(c) else os.path.join(REPO_ROOT, c)
        p = os.path.normpath(p)
        tried.append(p)
        if os.path.isdir(p):
            return p
    raise SystemExit(
        "FATAL: PAMAP2 Protocol directory not found.\nTried:\n  " +
        "\n  ".join(tried) +
        "\n\nDownload it (see data/raw/README.md):\n"
        "  curl -L -o PAMAP2.zip https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip\n"
    )


def load_subject(data_dir: str, subject: int) -> pd.DataFrame | None:
    path = os.path.join(data_dir, "subject%d.dat" % subject)
    if not os.path.isfile(path):
        return None
    usecols = [COL_TIMESTAMP, COL_ACTIVITY]
    for cols in NODE_COLS.values():
        usecols.extend(cols)
    names = ["timestamp", "activityID"]
    for node in NODE_COLS:
        names.extend(["%s_%s" % (node, a) for a in AXES])
    # PAMAP2 is single-space separated, so the fast C parser handles it (~1.6 s
    # per subject). Fall back to the regex/python engine only if that fails,
    # since a regex separator is ~10x slower over 376k rows.
    try:
        df = pd.read_csv(path, sep=" ", header=None, usecols=usecols,
                         names=names, na_values=["NaN"])
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", header=None, usecols=usecols,
                         names=names, na_values=["NaN"], engine="python")
    df["subject"] = subject
    return df


# --------------------------------------------------------------------------- #
# Step 2.2 -- activity ID verification (must run BEFORE anything else)
# --------------------------------------------------------------------------- #
def verify_activity_ids(frames: dict[int, pd.DataFrame],
                        unexpected_warn_rows: int = 1000) -> dict:
    print("=" * 78)
    print("STEP 2.2 -- activityID verification (label-map sanity check)")
    print("=" * 78)

    per_subject: dict[int, dict[int, int]] = {}
    global_counts: Counter = Counter()

    for subj in sorted(frames):
        counts = Counter(frames[subj]["activityID"].astype(int).tolist())
        per_subject[subj] = dict(sorted(counts.items()))
        global_counts.update(counts)
        print("\nsubject%d  (%d rows)" % (subj, len(frames[subj])))
        for aid in sorted(counts):
            name = PAMAP2_ALL_ACTIVITIES.get(aid, "!! UNKNOWN ID !!")
            tag = "  <- target" if aid in TARGET_ACTIVITIES else ""
            print("    id %-3d %-20s %8d rows%s" % (aid, name, counts[aid], tag))

    warnings: list[str] = []
    info: list[str] = []

    # (a) every expected ID must be present somewhere
    missing = [i for i in ORDERED_IDS if global_counts.get(i, 0) == 0]
    if missing:
        warnings.append(
            "MISSING expected activityIDs %s -- the label map may be wrong or "
            "the wrong files were loaded." % missing)

    # (b) expected but tiny -> also suspicious
    thin = [i for i in ORDERED_IDS if 0 < global_counts.get(i, 0) < 500]
    if thin:
        warnings.append(
            "Expected activityIDs %s have <500 rows total -- too thin to "
            "sample windows from reliably." % thin)

    # (c) unexpected IDs: known-PAMAP2 ones are excluded by design (info only);
    #     genuinely unknown ones with real volume mean the map is wrong (warn).
    for aid, n in sorted(global_counts.items()):
        if aid in TARGET_ACTIVITIES or aid == 0:
            continue
        if aid in PAMAP2_ALL_ACTIVITIES:
            info.append("id %d (%s): %d rows -- documented PAMAP2 activity, "
                        "excluded from the 8-class set by design."
                        % (aid, PAMAP2_ALL_ACTIVITIES[aid], n))
        elif n >= unexpected_warn_rows:
            warnings.append(
                "UNDOCUMENTED activityID %d with %d rows -- not in the PAMAP2 "
                "activity list at all. Label map is probably wrong; do NOT "
                "proceed." % (aid, n))

    print("\n" + "-" * 78)
    if info:
        print("INFO (expected, not a problem):")
        for m in info:
            print("   - " + m)
    if warnings:
        print("\n" + "!" * 78)
        print("!! LABEL-MAP WARNING -- do not silently proceed")
        print("!" * 78)
        for m in warnings:
            print("   *** " + m)
    else:
        print("OK: all 8 expected activityIDs present; no undocumented IDs.")
    print("-" * 78 + "\n")

    return {"per_subject_counts": per_subject,
            "global_counts": dict(sorted(global_counts.items())),
            "warnings": warnings, "info": info,
            "ok": not warnings}


# --------------------------------------------------------------------------- #
# windowing + features
# --------------------------------------------------------------------------- #
def segment_and_window(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """2.56 s windows, 50% overlap, never straddling an activity or a time gap.

    Windows are cut inside contiguous single-activity segments only. A window
    spanning an activity change would carry a meaningless label, and one
    spanning a recording gap would have fabricated statistics.
    """
    b = cfg["baseline"]
    win_s = float(b["window_seconds"])
    step_s = win_s * (1.0 - float(b["overlap"]))
    max_gap = float(b["max_sample_gap_s"])
    expected_n = int(round(win_s * float(b["sampling_rate_hz"])))
    min_n = int(round(expected_n * float(b["min_window_coverage"])))

    feat_cols = [c for c in df.columns
                 if c not in ("timestamp", "activityID", "subject")]

    d = df[df["activityID"].isin(TARGET_ACTIVITIES)].copy()   # drops id 0 too
    d = d.dropna(subset=feat_cols)                            # drop NaN accel
    if d.empty:
        return []
    d = d.sort_values("timestamp")

    # segment breaks: activity change OR time gap
    act = d["activityID"].to_numpy()
    ts = d["timestamp"].to_numpy()
    brk = np.empty(len(d), dtype=bool)
    brk[0] = True
    brk[1:] = (act[1:] != act[:-1]) | (np.diff(ts) > max_gap)
    d["_seg"] = np.cumsum(brk)

    windows: list[dict] = []
    subject = int(df["subject"].iloc[0])

    for _, seg in d.groupby("_seg", sort=False):
        t = seg["timestamp"].to_numpy()
        if t[-1] - t[0] < win_s:
            continue
        vals = seg[feat_cols].to_numpy()
        aid = int(seg["activityID"].iloc[0])
        start = t[0]
        while start + win_s <= t[-1] + 1e-9:
            lo = np.searchsorted(t, start, "left")
            hi = np.searchsorted(t, start + win_s, "left")
            if hi - lo >= min_n:
                windows.append({
                    "subject": subject, "activity_id": aid,
                    "t_start": float(start),
                    "n_samples": int(hi - lo),
                    "features": compute_features(vals[lo:hi], feat_cols),
                })
            start += step_s
    return windows


def compute_features(block: np.ndarray, feat_cols: list[str]) -> dict:
    """3 nodes x 3 axes x 5 stats = 45 features."""
    out: dict[str, float] = {}
    for j, col in enumerate(feat_cols):
        v = block[:, j]
        out[col + "_mean"] = float(np.mean(v))
        out[col + "_std"] = float(np.std(v))
        out[col + "_min"] = float(np.min(v))
        out[col + "_max"] = float(np.max(v))
        out[col + "_energy"] = float(np.mean(v ** 2))
    return out


def render_features(feats: dict) -> str:
    """Compact natural-language summary -- one line per node."""
    lines = []
    for node in NODE_COLS:
        parts = []
        for a in AXES:
            k = "%s_%s" % (node, a)
            parts.append(
                "%s mean %.2f std %.2f min %.2f max %.2f energy %.2f"
                % (a, feats[k + "_mean"], feats[k + "_std"], feats[k + "_min"],
                   feats[k + "_max"], feats[k + "_energy"]))
        lines.append("%-6s: %s" % (node, " | ".join(parts)))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def stratified_sample(windows: list[dict], n_total: int, rng: np.random.Generator,
                      spread_subjects: bool = True) -> list[dict]:
    """Even split across the 8 classes, spread across subjects within a class."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for w in windows:
        by_class[w["activity_id"]].append(w)

    per_class = max(1, n_total // len(ORDERED_IDS))
    picked: list[dict] = []
    for aid in ORDERED_IDS:
        pool = by_class.get(aid, [])
        if not pool:
            continue
        if spread_subjects:
            by_subj: dict[int, list[dict]] = defaultdict(list)
            for w in pool:
                by_subj[w["subject"]].append(w)
            for s in by_subj:
                rng.shuffle(by_subj[s])
            # round-robin across subjects so no class comes from one person
            chosen, subs, i = [], sorted(by_subj), 0
            while len(chosen) < per_class and any(by_subj[s] for s in subs):
                s = subs[i % len(subs)]
                if by_subj[s]:
                    chosen.append(by_subj[s].pop())
                i += 1
            picked.extend(chosen)
        else:
            idx = rng.permutation(len(pool))[:per_class]
            picked.extend(pool[i] for i in idx)
    rng.shuffle(picked)
    return picked


def build_few_shot(windows: list[dict], k_per_class: int,
                   rng: np.random.Generator) -> tuple[str, list[dict]]:
    by_class: dict[int, list[dict]] = defaultdict(list)
    for w in windows:
        by_class[w["activity_id"]].append(w)
    block, used = "", []
    for aid in ORDERED_IDS:
        pool = by_class.get(aid, [])
        if not pool:
            print("   WARNING: no few-shot examples available for %s"
                  % TARGET_ACTIVITIES[aid])
            continue
        idx = rng.permutation(len(pool))[:k_per_class]
        for i in idx:
            w = pool[i]
            block += EXAMPLE_BLOCK.format(features=render_features(w["features"]),
                                          letter=ID_TO_LETTER[aid])
            used.append({"subject": w["subject"], "activity_id": aid,
                         "t_start": w["t_start"]})
    if block:
        block = ("Here are labelled examples.\n\n" + block)
    return block, used


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def confusion(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    idx = {a: i for i, a in enumerate(ORDERED_IDS)}
    m = [[0] * len(ORDERED_IDS) for _ in ORDERED_IDS]
    for t, p in zip(y_true, y_pred):
        m[idx[t]][idx[p]] += 1
    return m


def most_confused(m: list[list[int]], top: int = 3) -> list[dict]:
    pairs = []
    for i, ti in enumerate(ORDERED_IDS):
        for j, tj in enumerate(ORDERED_IDS):
            if i != j and m[i][j] > 0:
                pairs.append({
                    "true": TARGET_ACTIVITIES[ti],
                    "predicted": TARGET_ACTIVITIES[tj],
                    "count": m[i][j],
                    "rate_of_true_class": (m[i][j] / sum(m[i])) if sum(m[i]) else 0.0,
                })
    pairs.sort(key=lambda d: -d["count"])
    return pairs[:top]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-windows", type=int, default=None)
    ap.add_argument("--verify-labels-only", action="store_true",
                    help="run Step 2.2 only; no LLM calls, no data needed beyond files")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    b = cfg["baseline"]
    rng = np.random.default_rng(int(b["seed"]))
    n_windows = args.n_windows or int(b["n_windows"])
    threshold = float(b["pass_threshold"])
    out_path = args.out or os.path.join(
        REPO_ROOT, "results", "phase1", "baseline_accuracy.json")

    data_dir = resolve_data_dir(cfg, args.data_dir)
    print("PAMAP2 Protocol dir: %s\n" % data_dir)

    test_subjects = list(b["test_subjects"])
    fewshot_subjects = list(b["fewshot_subjects"])
    overlap = set(test_subjects) & set(fewshot_subjects)
    if overlap:
        raise SystemExit("FATAL: few-shot and test subjects overlap: %s. "
                         "Few-shot examples must come from OTHER subjects."
                         % sorted(overlap))

    # ---- load ------------------------------------------------------------- #
    frames: dict[int, pd.DataFrame] = {}
    for s in sorted(set(test_subjects) | set(fewshot_subjects)):
        df = load_subject(data_dir, s)
        if df is None:
            print("   note: subject%d.dat not found -- skipping" % s)
            continue
        frames[s] = df
    if not frames:
        raise SystemExit("FATAL: no subject files loaded from %s" % data_dir)

    # ---- Step 2.2 --------------------------------------------------------- #
    label_report = verify_activity_ids(frames)
    if args.verify_labels_only:
        return 0 if label_report["ok"] else 1

    # ---- windows ---------------------------------------------------------- #
    print("Windowing (%.2fs, %.0f%% overlap, single-activity segments only)..."
          % (float(b["window_seconds"]), float(b["overlap"]) * 100))
    test_windows, fs_windows = [], []
    for s, df in frames.items():
        w = segment_and_window(df, cfg)
        (test_windows if s in test_subjects else fs_windows).extend(w)
        print("   subject%d -> %d windows" % (s, len(w)))

    if not test_windows:
        raise SystemExit("FATAL: no test windows produced.")

    few_shot, fs_used = build_few_shot(fs_windows, int(b["few_shot_per_class"]), rng)
    sample = stratified_sample(test_windows, n_windows, rng)
    print("\nfew-shot examples: %d (from subjects %s)"
          % (len(fs_used), sorted({d["subject"] for d in fs_used})))
    print("test windows     : %d (from subjects %s)\n"
          % (len(sample), sorted({w["subject"] for w in sample})))

    # ---- classify --------------------------------------------------------- #
    try:
        backend = resolve_backends(cfg, args.backend)[0]
    except LogprobError as exc:
        print("FATAL: %s" % exc)
        return 1
    n_probs = int(cfg["request"].get("step2_n_probs", 20))
    conf_method = confidence_method_for(cfg, backend)
    print("resolved: %s" % describe_backend(cfg, backend))
    print("Classifying (argmax over the 8 label tokens, not free-text)...")

    y_true, y_pred, per_window = [], [], []
    for n, w in enumerate(sample, 1):
        prompt = PROMPT_TEMPLATE.format(legend=LEGEND, few_shot=few_shot,
                                        query=render_features(w["features"]))
        try:
            res = score_labels(cfg, backend, prompt, LETTERS, n_probs)
        except LogprobError as exc:
            print("\nFATAL during classification: %s" % exc)
            return 1
        pred = LETTER_TO_ID[res["argmax"]]
        y_true.append(w["activity_id"])
        y_pred.append(pred)
        per_window.append({
            "subject": w["subject"], "t_start": w["t_start"],
            "true": TARGET_ACTIVITIES[w["activity_id"]],
            "predicted": TARGET_ACTIVITIES[pred],
            "confidence": res["distribution"][res["argmax"]],
            "distribution": res["distribution"],
        })
        if n % 10 == 0 or n == len(sample):
            acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
            print("   %3d/%d  running accuracy %.3f" % (n, len(sample), acc))

    # ---- metrics ---------------------------------------------------------- #
    overall = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    m = confusion(y_true, y_pred)
    per_class = {}
    for i, aid in enumerate(ORDERED_IDS):
        tot = sum(m[i])
        per_class[TARGET_ACTIVITIES[aid]] = (m[i][i] / tot) if tot else None

    passed = overall >= threshold
    confused = most_confused(m, 3)

    payload = {
        "overall_accuracy": overall,
        "per_class_accuracy": per_class,
        "confusion_matrix": m,
        "confusion_matrix_labels": [TARGET_ACTIVITIES[a] for a in ORDERED_IDS],
        "n_windows": len(sample),
        "subjects_used": sorted({w["subject"] for w in sample}),
        "fewshot_subjects_used": sorted({d["subject"] for d in fs_used}),
        "model": cfg["backends"][backend].get("model", backend),
        "backend": backend,
        "confidence_method": conf_method,
        "pass": passed,
        "pass_threshold": threshold,
        "prompt_template": PROMPT_TEMPLATE,
        "legend": LEGEND,
        "few_shot_block": few_shot,
        "example_rendered_prompt": PROMPT_TEMPLATE.format(
            legend=LEGEND, few_shot=few_shot,
            query=render_features(sample[0]["features"])),
        "label_verification": label_report,
        "most_confused_pairs": confused,
        "per_window": per_window,
        "window_config": {
            "window_seconds": b["window_seconds"], "overlap": b["overlap"],
            "sampling_rate_hz": b["sampling_rate_hz"],
            "features_per_window": 45,
            "column_map": {"timestamp": COL_TIMESTAMP,
                           "activityID": COL_ACTIVITY, **NODE_COLS},
        },
    }
    _write(out_path, payload)

    # ---- report ----------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("BASELINE ACCURACY -- %d windows, subjects %s"
          % (len(sample), payload["subjects_used"]))
    print("=" * 78)
    print("overall accuracy : %.4f   (threshold %.2f, chance %.3f)"
          % (overall, threshold, 1.0 / len(ORDERED_IDS)))
    print("\nper-class accuracy:")
    for k, v in per_class.items():
        if v is None:
            print("   %-20s   n/a (no windows)" % k)
        else:
            print("   %-20s %6.3f  %s" % (k, v, "#" * int(round(v * 40))))

    print("\nconfusion matrix (rows = true, cols = predicted):")
    hdr = "".join("%6s" % TARGET_ACTIVITIES[a][:5] for a in ORDERED_IDS)
    print("%-20s%s" % ("", hdr))
    for i, aid in enumerate(ORDERED_IDS):
        print("%-20s%s" % (TARGET_ACTIVITIES[aid][:19],
                           "".join("%6d" % c for c in m[i])))

    print("\n" + "=" * 78)
    if passed:
        print("RESULT: PASS  (%.4f >= %.2f)" % (overall, threshold))
        print("Headroom above chance is sufficient. Phase 2 may proceed.")
    else:
        print("RESULT: FAIL  (%.4f < %.2f)" % (overall, threshold))
        print("STOP. Do not proceed to Phase 2.")
        print("\n3 most-confused class pairs:")
        for c in confused:
            print("   %-20s -> %-20s %4d windows (%.1f%% of true class)"
                  % (c["true"], c["predicted"], c["count"],
                     c["rate_of_true_class"] * 100))
        print("\n" + _diagnose(confused, per_class, overall))
    print("=" * 78)
    print("wrote %s" % out_path)
    return 0 if passed else 1


def _diagnose(confused: list[dict], per_class: dict, overall: float) -> str:
    """Rank the four candidate fixes by what the confusion data actually shows."""
    stairs = {"ascending_stairs", "descending_stairs"}
    stair_involved = sum(c["count"] for c in confused
                         if c["true"] in stairs or c["predicted"] in stairs)
    total_conf = sum(c["count"] for c in confused) or 1
    stair_share = stair_involved / total_conf

    postural = {"lying", "sitting", "standing"}
    postural_conf = sum(c["count"] for c in confused
                        if c["true"] in postural and c["predicted"] in postural)

    lines = ["DIAGNOSIS (ranked by what the confusion matrix shows):"]
    if stair_share >= 0.5:
        lines += [
            "  1. (c) DROP THE TWO STAIR CLASSES -> 6-class set. %.0f%% of the "
            "top confusions involve stairs; they are near-indistinguishable "
            "from walking on accelerometer-only features." % (stair_share * 100),
            "  2. (b) Richer features (gyro channels, cross-axis correlation) -- "
            "gyro is the channel that actually separates stair climbing.",
            "  3. (a) Better few-shot examples.",
            "  4. (d) Larger model.",
        ]
    elif postural_conf >= 0.3 * total_conf:
        lines += [
            "  1. (b) RICHER FEATURES. Confusion is concentrated among static "
            "postures (lying/sitting/standing), which differ mainly by gravity "
            "orientation -- add per-axis mean-gravity / tilt features.",
            "  2. (a) Better few-shot examples emphasising orientation.",
            "  3. (c) Reduce class count.",
            "  4. (d) Larger model.",
        ]
    elif overall < 0.25:
        lines += [
            "  1. (a) FEW-SHOT / PROMPT FIRST. Accuracy is near chance, which "
            "usually means the model is not following the letter-answer format "
            "rather than that the features are uninformative. Inspect "
            "example_rendered_prompt in the JSON.",
            "  2. (d) Larger model.",
            "  3. (b) Richer features.",
            "  4. (c) Reduce class count.",
        ]
    else:
        lines += [
            "  1. (a) Better/more few-shot examples -- confusion is diffuse, "
            "no single class pair dominates.",
            "  2. (b) Richer features.",
            "  3. (d) Larger model.",
            "  4. (c) Reduce class count (last resort -- it shrinks the "
            "dynamic range available for the degradation study).",
        ]
    return "\n".join(lines)


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
