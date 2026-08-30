#!/usr/bin/env python3
"""PROBE: does foregrounding ORIENTATION clear Gate 2?

Gate 2 failed at 0.4722 (8-class) and 0.5833 (6-class). Both runs' residual
error sits on the static postures -- lying / sitting / standing -- which differ
almost entirely by gravity orientation. The ranked diagnosis put "(b) richer
orientation features" first. This measures it.

What changes
------------
V0  the shipped rendering: 45 raw numbers per window, in m/s^2, with the
    gravity direction implicit across three separate per-axis `mean` fields.

V1  the same underlying features, re-presented:
      - accelerations converted to g (Phase 2's project-wide convention), so
        1.00 means "one gravity" and a static limb reads ~1.00
      - an explicit per-node ORIENTATION line: the unit mean-acceleration
        vector, which for a near-static node IS the gravity direction and
        therefore identifies posture directly
      - an explicit MOTION line: mean per-axis standard deviation, the
        static-vs-dynamic discriminator
      - the raw per-axis detail kept, but demoted below the summary
    No new sensor channels: this is a presentation change over the SAME 45
    features, which is what makes it a fair test of the diagnosis.

Nothing in Phase 1's shipped script is modified. Results land in
results/phase1/feature_variant_probe.json.

Usage
-----
    python scripts/probe_feature_variants.py --backend cerebras --n-windows 96
    python scripts/probe_feature_variants.py --variant V1 --label-set 6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import check_baseline_accuracy as B  # noqa: E402
from _llm_client import REPO_ROOT, describe_backend, load_config  # noqa: E402

G = 9.80665

SIX = {1: "lying", 2: "sitting", 3: "standing", 4: "walking",
       5: "running", 6: "cycling"}
EIGHT = dict(B.TARGET_ACTIVITIES)

# Saved so a variant can be reverted between runs in one process.
V0_RENDER = B.render_features
V0_PROMPT = B.PROMPT_TEMPLATE

# The shipped prompt ends with "Answer:" and relies on the model continuing
# with a bare letter. gpt-4o happens to do that; gemma-4-31b does not -- its
# top tokens were 'Answer', 'To', 'Based', 'The', i.e. it wants to write prose.
# A prompt that only works on one model is a reproducibility problem, so the
# instruction is made explicit and the trailing "Answer:" removed.
STRICT_TAIL = ("Reply with exactly one character: the single letter from the "
               "legend. No punctuation, no explanation, no preamble.")

V0F_PROMPT = """You are an expert at recognising human activity from body-worn inertial sensors.

Each window is 2.56 seconds of tri-axial accelerometer data (units m/s^2, gravity included) from three body-worn nodes: wrist, chest and ankle. For each node and axis you are given mean, standard deviation, minimum, maximum and energy (mean of squares).

Legend:
{legend}

{few_shot}Now classify this window.

{query}

""" + STRICT_TAIL

V1_PROMPT = """You are an expert at recognising human activity from body-worn inertial sensors.

Each window is 2.56 seconds of tri-axial accelerometer data from three body-worn nodes: wrist, chest and ankle. Values are in g (1.00 g = gravity).

For each node you are given:
  ORIENTATION - the mean acceleration direction as a unit vector (x, y, z), and its magnitude in g.
      When a node is nearly still the magnitude is close to 1.00 g and this direction IS the direction of gravity, which tells you how that body part is oriented. This is the main way to tell lying, sitting and standing apart: they are all still, and differ only in orientation.
  MOTION - the mean per-axis standard deviation, in g. Near 0.00 means still; large means vigorous movement. This separates the still postures from walking, running and cycling.
  DETAIL - per-axis mean/std/min/max for reference.

Legend:
{legend}

{few_shot}Now classify this window.

{query}

""" + STRICT_TAIL


def render_v1(feats: dict) -> str:
    """Orientation- and motion-first rendering of the SAME 45 features."""
    lines = []
    for node in B.NODE_COLS:
        means = [feats["%s_%s_mean" % (node, a)] / G for a in B.AXES]
        stds = [feats["%s_%s_std" % (node, a)] / G for a in B.AXES]
        mag = math.sqrt(sum(m * m for m in means))
        if mag > 1e-9:
            unit = [m / mag for m in means]
        else:
            unit = [0.0, 0.0, 0.0]
        motion = sum(stds) / len(stds)
        state = "still" if motion < 0.08 else (
            "moderate" if motion < 0.35 else "vigorous")
        lines.append(
            "%-6s ORIENTATION (%+.2f, %+.2f, %+.2f)  magnitude %.2f g"
            % (node, unit[0], unit[1], unit[2], mag))
        lines.append(
            "       MOTION      %.3f g  (%s)" % (motion, state))
        detail = []
        for i, a in enumerate(B.AXES):
            detail.append("%s mean %+.2f std %.2f min %+.2f max %+.2f"
                          % (a, means[i], stds[i],
                             feats["%s_%s_min" % (node, a)] / G,
                             feats["%s_%s_max" % (node, a)] / G))
        lines.append("       DETAIL      " + " | ".join(detail))
    return "\n".join(lines)


def apply_variant(variant: str) -> None:
    if variant == "V1":
        B.render_features = render_v1
        B.PROMPT_TEMPLATE = V1_PROMPT
    elif variant == "V0F":
        B.render_features = V0_RENDER
        B.PROMPT_TEMPLATE = V0F_PROMPT
    else:
        B.render_features = V0_RENDER
        B.PROMPT_TEMPLATE = V0_PROMPT


def apply_label_set(mapping: dict) -> None:
    ids = list(mapping)
    letters = [chr(ord("A") + i) for i in range(len(ids))]
    B.TARGET_ACTIVITIES = dict(mapping)
    B.ORDERED_IDS = ids
    B.LETTERS = letters
    B.LETTER_TO_ID = dict(zip(letters, ids))
    B.ID_TO_LETTER = {v: k for k, v in B.LETTER_TO_ID.items()}
    B.LEGEND = "\n".join("%s = %s" % (l, mapping[i])
                         for l, i in zip(letters, ids))


def run_one(variant: str, label_set: str, backend: str, n_windows: int,
            tmp: str) -> dict:
    apply_variant(variant)
    apply_label_set(SIX if label_set == "6" else EIGHT)
    argv = ["check_baseline_accuracy.py", "--n-windows", str(n_windows),
            "--out", tmp, "--backend", backend]
    # Delete the temp file FIRST. Without this, a cell that aborts (a rate
    # limit, say) leaves the PREVIOUS cell's output in place and this function
    # silently returns it as the new result -- which is exactly how three
    # different variants came back with one identical accuracy.
    for p in (tmp, os.path.join(REPO_ROOT, "frontend", "results", "phase1",
                                os.path.basename(tmp))):
        if os.path.isfile(p):
            os.remove(p)

    old = sys.argv
    sys.argv = argv
    try:
        rc = B.main()
    finally:
        sys.argv = old

    if not os.path.isfile(tmp):
        raise RuntimeError(
            "run %s/%s produced no output (exit %s) -- the cell aborted, most "
            "likely a rate limit or an unparseable answer. Refusing to reuse a "
            "stale result." % (variant, label_set, rc))
    with open(tmp, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    # Cross-check that the file we just read really is this cell's.
    n_classes = len(d.get("confusion_matrix_labels") or [])
    expected = 6 if label_set == "6" else 8
    if n_classes != expected:
        raise RuntimeError(
            "run %s/%s returned a %d-class result, expected %d -- stale or "
            "mismatched output" % (variant, label_set, n_classes, expected))
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="cerebras")
    ap.add_argument("--n-windows", type=int, default=96)
    ap.add_argument("--variant", default="all",
                    choices=["V0", "V0F", "V1", "all"])
    ap.add_argument("--label-set", default="both", choices=["8", "6", "both"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    tmp = os.path.join(REPO_ROOT, "results", "phase1", "_fv_tmp.json")
    out_path = args.out or os.path.join(REPO_ROOT, "results", "phase1",
                                        "feature_variant_probe.json")

    variants = (["V0", "V0F", "V1"] if args.variant == "all"
                else [args.variant])
    label_sets = ["8", "6"] if args.label_set == "both" else [args.label_set]

    print("=" * 78)
    print("PROBE: ORIENTATION-FIRST FEATURE PRESENTATION")
    print("=" * 78)
    print("  %s" % describe_backend(cfg, args.backend))
    print("  windows per cell: %d\n" % args.n_windows)

    runs: dict = {}
    for ls in label_sets:
        for v in variants:
            key = "%s/%sclass" % (v, ls)
            print("-" * 78)
            print("RUN %s" % key)
            print("-" * 78)
            try:
                r = run_one(v, ls, args.backend, args.n_windows, tmp)
            except (FileNotFoundError, RuntimeError) as exc:
                # The inner run aborted (e.g. the model never emitted a label
                # token). Record the failure rather than crashing the sweep.
                print("  -> RUN FAILED: %s" % str(exc)[:150])
                runs[key] = {"variant": v, "label_set": int(ls),
                             "overall_accuracy": None, "pass": False,
                             "failed": True, "reason": str(exc)[:300]}
                continue
            runs[key] = {
                "variant": v, "label_set": int(ls),
                "overall_accuracy": r["overall_accuracy"],
                "pass": r["overall_accuracy"] >= 0.65,
                "per_class_accuracy": r["per_class_accuracy"],
                "confusion_matrix": r["confusion_matrix"],
                "confusion_matrix_labels": r["confusion_matrix_labels"],
                "most_confused_pairs": r["most_confused_pairs"],
                "n_windows": r["n_windows"],
                "model": r.get("model"), "backend": r.get("backend"),
                "prompt_template": r.get("prompt_template"),
                "example_rendered_prompt": r.get("example_rendered_prompt"),
            }

    payload = {
        "generated_by": "scripts/probe_feature_variants.py",
        "status": "PROBE -- Phase 1's shipped script and result are unmodified",
        "threshold": 0.65,
        "backend": args.backend,
        "n_windows": args.n_windows,
        "variant_description": {
            "V0": "shipped rendering: 45 raw numbers in m/s^2, orientation "
                  "implicit across three per-axis means; prompt ends "
                  "'Answer:' and relies on the model continuing with a letter",
        "V0F": "V0 features, but an explicit single-character answer "
               "instruction -- isolates answer-format effect from features",
            "V1": "same 45 features, re-presented: values in g, explicit "
                  "per-node ORIENTATION unit vector + magnitude, explicit "
                  "MOTION level, raw detail demoted",
        },
        "runs": runs,
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
    print("FEATURE VARIANT COMPARISON  (threshold 0.65)")
    print("=" * 78)
    print("  %-16s %10s %10s" % ("cell", "accuracy", "verdict"))
    for k in sorted(runs):
        r = runs[k]
        acc = r.get("overall_accuracy")
        print("  %-16s %10s %10s"
              % (k, "FAILED" if acc is None else "%.4f" % acc,
                 "PASS" if r["pass"] else "FAIL"))
    for ls in label_sets:
        for lo, hi, name in (("V0", "V0F", "answer format"),
                             ("V0F", "V1", "orientation features"),
                             ("V0", "V1", "combined")):
            a = runs.get("%s/%sclass" % (lo, ls))
            b = runs.get("%s/%sclass" % (hi, ls))
            if a and b and a.get("overall_accuracy") is not None                     and b.get("overall_accuracy") is not None:
                print("  %s-class  %-4s -> %-4s  %+.4f   (%s)"
                      % (ls, lo, hi,
                         b["overall_accuracy"] - a["overall_accuracy"], name))
    print("\nwrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
