#!/usr/bin/env python3
"""Can this provider EXPRESS degraded confidence at all?

Phase 1 measured gemma-4-31b at max p = 0.99999 -- saturated. That is a threat
to Phase 6's entire dependent variable: if confidence is pinned at 1.0, the
Overconfidence Gap collapses to (1 - accuracy) by construction, and "confidence
does not track degradation" becomes true by arithmetic rather than a finding.

So measure it before building anything on top. Cheap: ~2 calls per window.

Three signals, deliberately at different resolutions:

  max_p       softmax maximum. Saturates first and hides everything after.
  log_margin  top1_logprob - top2_logprob. THE key measurement: it keeps
              resolution long after the softmax has pinned at 1.0, because it
              lives in log space where the difference is still finite.
  entropy     over the label distribution; a whole-distribution view.

Verdict, stated plainly:
  max_p moves                  -> softmax confidence is usable, proceed.
  max_p pinned, margin moves   -> use log_margin as this provider's primary
                                  confidence signal; record that in metadata.
  nothing moves                -> this provider CANNOT express degraded
                                  confidence. Phase 6's headline must come from
                                  another provider. Say so; do not proceed and
                                  emit a flat, uninterpretable chart.

Prompts are NOT tuned to manufacture headroom. This reports what the model does.

Usage
-----
    python scripts/measure_confidence_headroom.py --windows 20
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import check_baseline_accuracy as B  # noqa: E402
from _llm_client import (  # noqa: E402
    LogprobError,
    call_backend,
    confidence_method_for,
    describe_backend,
    load_config,
    norm_token,
    resolve_backends,
)
from core.results_paths import resolve_from_cfg, write_json  # noqa: E402
from datasets.dataset_replay_source import DatasetReplaySource  # noqa: E402
from injection.registry import make_injector  # noqa: E402
from features.extractors import render, window_features  # noqa: E402
from health.window_view import blind  # noqa: E402

PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "agent", "prompts", "system_v1.txt")
with open(PROMPT_PATH, encoding="utf-8") as _fh:
    SYSTEM_V1 = _fh.read()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def confidence_signals(cfg: dict, backend: str, prompt: str,
                       letters: list[str], n_probs: int) -> dict:
    """max_p, log_margin and entropy from ONE call's raw candidates.

    Reuses Phase 1's call + token normalisation exactly; the softmax is
    recomputed here only so the log-space margin can be read off the same
    candidate list before normalisation collapses it.
    """
    call = call_backend(cfg, backend, prompt, n_probs)
    wanted = {l.upper() for l in letters}

    per_letter: dict[str, list[float]] = {}
    for tok, lp in call["candidates"]:
        k = norm_token(tok)
        if k in wanted:
            per_letter.setdefault(k, []).append(lp)
    if not per_letter:
        seen = [t for t, _ in call["candidates"]][:10]
        raise LogprobError(
            "no label token in the returned top-k; model's top tokens were %r"
            % seen)

    def lse(xs):
        m = max(xs)
        return m + math.log(sum(math.exp(x - m) for x in xs))

    merged = {k: lse(v) for k, v in per_letter.items()}
    ordered = sorted(merged.items(), key=lambda kv: -kv[1])

    top1_lp = ordered[0][1]
    top2_lp = ordered[1][1] if len(ordered) > 1 else None
    # The key quantity: still finite and informative when max_p reads 1.00000.
    log_margin = (top1_lp - top2_lp) if top2_lp is not None else None

    mx = max(merged.values())
    exps = {k: math.exp(v - mx) for k, v in merged.items()}
    tot = sum(exps.values())
    probs = {k: v / tot for k, v in exps.items()}
    ent = -sum(p * math.log(p) for p in probs.values() if p > 0)

    return {"max_p": max(probs.values()), "log_margin": log_margin,
            "entropy": ent, "predicted": ordered[0][0],
            "n_labels_seen": len(merged)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--dataset", default="pamap2")
    ap.add_argument("--subject", default="subject101")
    ap.add_argument("--target-node", default="ankle")
    args = ap.parse_args()

    cfg = load_config()
    backend = resolve_backends(cfg, args.backend)[0]
    method = confidence_method_for(cfg, backend)
    n_probs = int(cfg["request"].get("step2_n_probs", 20))
    outdir = resolve_from_cfg(cfg, backend)

    print("=" * 78)
    print("CONFIDENCE HEADROOM PROBE")
    print("=" * 78)
    print("  %s" % describe_backend(cfg, backend))
    print("  output -> %s" % os.path.relpath(outdir, REPO_ROOT))
    print("  clean vs dropout sev4, %d windows, 2 calls each\n" % args.windows)


    src = DatasetReplaySource(args.dataset, subjects=[args.subject],
                              label_set="PAMAP2_8")
    inj = make_injector(
        DatasetReplaySource(args.dataset, subjects=[args.subject],
                            label_set="PAMAP2_8"),
        "dropout", 4, args.target_node, 20260830)

    clean_ws, inj_ws = [], []
    for i, w in enumerate(src.windows()):
        if i >= args.windows:
            break
        clean_ws.append(w)
    for i, w in enumerate(inj.windows()):
        if i >= args.windows:
            break
        inj_ws.append(w)

    # Few-shot from disjoint subjects, exactly as Gate 2 does.
    fs_src = DatasetReplaySource(args.dataset, subjects=["subject102"],
                                 label_set="PAMAP2_8")
    fs_by_label: dict = {}
    for i, w in enumerate(fs_src.windows()):
        if i >= 400:
            break
        fs_by_label.setdefault(w.label, []).append(w)
    few = ""
    for lab in B.TARGET_ACTIVITIES.values():
        for w in (fs_by_label.get(lab) or [])[:2]:
            few += B.EXAMPLE_BLOCK.format(
                features=render(window_features(blind(w))),
                letter=B.ID_TO_LETTER[_label_id(lab)])
    if few:
        few = "Here are labelled examples.\n\n" + few

    rows = []
    for cond, windows in (("clean", clean_ws), ("dropout_sev4", inj_ws)):
        for idx, w in enumerate(windows):
            prompt = SYSTEM_V1.format(
                legend=B.LEGEND, few_shot=few,
                query=render(window_features(blind(w))))
            try:
                s = confidence_signals(cfg, backend, prompt, B.LETTERS, n_probs)
            except LogprobError as exc:
                print("  %-13s w%-3d FAILED: %s" % (cond, idx, str(exc)[:90]))
                continue
            s.update({"condition": cond, "window": idx, "true_label": w.label})
            rows.append(s)
            if idx % 5 == 0:
                print("  %-13s w%-3d max_p %.6f  log_margin %s  entropy %.4f"
                      % (cond, idx, s["max_p"],
                         "None" if s["log_margin"] is None
                         else "%.4f" % s["log_margin"], s["entropy"]))

    if not rows:
        print("\nNo usable measurements -- cannot judge headroom.")
        return 1

    def agg(cond, key):
        vals = [r[key] for r in rows
                if r["condition"] == cond and r.get(key) is not None]
        if not vals:
            return None
        return {"n": len(vals), "mean": statistics.mean(vals),
                "min": min(vals), "max": max(vals),
                "sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0}

    # The design is PAIRED: window i is measured clean and injected. Comparing
    # group means against unpaired spread throws that away and is far less
    # sensitive -- between-window variation (different activities) dwarfs the
    # within-window effect of injection. Analyse the per-window deltas.
    by_win = {}
    for r in rows:
        by_win.setdefault(r["window"], {})[r["condition"]] = r

    summary = {}
    for key in ("max_p", "log_margin", "entropy"):
        c, d = agg("clean", key), agg("dropout_sev4", key)
        deltas = []
        for w, pair in by_win.items():
            a, b = pair.get("clean"), pair.get("dropout_sev4")
            if a and b and a.get(key) is not None and b.get(key) is not None:
                deltas.append(b[key] - a[key])
        paired = None
        if len(deltas) >= 3:
            md = statistics.mean(deltas)
            sd = statistics.pstdev(deltas)
            n_pos = sum(1 for x in deltas if x > 0)
            n_neg = sum(1 for x in deltas if x < 0)
            # Cohen's d for paired data, and how consistently signed the shift
            # is. A real effect is both sizeable AND mostly one-directional.
            paired = {
                "n_pairs": len(deltas),
                "mean_delta": md,
                "sd_delta": sd,
                "cohens_d": (md / sd) if sd > 1e-12 else None,
                "n_decreased": n_neg, "n_increased": n_pos,
                "consistency": max(n_pos, n_neg) / len(deltas),
            }
        # "Moves" now means: a paired shift of non-trivial size that is
        # consistently signed across windows.
        moves = bool(paired and paired["cohens_d"] is not None
                     and abs(paired["cohens_d"]) >= 0.5
                     and paired["consistency"] >= 0.7)
        summary[key] = {"clean": c, "dropout_sev4": d,
                        "delta": (d["mean"] - c["mean"]) if (c and d) else None,
                        "paired": paired, "moves": moves}

    print("\n" + "=" * 78)
    print("HEADROOM  (clean -> dropout sev4)")
    print("=" * 78)
    print("  %-11s %10s %10s %10s %8s %7s %8s" %
          ("signal", "clean", "injected", "d(paired)", "cohen d", "consist",
           "moves?"))
    for key in ("max_p", "log_margin", "entropy"):
        s = summary[key]
        c = s["clean"]["mean"] if s["clean"] else float("nan")
        d = s["dropout_sev4"]["mean"] if s["dropout_sev4"] else float("nan")
        pr = s.get("paired") or {}
        print("  %-11s %10.5f %10.5f %10.5f %8s %7s %8s"
              % (key, c, d, pr.get("mean_delta", float("nan")),
                 "--" if pr.get("cohens_d") is None else "%.2f" % pr["cohens_d"],
                 "--" if pr.get("consistency") is None else "%.0f%%" % (100*pr["consistency"]),
                 "YES" if s["moves"] else "no"))

    saturated = bool(summary["max_p"]["clean"]
                     and summary["max_p"]["clean"]["mean"] > 0.99)
    if summary["max_p"]["moves"]:
        verdict = "max_p_usable"
        primary = "max_p"
        text = ("max_p moves measurably between clean and heavily degraded "
                "input. Softmax confidence is usable for this provider; "
                "proceed normally.")
    elif summary["log_margin"]["moves"]:
        verdict = "use_log_margin"
        primary = "log_margin"
        text = ("max_p is pinned (mean %.5f) but the LOG-SPACE MARGIN moves. "
                "Use log_margin as this provider's primary confidence signal "
                "and record that choice in every result row. Softmax "
                "confidence would understate the effect to the point of "
                "invisibility." % (summary["max_p"]["clean"]["mean"]
                                   if summary["max_p"]["clean"] else float("nan")))
    else:
        verdict = "no_headroom"
        primary = None
        text = ("NOTHING MOVES. This provider cannot express degraded "
                "confidence on this task. Phase 6's headline result must come "
                "from a different provider -- gpt-4o once credits are "
                "restored, or self-consistency via anthropic. Running the "
                "Phase 6 sweep on this provider would produce a flat, "
                "uninterpretable chart whose 'finding' is an artefact of "
                "saturation.")

    print("\n" + "=" * 78)
    print("VERDICT: %s" % verdict.upper())
    print("=" * 78)
    for line in _wrap(text, 76):
        print("  " + line)

    payload = {
        "generated_by": "scripts/measure_confidence_headroom.py",
        "phase": 5,
        "backend": backend,
        "model": cfg["backends"][backend].get("model"),
        "confidence_method": method,
        "dataset": args.dataset, "subject": args.subject,
        "injection": "dropout sev4 on %s" % args.target_node,
        "n_windows_per_condition": args.windows,
        "prompt": "agent/prompts/system_v1.txt (not retuned for this probe)",
        "extractor": "features/extractors.py -- missing data reported as NO DATA, never zero-filled",
        "signals": summary,
        "max_p_saturated": saturated,
        "verdict": verdict,
        "primary_confidence_signal": primary,
        "verdict_text": text,
        "rows": rows,
    }
    p = write_json(outdir, "confidence_headroom.json", payload)
    print("\nwrote %s" % os.path.relpath(p, REPO_ROOT))
    return 0


# --- small helpers reusing Gate 2's feature machinery ---------------------- #
def _feat_cols():
    return ["%s_%s" % (n, a) for n in B.NODE_COLS for a in B.AXES]


def _accel_block(w):
    import numpy as np
    nodes = list(B.NODE_COLS)
    n = len(w.frames[nodes[0]])
    out = np.empty((n, 9), dtype=float)
    for j, node in enumerate(nodes):
        for k in range(3):
            out[:, j * 3 + k] = [f.accel_g[k] * 9.80665
                                 for f in w.frames[node]]
    return out


def _label_id(name):
    for i, v in B.TARGET_ACTIVITIES.items():
        if v == name:
            return i
    raise KeyError(name)


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for wd in words:
        if len(line) + len(wd) + 1 > width:
            out.append(line); line = wd
        else:
            line = (line + " " + wd).strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    sys.exit(main())
