#!/usr/bin/env python3
"""Run the S1 agent over healthy AND pre-injected windows.

Writes per-window rows into the resolved provider's folder, each carrying
provider, model, confidence_method, backend, prompt version, seed, and BOTH
max_p and log_margin.

Determinism protocol
--------------------
Every window is classified TWICE and the spread recorded. Phase 1 found gpt-4o
varies run-to-run at temperature 0 (max_p spread 0.046) while Cerebras is
bit-identical. A study whose dependent variable IS a confidence value has to
report the reproducibility of that value, so this is a methods-section number,
not an implementation detail.

Usage
-----
    python scripts/run_phase5_samples.py --healthy 4 --injected 4
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import check_baseline_accuracy as B  # noqa: E402
from _llm_client import LogprobError, describe_backend, load_config  # noqa: E402
from agent.graph import S1Agent  # noqa: E402
from agent.llm_client_adapter import LLMClientAdapter  # noqa: E402
from core.results_paths import resolve_from_cfg, write_json  # noqa: E402
from datasets.dataset_replay_source import DatasetReplaySource  # noqa: E402
from injection.registry import make_injector  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260830
EVAL_SUBJECTS = ["subject101", "subject105", "subject106"]

# The injected conditions to sample. Chosen to span the taxonomy rather than
# to flatter any one of them.
INJECTIONS = [("dropout", 4), ("packet_loss", 3), ("rate_degradation", 4),
              ("clock_desync", 4)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--healthy", type=int, default=4)
    ap.add_argument("--injected", type=int, default=4)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--subject", default="subject101")
    ap.add_argument("--target-node", default="ankle")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config()
    adapter = LLMClientAdapter(cfg, args.backend)
    outdir = resolve_from_cfg(cfg, adapter.backend)

    # If the headroom probe named a primary signal for this provider, use it.
    primary = "max_p"
    hp = os.path.join(outdir, "confidence_headroom.json")
    headroom_verdict = None
    if os.path.isfile(hp):
        import json
        h = json.load(open(hp, encoding="utf-8"))
        headroom_verdict = h.get("verdict")
        primary = h.get("primary_confidence_signal") or "max_p"

    agent = S1Agent(adapter=adapter, label_set=B.LETTERS, legend=B.LEGEND,
                    primary_confidence_signal=primary,
                    eval_subjects=EVAL_SUBJECTS)

    print("=" * 78)
    print("PHASE 5 -- S1 AGENT SAMPLES")
    print("=" * 78)
    print("  %s" % describe_backend(cfg, adapter.backend))
    print("  graph runtime      : %s" % agent.runtime)
    print("  primary confidence : %s" % primary)
    if headroom_verdict:
        print("  headroom verdict   : %s" % headroom_verdict)
    print("  repeats per window : %d  (determinism protocol)" % args.repeats)
    print("  output -> %s\n" % os.path.relpath(outdir, REPO_ROOT))

    def src():
        return DatasetReplaySource("pamap2", subjects=[args.subject],
                                   label_set="PAMAP2_8")

    # STRATIFY across activities. Windows arrive in activity-segment order, so
    # simply taking the first N gives N windows of one class -- an earlier run
    # scored 0.00 accuracy purely because every sampled window was `lying`.
    def stratified(source, n):
        by_label: dict = {}
        for idx, w in enumerate(source.windows()):
            by_label.setdefault(w.label, []).append((idx, w))
        labels = [l for l in B.TARGET_ACTIVITIES.values() if l in by_label]
        picked, r = [], 0
        while len(picked) < n and labels:
            lab = labels[r % len(labels)]
            pool = by_label[lab]
            take = len([p for p in picked if p[2].label == lab])
            if take < len(pool):
                # middle of the class's run: the first windows of a segment
                # often catch the subject still settling into the activity
                i = (len(pool) // 2 + take) % len(pool)
                picked.append((lab, pool[i][0], pool[i][1]))
            else:
                labels.remove(lab)
                continue
            r += 1
        return picked

    jobs = []
    for lab, idx, w in stratified(src(), args.healthy):
        jobs.append(("healthy", None, None, idx, w))
    per_inj = max(1, args.injected // len(INJECTIONS))
    for ftype, sev in INJECTIONS:
        inj = make_injector(src(), ftype, sev, args.target_node, SEED)
        for lab, idx, w in stratified(inj, per_inj):
            jobs.append(("injected", ftype, sev, idx, w))

    rows = []
    for cond, ftype, sev, idx, w in jobs:
        reps = []
        for rep in range(args.repeats):
            try:
                r = agent.run(w)
            except LogprobError as exc:
                print("  %-9s %-18s w%-2d FAILED: %s"
                      % (cond, ftype or "-", idx, str(exc)[:80]))
                reps = []
                break
            reps.append(r)
        if not reps:
            continue

        r0 = reps[0]
        mp = [r["max_p"] for r in reps]
        lm = [r["log_margin"] for r in reps if r["log_margin"] is not None]
        row = {
            "condition": cond,
            "failure_type": ftype, "severity": sev,
            "window": idx,
            "true_label": w.label,
            "predicted": r0["predicted"],
            "correct": None,     # filled below once mapped to a label name
            "max_p": r0["max_p"],
            "log_margin": r0["log_margin"],
            "entropy": r0["entropy"],
            "primary_confidence_signal": r0["primary_confidence_signal"],
            "primary_confidence_value": r0["primary_confidence_value"],
            "distribution": r0["distribution"],
            "health_states": r0["health_states"],
            "health_diagnoses": r0["health_diagnoses"],
            "trust_weights": r0["trust_weights"],
            "confidence_method": r0["confidence_method"],
            "provider": r0["provider"], "model": r0["model"],
            "backend": adapter.backend,
            "prompt_version": r0["prompt_version"],
            "flags": r0["flags"],
            "seed": SEED,
            "repeats": len(reps),
            "max_p_repeats": mp,
            "max_p_spread": (max(mp) - min(mp)) if len(mp) > 1 else 0.0,
            "log_margin_repeats": lm,
            "log_margin_spread": (max(lm) - min(lm)) if len(lm) > 1 else 0.0,
            "bit_identical_repeats": bool(len(set(
                (round(r["max_p"], 12), r["predicted"]) for r in reps)) == 1),
        }
        pred_name = B.TARGET_ACTIVITIES[B.LETTER_TO_ID[r0["predicted"]]]
        row["predicted_label"] = pred_name
        row["correct"] = bool(pred_name == w.label)
        rows.append(row)
        print("  %-9s %-18s w%-2d  true=%-18s pred=%-18s %s  max_p %.4f "
              "log_margin %s  spread %.5f"
              % (cond, "%s sev%s" % (ftype, sev) if ftype else "-", idx,
                 w.label, pred_name, "OK " if row["correct"] else "MISS",
                 row["max_p"],
                 "--" if row["log_margin"] is None else "%.3f" % row["log_margin"],
                 row["max_p_spread"]))

    if not rows:
        print("\nNo rows produced.")
        return 1

    # ---- determinism summary --------------------------------------------- #
    spreads = [r["max_p_spread"] for r in rows]
    lm_spreads = [r["log_margin_spread"] for r in rows
                  if r["log_margin_repeats"]]
    determinism = {
        "n_windows": len(rows), "repeats": args.repeats,
        "mean_abs_delta_max_p": statistics.mean(spreads),
        "max_abs_delta_max_p": max(spreads),
        "mean_abs_delta_log_margin": (statistics.mean(lm_spreads)
                                      if lm_spreads else None),
        "bit_identical": all(r["bit_identical_repeats"] for r in rows),
        "protocol": "every window classified %d times in the same run"
                    % args.repeats,
    }

    def group(cond):
        g = [r for r in rows if r["condition"] == cond]
        if not g:
            return None
        lm = [r["log_margin"] for r in g if r["log_margin"] is not None]
        return {"n": len(g),
                "accuracy": sum(r["correct"] for r in g) / len(g),
                "mean_max_p": statistics.mean(r["max_p"] for r in g),
                "mean_log_margin": statistics.mean(lm) if lm else None,
                "mean_entropy": statistics.mean(r["entropy"] for r in g)}

    payload = {
        "generated_by": "scripts/run_phase5_samples.py",
        "phase": 5, "system": "S1",
        "provider": adapter.backend, "model": adapter.model,
        "confidence_method": adapter.confidence_method,
        "prompt_version": rows[0]["prompt_version"],
        "graph_runtime": agent.runtime,
        "flags": rows[0]["flags"],
        "seed": SEED, "eval_subjects": EVAL_SUBJECTS,
        "primary_confidence_signal": primary,
        "headroom_verdict": headroom_verdict,
        "determinism": determinism,
        "by_condition": {"healthy": group("healthy"),
                         "injected": group("injected")},
        "rows": rows,
    }
    write_json(outdir, "phase5_samples.json", payload)
    write_json(outdir, "determinism.json", determinism)

    h, i = payload["by_condition"]["healthy"], payload["by_condition"]["injected"]
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, g in (("healthy", h), ("injected", i)):
        if g:
            print("  %-9s n=%-3d accuracy %.2f  mean max_p %.4f  "
                  "mean log_margin %s"
                  % (name, g["n"], g["accuracy"], g["mean_max_p"],
                     "--" if g["mean_log_margin"] is None
                     else "%.3f" % g["mean_log_margin"]))
    if h and i:
        print("\n  QUALITATIVE CHECK -- any visible confidence difference?")
        dmp = i["mean_max_p"] - h["mean_max_p"]
        print("     delta mean max_p      %+.5f" % dmp)
        if i["mean_log_margin"] is not None and h["mean_log_margin"] is not None:
            print("     delta mean log_margin %+.4f"
                  % (i["mean_log_margin"] - h["mean_log_margin"]))
    print("\n  determinism: mean |delta max_p| across repeats %.6f, "
          "bit-identical: %s"
          % (determinism["mean_abs_delta_max_p"],
             determinism["bit_identical"]))
    print("\nwrote %s" % os.path.join(os.path.relpath(outdir, REPO_ROOT),
                                      "phase5_samples.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
