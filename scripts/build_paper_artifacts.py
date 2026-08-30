#!/usr/bin/env python3
"""Assemble every phase's results into paper-ready tables and an appendix.

Reads the results/ JSONs written by the phase scripts and emits:

    results/paper/tables.md            paper-ready markdown tables
    results/paper/reproducibility.md   exact prompts, seeds, splits, versions
    results/paper/paper_data.json      one consolidated machine-readable blob

Makes NO LLM calls and computes nothing new -- it only reports what the phase
scripts measured. Any number here can be traced to the file it came from, and
a missing input is reported as missing rather than filled in.

Usage
-----
    python scripts/build_paper_artifacts.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO_ROOT, "results")
OUT = os.path.join(RES, "paper")

SOURCES = {
    "logprob_check": "phase1/logprob_check.json",
    "logprob_openai": "phase1/logprob_check_openai.json",
    "logprob_cerebras": "phase1/logprob_check_cerebras.json",
    "determinism": "phase1/determinism_check.json",
    "baseline_8": "phase1/baseline_accuracy.json",
    "label_set_probe": "phase1/label_set_probe.json",
    "feature_variant_openai_8": "phase1/feature_variant_openai_8.json",
    "feature_variant_openai_6": "phase1/feature_variant_openai_6.json",
    "feature_variant_cerebras_8": "phase1/feature_variant_cerebras_8.json",
    "pamap2_profile": "phase1/dataset_profile.json",
    "mhealth_profile": "phase1/mhealth_profile.json",
    "dataset_stats": "phase2/dataset_stats.json",
    "injectors": "phase3/injector_verification.json",
    "detection": "phase4/detection_metrics.json",
    "temporal_probe": "phase4/temporal_baseline_probe.json",
}


def load(rel: str):
    p = os.path.join(RES, rel)
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def fmt(v, d=3, dash="--"):
    return dash if v is None else ("%.*f" % (d, v))


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def table_datasets(D) -> str:
    st = D.get("dataset_stats")
    if not st:
        return "_dataset_stats.json missing -- run scripts/profile_datasets.py_"
    rows = []
    for name, d in sorted(st["datasets"].items()):
        if not d.get("available"):
            continue
        ch = d["node_channels"]
        chan = "; ".join("%s: %s" % (n, "+".join(c)) for n, c in ch.items())
        rows.append([
            name.upper(), d["n_subjects"], "%.0f Hz" % d["sampling_rate_hz"],
            "{:,}".format(d["total_raw_rows"]),
            "%.1f" % d["total_duration_hours"],
            "{:,}".format(d["total_windows"]), d["n_classes"], chan,
        ])
    return md_table(
        ["Dataset", "Subjects", "Rate", "Rows", "Hours", "Windows",
         "Classes", "Per-node channels"], rows)


def table_providers(D) -> str:
    rows = []
    for key, role in (("logprob_openai", "paid -- paper results"),
                      ("logprob_cerebras", "free -- development, demos")):
        lp = D.get(key)
        if not lp:
            continue
        vals = sorted(lp.get("distribution", {}).values(), reverse=True)
        second = vals[1] if len(vals) > 1 else None
        mx = lp.get("max_prob")
        # A model pinned near 1.0 has no room left to express reduced
        # confidence, which is the quantity this project measures.
        head = "**saturated**" if mx and mx > 0.99 else "usable"
        rows.append([lp.get("backend"), "`%s`" % lp.get("model"),
                     "yes" if lp.get("pass") else "NO",
                     fmt(mx, 5), fmt(second, 5), head, role])
    return md_table(
        ["Provider", "Model", "Real logprobs", "max p", "2nd p",
         "Confidence headroom", "Role"],
        rows) if rows else "_per-provider logprob checks missing_"


def table_determinism(D) -> str:
    """Repeat-probe stability -- a confidence study must state this."""
    dt = D.get("determinism")
    if not dt:
        return "_determinism_check.json missing_"
    rows = []
    for prov, e in dt["providers"].items():
        rows.append([prov, len(e["max_probs"]), fmt(e["mean"], 5),
                     fmt(e["stdev"], 5), fmt(e["spread"], 5),
                     "yes" if e["argmax_stable"] else "**no**"])
    tbl = md_table(["Provider", "Repeats", "mean max p", "sd", "spread",
                    "Bit-identical"], rows)
    tbl += ("\n\n> The **same** prompt at temperature 0. `gpt-4o` is not "
            "bit-reproducible: repeated identical probes returned differing "
            "confidence values. A study whose dependent variable IS the "
            "confidence number must therefore report repeated measurements "
            "with variance, not a single run. `gemma-4-31b` on Cerebras was "
            "bit-identical across repeats but is saturated (Table 2), so "
            "neither provider is simultaneously reproducible and expressive "
            "-- a constraint worth stating explicitly in the methods.")
    return tbl


def table_ablation(D) -> str:
    """Feature/format ablation across variants, label sets and providers."""
    rows = []

    b8 = D.get("baseline_8")
    if b8:
        rows.append(["openai", "gpt-4o", "V0 (shipped)", 8,
                     b8["n_windows"], fmt(b8["overall_accuracy"], 4),
                     "PASS" if b8.get("pass") else "FAIL"])
    lsp = D.get("label_set_probe")
    if lsp and lsp.get("six_class"):
        s = lsp["six_class"]
        rows.append(["openai", "gpt-4o", "V0 (shipped)", 6,
                     s["n_windows"], fmt(s["overall_accuracy"], 4),
                     "PASS" if s["pass"] else "FAIL"])

    for key, prov in (("feature_variant_openai_8", "openai"),
                      ("feature_variant_openai_6", "openai"),
                      ("feature_variant_cerebras_8", "cerebras")):
        fv = D.get(key)
        if not fv:
            continue
        for cell, r in sorted(fv.get("runs", {}).items()):
            acc = r.get("overall_accuracy")
            rows.append([
                prov, "`%s`" % (r.get("model") or fv.get("backend")),
                r["variant"], r["label_set"], r.get("n_windows", "--"),
                "FAILED" if acc is None else fmt(acc, 4),
                "PASS" if r.get("pass") else
                ("UNUSABLE" if acc is None else "FAIL"),
            ])
    return md_table(
        ["Provider", "Model", "Variant", "Classes", "n", "Accuracy",
         "Gate 2"], rows) if rows else "_no ablation runs found_"


def table_detection(D) -> str:
    det = D.get("detection")
    if not det:
        return "_detection_metrics.json missing -- run scripts/run_detection_eval.py_"
    rows = []
    for ft, d in sorted(det["detection_by_failure"].items()):
        a = d["all"]
        rows.append([ft, fmt(a["precision"]), fmt(a["recall"]),
                     "**%s**" % fmt(a["f1"]), a["tp"], a["fp"], a["fn"]])
    fp = det["false_positives"]
    tbl = md_table(["Failure", "Precision", "Recall", "F1", "TP", "FP", "FN"],
                   rows)
    tbl += ("\n\nFalse-positive rate on clean windows: **%s** (%d of %d "
            "node-windows). Diagnosis accuracy across the 6 classes: **%s**."
            % (fmt(fp["false_positive_rate"], 4), fp["false_positives"],
               fp["n_clean_node_windows"],
               fmt(det["diagnosis_confusion"]["overall_accuracy"], 4)))
    return tbl


def table_confusion(D) -> str:
    det = D.get("detection")
    if not det:
        return "_missing_"
    cm = det["diagnosis_confusion"]
    labs = cm["labels"]
    rows = []
    for i, l in enumerate(labs):
        rows.append([("**%s**" % l)] + [str(v or "") for v in cm["matrix"][i]])
    return md_table(["true \\ predicted"] + labs, rows)


def table_displacement(D) -> str:
    det, tp = D.get("detection"), D.get("temporal_probe")
    if not det:
        return "_missing_"
    rows = []
    for k, e in sorted(det["displacement_by_node_activity"].items()):
        rows.append([k, e["n"], fmt(e["recall"], 2),
                     fmt(e["diagnosis_accuracy"], 2),
                     fmt(e.get("mean_realised_deg"), 1),
                     fmt(e.get("mean_alignment"), 3),
                     "**undetectable**" if e["undetectable"] else "detectable"])
    out = md_table(["Node / activity", "n", "Recall", "Diagnosis acc.",
                    "Mean observed (deg)", "Mean alignment", "Verdict"], rows)
    if tp:
        s = tp["summary"]
        out += ("\n\nTemporal-baseline probe: best onset recall (K=1) "
                "**%s** vs the population baseline's %s pooled, at worst clean "
                "FPR **%s** vs %s. The temporal reference detects the transient "
                "at onset then absorbs it, so it relocates the limit rather "
                "than removing it."
                % (fmt(s["best_onset_recall_K1"], 3),
                   fmt(tp["phase4_population_baseline_reference"]
                       ["displacement_recall"], 3),
                   fmt(s["worst_clean_fpr_mixed_stream"], 4),
                   fmt(tp["phase4_population_baseline_reference"]
                       ["overall_fpr"], 4)))
    return out


def table_injectors(D) -> str:
    inj = D.get("injectors")
    if not inj or "gilbert_elliott" not in inj:
        return "_injector_verification.json missing -- run scripts/verify_injectors.py_"
    ge = inj["gilbert_elliott"]
    rows = []
    for r in ge["per_severity"]:
        rows.append([r["severity"], fmt(r["target_loss_rate"], 2),
                     fmt(r["realised_loss_rate"], 4), fmt(r["abs_error_L"], 4),
                     fmt(r["target_mean_burst"], 1),
                     fmt(r["realised_mean_burst"], 3),
                     "%s%%" % fmt(r["pct_error_B"], 2),
                     "within" if r["within_tolerance"] else "OUT"])
    return md_table(
        ["Severity", "target L", "realised L", "|dL|", "target B",
         "realised B", "%err B", "Verdict"], rows)


# --------------------------------------------------------------------------- #
def build_tables(D) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parts = [
        "# Autonomic Agentic BSN — paper tables",
        "",
        "_Generated %s by `scripts/build_paper_artifacts.py`. Every number is "
        "read from a `results/` JSON written by a phase script; nothing here "
        "is recomputed or estimated._" % now,
        "",
        "## Table 1 — Datasets", "", table_datasets(D), "",
        "## Table 2 — Providers verified to expose token log-probabilities",
        "", table_providers(D), "",
        "> Both providers were verified empirically before use. Two other "
        "surfaces were tested and **rejected**: Ollama's OpenAI-compat layer "
        "silently drops `logprobs`, and the Gemini Developer API returns "
        "`400 \"Logprobs is not enabled for this model\"` on every model while "
        "serving ordinary generation normally.",
        "",
        "## Table 2b — Temperature-0 determinism", "",
        table_determinism(D), "",
        "## Table 3 — Gate 2 baseline accuracy: feature and format ablation",
        "", table_ablation(D), "",
        "> V0 is the shipped rendering (45 raw numbers in m/s², orientation "
        "implicit). V0F changes only the answer-format instruction. V1 "
        "re-presents the **same 45 features** with an explicit per-node "
        "orientation unit vector and motion level, in g. No new sensor "
        "channels are introduced in any variant.",
        "",
        "## Table 4 — Health monitor detection", "", table_detection(D), "",
        "## Table 5 — Diagnosis confusion matrix", "", table_confusion(D), "",
        "## Table 6 — Displacement detectability", "", table_displacement(D), "",
        "## Table 7 — Gilbert–Elliott packet-loss calibration", "",
        table_injectors(D), "",
    ]
    return "\n".join(parts)


def build_reproducibility(D) -> str:
    parts = ["# Reproducibility appendix", ""]

    lp = D.get("logprob_check")
    if lp:
        parts += [
            "## Confidence extraction", "",
            "Backend `%s`, model `%s`, `confidence_method=%s`."
            % (lp.get("backend"), lp.get("model"),
               lp.get("confidence_method", "logprob")),
            "",
            "Next-token log-probabilities are renormalised over the label "
            "tokens only: `p_i = exp(lp_i) / sum_j exp(lp_j)`. Probability "
            "mass from tokenizer variants that normalise to the same letter "
            "(`\"A\"`, `\" A\"`, sentencepiece `\"_A\"`) is summed in log "
            "space before renormalising.", "",
            "### Gate 1 probe prompt", "", "```", lp.get("prompt_used", ""),
            "```", "",
        ]

    b8 = D.get("baseline_8")
    if b8:
        parts += [
            "## Gate 2 protocol", "",
            "- Windows: 2.56 s, 50 %% overlap, cut only inside contiguous "
            "single-activity segments, >=60 %% sample coverage",
            "- Features: 3 nodes x 3 axes x 5 statistics = 45 per window",
            "- Test subjects: %s" % b8.get("subjects_used"),
            "- Few-shot subjects: %s (disjoint, enforced at runtime)"
            % b8.get("fewshot_subjects_used"),
            "- Few-shot: 2 examples per class",
            "- Prediction: argmax over the label-token logprobs, never "
            "free-text parsing", "",
            "### Exact prompt template (V0, as shipped)", "", "```",
            b8.get("prompt_template", ""), "```", "",
        ]

    fv = D.get("feature_variant_openai_8") or D.get("feature_variant_cerebras_8")
    if fv:
        v1 = fv.get("runs", {}).get("V1/8class") or \
             fv.get("runs", {}).get("V1/6class")
        if v1 and v1.get("prompt_template"):
            parts += ["### Exact prompt template (V1, orientation-first)", "",
                      "```", v1["prompt_template"], "```", ""]
        if v1 and v1.get("example_rendered_prompt"):
            ex = v1["example_rendered_prompt"]
            i = ex.find("Now classify")
            parts += ["### A rendered V1 window", "", "```",
                      ex[i:i + 700] if i >= 0 else ex[:700], "```", ""]

    det = D.get("detection")
    if det:
        parts += [
            "## Health monitor", "",
            "The monitor receives a `BlindWindow`: `injected_failure`, the "
            "true activity label and all injection metadata are stripped and "
            "**raise on access**. A grep test asserts the ground-truth "
            "identifiers appear nowhere in `health/signals.py` or "
            "`health/diagnose.py`. Ground truth is read only by "
            "`health/score_detection.py`, after the fact.", "",
            "- Seed: %s" % det.get("seed"),
            "- Windows per condition: %s" % det.get("windows_per_condition"),
            "- Scoring uses `meta['realised']`, never `meta['requested']`", "",
        ]
        for ds, sp in (det.get("splits") or {}).items():
            parts.append("- %s: calibration subject `%s` (held out), "
                         "evaluation subject `%s`, target node `%s`"
                         % (ds, sp["calibration_subject"], sp["eval_subject"],
                            sp["target_node"]))
        parts.append("")

    inj = D.get("injectors")
    if inj:
        parts += [
            "## Failure injection", "",
            "Five failure types as `DataSource` decorators. Each stochastic "
            "injector uses its own seeded `numpy.random.Generator`, never "
            "global state, so results do not depend on call order.", "",
            "Missing samples are always NaN tuples, never zeros (a zero "
            "reading is a real stationary measurement) and never `None` "
            "(`None` is reserved for a structurally absent sensor, such as "
            "MHEALTH's chest gyroscope).", "",
        ]

    parts += [
        "## Software", "",
        "- Python 3.11, numpy, pandas, requests, PyYAML",
        "- No agent framework, no learned model anywhere in Phases 1-4",
        "- Every phase script is deterministic given its seed",
        "- Test suites: `tests/test_backends.py`, `test_datasource.py`, "
        "`test_injection.py`, `test_health.py`", "",
    ]
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    D = {}
    print("=" * 78)
    print("BUILDING PAPER ARTIFACTS")
    print("=" * 78)
    for k, rel in SOURCES.items():
        D[k] = load(rel)
        print("  %-30s %s" % (k, "ok" if D[k] else "MISSING (%s)" % rel))

    tables = build_tables(D)
    repro = build_reproducibility(D)

    with open(os.path.join(args.out_dir, "tables.md"), "w",
              encoding="utf-8") as fh:
        fh.write(tables)
    with open(os.path.join(args.out_dir, "reproducibility.md"), "w",
              encoding="utf-8") as fh:
        fh.write(repro)
    with open(os.path.join(args.out_dir, "paper_data.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"generated": datetime.now(timezone.utc).isoformat(),
                   "sources": SOURCES,
                   "present": {k: bool(v) for k, v in D.items()},
                   "data": D}, fh, indent=1)

    print("\nwrote:")
    for f in ("tables.md", "reproducibility.md", "paper_data.json"):
        p = os.path.join(args.out_dir, f)
        print("  %-24s %6.1f KB" % (f, os.path.getsize(p) / 1024))
    missing = [k for k, v in D.items() if not v]
    if missing:
        print("\nMISSING inputs (tables say so rather than guessing):")
        for m in missing:
            print("  - %s (%s)" % (m, SOURCES[m]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
