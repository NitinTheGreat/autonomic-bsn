#!/usr/bin/env python3
"""Phase 1, Step 1 -- can we extract REAL token log-probabilities?

This answers one yes/no question: does our local LLM serving stack hand back a
genuine next-token probability distribution we can use as a confidence signal?
If it does not, the entire "confidently wrong" research question is unmeasurable
and nothing else should be built.

Usage
-----
    python scripts/check_logprobs.py                 # backend from models.yaml
    python scripts/check_logprobs.py --backend auto  # try the fallback chain
    python scripts/check_logprobs.py --backend vllm

Writes results/phase1/logprob_check.json and exits 0 (PASS) or 1 (FAIL).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

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

# Six labels, one letter each -- deliberately a *fake* feature summary. Step 1
# tests the plumbing, not the science: we only need to know the numbers coming
# back are real model confidence and not a uniform placeholder.
LETTERS = ["A", "B", "C", "D", "E", "F"]

DUMMY_PROMPT = """You are classifying human activity from body-worn sensor features.

Legend:
A = lying
B = sitting
C = standing
D = walking
E = running
F = cycling

Sensor window features (3 accelerometers, m/s^2):
  wrist  : x mean -0.15 std 3.82 min -9.40 max 8.10 energy 14.9
  chest  : x mean  0.42 std 3.11 min -7.20 max 7.85 energy 10.1
  ankle  : x mean  1.05 std 5.94 min -12.6 max 13.2 energy 36.2
High variance on all three nodes with large ankle swing.

Answer with exactly one letter from the legend.
Answer:"""


def evaluate(dist: dict, margin: float) -> tuple[bool, list[str]]:
    """Apply the two spec'd assertions. Returns (passed, reasons)."""
    reasons: list[str] = []
    total = sum(dist.values())
    uniform = 1.0 / len(LETTERS)
    max_prob = max(dist.values())

    if abs(total - 1.0) > 1e-6:
        reasons.append(
            "distribution does not sum to 1.0 within 1e-6 (got %.12f)" % total)

    # A uniform distribution means we are reading noise, not model confidence.
    if max_prob <= uniform + margin:
        reasons.append(
            "distribution is ~uniform: max_prob %.4f does not exceed 1/%d "
            "(%.4f) by the required margin %.4f. This is the signature of "
            "reading noise rather than real confidence."
            % (max_prob, len(LETTERS), uniform, margin)
        )
    return (not reasons), reasons


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default=None,
                    help="llamacpp | vllm | ollama_native | auto")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="override output json path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    margin = float(cfg["request"].get("uniform_margin", 0.05))
    n_probs = int(cfg["request"].get("step1_n_probs", 6))
    out_path = args.out or os.path.join(
        REPO_ROOT, "results", "phase1", "logprob_check.json")

    try:
        order = resolve_backends(cfg, args.backend)
    except LogprobError as exc:
        print("RESULT: FAIL")
        print(exc)
        return 1

    print("=" * 72)
    print("PHASE 1 / STEP 1 -- token logprob extraction check")
    print("=" * 72)
    # Provenance banner: every run's log records which stack produced it.
    for b in order:
        print("resolved        : %s" % describe_backend(cfg, b))
    print("backends to try : %s" % ", ".join(order))
    print("n_probs         : %d   temperature: %s"
          % (n_probs, cfg["request"]["temperature"]))
    print()

    attempts: list[dict] = []
    winner = None

    for backend in order:
        ep = cfg.get("backends", {}).get(backend, {}).get("endpoint", "?")
        print("-> trying %-14s %s" % (backend, ep))
        try:
            res = score_labels(cfg, backend, DUMMY_PROMPT, LETTERS, n_probs)
        except LogprobError as exc:
            print("   FAILED: %s\n" % exc)
            attempts.append({"backend": backend, "endpoint": ep,
                             "ok": False, "error": str(exc)})
            continue

        passed, reasons = evaluate(res["distribution"], margin)
        attempts.append({"backend": backend, "endpoint": res["endpoint"],
                         "confidence_method": res.get("confidence_method"),
                         "ok": passed, "error": None if passed
                         else "; ".join(reasons)})
        if passed:
            print("   OK -- real, non-uniform distribution\n")
            winner = res
            break
        print("   REJECTED: %s\n" % "; ".join(reasons))

    # ---------------------------------------------------------------- report --
    if winner is None:
        notes = ("No backend produced a usable non-uniform logprob "
                 "distribution. Start llama.cpp with: llama-server -m "
                 "<path-to-gguf> -c 4096 --port 8080")
        payload = {
            "backend": None,
            "endpoint": None,
            "model": None,
            "pass": False,
            "distribution": {},
            "max_prob": None,
            "notes": notes,
            "attempts": attempts,
        }
        _write(out_path, payload)
        print("RESULT: FAIL")
        print(notes)
        print("\nwrote %s" % out_path)
        return 1

    dist = winner["distribution"]
    max_prob = max(dist.values())
    total = sum(dist.values())

    print("parsed distribution (softmax-renormalised over the 6 label tokens):")
    for letter in LETTERS:
        p = dist[letter]
        bar = "#" * int(round(p * 50))
        print("   %s  %.6f  %s" % (letter, p, bar))
    print()
    print("   sum      = %.12f   (tolerance 1e-6)" % total)
    print("   max_prob = %.6f   (uniform would be %.6f)"
          % (max_prob, 1.0 / len(LETTERS)))
    if winner["missing_letters"]:
        print("   note: letters absent from top-k -> %s"
              % ", ".join(winner["missing_letters"]))
    print()

    notes = (
        "Softmax renormalised over the %d label tokens returned by %s. "
        "Sum-to-1 within 1e-6 and max_prob exceeds uniform (1/6) by >= %.3f, "
        "so these are real pre-sampling model logprobs, not noise. "
        "Later phases MUST use backend '%s'."
        % (len(LETTERS), winner["backend"], margin, winner["backend"])
    )
    payload = {
        "backend": winner["backend"],
        "endpoint": winner["endpoint"],
        "model": winner["model"],
        "confidence_method": winner.get("confidence_method", "logprob"),
        "pass": True,
        "distribution": dist,
        "max_prob": max_prob,
        "notes": notes,
        "attempts": attempts,
        "prompt_used": DUMMY_PROMPT,
        "raw_candidates": [[t, lp] for t, lp in winner["raw_candidates"]],
    }
    _write(out_path, payload)

    print("RESULT: PASS")
    print("backend '%s' (%s) returns usable logprobs." %
          (winner["backend"], winner["model"]))
    print("wrote %s" % out_path)
    return 0


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    # Mirror into frontend/ so `cd frontend && python -m http.server 8000` can
    # fetch it (http.server refuses to serve paths above its root).
    mirror = os.path.join(REPO_ROOT, "frontend", "results", "phase1",
                          os.path.basename(path))
    os.makedirs(os.path.dirname(mirror), exist_ok=True)
    with open(mirror, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
