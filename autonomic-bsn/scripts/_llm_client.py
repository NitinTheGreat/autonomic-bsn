"""Shared next-token logprob extraction for Phase 1.

The whole project rests on one capability: getting a REAL pre-sampling
next-token log-probability distribution out of a locally served LLM. This
module is the single place that talks to a backend, so check_logprobs.py and
check_baseline_accuracy.py provably use the identical code path (spec Step 2.7).

Backends
--------
llamacpp        POST /completion            n_probs + post_sampling_probs=false
vllm            POST /v1/completions        logprobs=N (+ echo)
ollama_native   POST /api/generate          logprobs=true / top_logprobs=N

Explicitly NOT supported, by design
-----------------------------------
Ollama's OpenAI-compat layer (/v1/chat/completions, /v1/completions on :11434).
It silently drops `logprobs` even when asked -- listed as unsupported in
Ollama's own OpenAI-compatibility docs, tracked by ollama/ollama#16117. It
answers 200 OK with a normal-looking body and no confidence data, so building
against it fails *invisibly*. _reject_ollama_compat() below hard-blocks it.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Iterable

import requests
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sentencepiece renders a leading space as U+2581 LOWER ONE EIGHTH BLOCK.
_SP_SPACE = "▁"


class LogprobError(RuntimeError):
    """Raised when a backend does not return usable logprob data.

    Always loud, never swallowed: a missing or empty logprob field is exactly
    the failure mode this phase exists to catch.
    """


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config(path: str | None = None) -> dict:
    path = path or os.path.join(REPO_ROOT, "configs", "models.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _reject_ollama_compat(endpoint: str) -> None:
    """Hard-fail if a backend is pointed at Ollama's /v1 compat layer."""
    if re.search(r":11434\b", endpoint) and "/v1/" in endpoint:
        raise LogprobError(
            "Refusing to use Ollama's OpenAI-compat endpoint: " + endpoint +
            "\nIt silently drops logprobs (ollama/ollama#16117). Use the "
            "native endpoint http://localhost:11434/api/generate instead."
        )


# --------------------------------------------------------------------------- #
# math
# --------------------------------------------------------------------------- #
def softmax_from_logprobs(logprobs: Iterable[float]) -> list[float]:
    """p_i = exp(lp_i) / sum_j exp(lp_j), max-shifted for stability.

    Renormalising over the returned top-k is intentional: the backend hands us
    a truncated distribution, and we want it to sum to exactly 1 over the
    candidates we actually scored.
    """
    lps = list(logprobs)
    if not lps:
        raise LogprobError("softmax over an empty logprob list")
    m = max(lps)
    exps = [math.exp(lp - m) for lp in lps]
    total = sum(exps)
    if total <= 0.0 or not math.isfinite(total):
        raise LogprobError("degenerate softmax denominator: " + repr(total))
    return [e / total for e in exps]


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


# --------------------------------------------------------------------------- #
# per-backend raw extraction -> list[(token, logprob)]
# --------------------------------------------------------------------------- #
def _extract_llamacpp(resp: dict) -> list[tuple[str, float]]:
    probs = resp.get("completion_probabilities")
    if not probs:
        raise LogprobError(
            "llama.cpp response has no `completion_probabilities`. Ensure the "
            "server is recent enough and that n_probs > 0 was sent."
        )
    first = probs[0]
    # Field naming drifted across llama.cpp versions:
    #   older: "probs":        [{"tok_str": "A", "prob": .., "logprob": ..}]
    #   newer: "top_logprobs": [{"token":   "A", "logprob": ..}]
    entries = first.get("probs") or first.get("top_logprobs")
    if not entries:
        raise LogprobError(
            "completion_probabilities[0] has neither `probs` nor "
            "`top_logprobs`. Got keys: " + repr(sorted(first))
        )
    out: list[tuple[str, float]] = []
    for e in entries:
        tok = e.get("tok_str", e.get("token"))
        lp = e.get("logprob")
        if lp is None and e.get("prob") is not None:
            # post_sampling_probs=true path. Recoverable, but post-sampling
            # values are temperature/top-p distorted, so we flag it upstream.
            p = float(e["prob"])
            lp = math.log(p) if p > 0 else -float("inf")
        if tok is None or lp is None:
            continue
        out.append((tok, float(lp)))
    if not out:
        raise LogprobError("llama.cpp returned candidates but none had a logprob")
    return out


def _extract_vllm(resp: dict) -> list[tuple[str, float]]:
    choices = resp.get("choices")
    if not choices:
        raise LogprobError("vLLM response has no `choices`: " + repr(sorted(resp)))
    lp_obj = choices[0].get("logprobs")
    if not lp_obj:
        raise LogprobError(
            "vLLM returned no `logprobs` object. Was `logprobs: N` sent, and is "
            "this really vLLM (not an Ollama compat shim)?"
        )
    top = lp_obj.get("top_logprobs")
    if not top:
        raise LogprobError(
            "vLLM logprobs has no `top_logprobs`: " + repr(sorted(lp_obj)))
    # With echo=true the array covers prompt tokens + the generated one; the
    # generated token's distribution is the last non-null entry.
    last = next((t for t in reversed(top) if t), None)
    if not last:
        raise LogprobError("vLLM top_logprobs contained only null entries")
    return [(tok, float(lp)) for tok, lp in last.items()]


def _extract_ollama_native(resp: dict) -> list[tuple[str, float]]:
    """Ollama native /api/generate.

    Native logprob support is a recent addition; older daemons answer 200 OK
    with no logprob field at all. Per spec we verify explicitly and fail loudly
    rather than degrading to free-text parsing.
    """
    lps = resp.get("logprobs")
    if not lps:
        raise LogprobError(
            "Ollama native /api/generate returned NO `logprobs` field.\n"
            "This daemon is too old for logprob support (or the option was "
            "ignored). Do NOT fall back to the /v1 compat layer -- it drops "
            "logprobs too (ollama/ollama#16117). Upgrade Ollama, or use "
            "llama.cpp / vLLM instead.\n"
            "Response keys were: " + repr(sorted(resp))
        )
    first = lps[0]
    top = first.get("top_logprobs")
    if not top:
        raise LogprobError(
            "Ollama returned `logprobs` but no per-token `top_logprobs` -- only "
            "the sampled token's own logprob is present, which is not a "
            "distribution. Send top_logprobs=N."
        )
    return [(t.get("token"), float(t.get("logprob"))) for t in top
            if t.get("token") is not None and t.get("logprob") is not None]


_EXTRACTORS = {
    "llamacpp": _extract_llamacpp,
    "vllm": _extract_vllm,
    "ollama_native": _extract_ollama_native,
}


# --------------------------------------------------------------------------- #
# request building
# --------------------------------------------------------------------------- #
def _build_payload(style: str, cfg_b: dict, prompt: str, n_probs: int,
                   temperature: float) -> dict:
    if style == "llamacpp":
        return {
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": n_probs,
            "post_sampling_probs": False,   # we want PRE-sampling logprobs
            "temperature": temperature,
            "cache_prompt": True,           # big win: few-shot prefix is reused
        }
    if style == "vllm":
        return {
            "model": cfg_b["model"],
            "prompt": prompt,
            "max_tokens": 1,
            "logprobs": n_probs,
            "echo": bool(cfg_b.get("echo", True)),
            "temperature": temperature,
        }
    if style == "ollama_native":
        return {
            "model": cfg_b["model"],
            "prompt": prompt,
            "stream": False,
            "logprobs": True,
            "top_logprobs": n_probs,
            "options": {"temperature": temperature, "num_predict": 1},
        }
    raise LogprobError("unknown backend style: " + repr(style))


def call_backend(cfg: dict, backend: str, prompt: str, n_probs: int) -> dict:
    """Call one backend. Returns {'candidates': [(tok, logprob)], 'raw': ...}."""
    if backend not in cfg.get("backends", {}):
        raise LogprobError("backend " + repr(backend) + " is not in models.yaml")
    cfg_b = cfg["backends"][backend]
    endpoint = cfg_b["endpoint"]
    _reject_ollama_compat(endpoint)
    style = cfg_b.get("style", backend)
    payload = _build_payload(style, cfg_b, prompt, n_probs,
                             float(cfg["request"]["temperature"]))
    try:
        r = requests.post(endpoint, json=payload,
                          timeout=float(cfg["request"]["timeout_s"]))
    except requests.RequestException as exc:
        raise LogprobError(
            backend + ": cannot reach " + endpoint + " -- " + str(exc)) from exc
    if r.status_code != 200:
        raise LogprobError(backend + ": HTTP " + str(r.status_code) + " from " +
                           endpoint + ": " + r.text[:300])
    try:
        resp = r.json()
    except json.JSONDecodeError as exc:
        raise LogprobError(
            backend + ": non-JSON response: " + r.text[:300]) from exc

    candidates = _EXTRACTORS[style](resp)
    return {"candidates": candidates, "raw": resp, "endpoint": endpoint,
            "model": cfg_b.get("model", backend), "backend": backend}


# --------------------------------------------------------------------------- #
# label scoring
# --------------------------------------------------------------------------- #
def norm_token(tok: str) -> str:
    """Map ' A', 'A', sentencepiece '_A', '\\nA' -> 'A'.

    Without this, a model that emits ' A' rather than 'A' looks like it never
    produced a label at all, and real probability mass is silently dropped.
    """
    return tok.replace(_SP_SPACE, " ").strip().upper()


def score_labels(cfg: dict, backend: str, prompt: str, letters: list[str],
                 n_probs: int) -> dict:
    """Softmax-renormalised distribution over `letters` for the next token.

    Mass from tokenizer variants that normalise to the same letter (e.g. 'A'
    and ' A') is summed in log space before renormalising.
    """
    call = call_backend(cfg, backend, prompt, n_probs)
    wanted = {l.upper() for l in letters}

    per_letter: dict[str, list[float]] = {}
    for tok, lp in call["candidates"]:
        key = norm_token(tok)
        if key in wanted:
            per_letter.setdefault(key, []).append(lp)

    if not per_letter:
        seen = [t for t, _ in call["candidates"]][:12]
        raise LogprobError(
            "None of the label tokens appeared in the returned top-k. Wanted " +
            repr(sorted(wanted)) + ", model's top tokens were " + repr(seen) +
            ". Raise n_probs, or the prompt is not steering the model to "
            "answer with a bare letter."
        )

    keys = sorted(per_letter)
    merged = [_logsumexp(per_letter[k]) for k in keys]
    probs = softmax_from_logprobs(merged)

    dist = {k: 0.0 for k in letters}      # letters absent from top-k -> 0.0
    dist.update({k: p for k, p in zip(keys, probs)})

    return {
        "distribution": dist,
        "argmax": max(dist, key=dist.get),
        "n_letters_seen": len(keys),
        "missing_letters": sorted(wanted - set(keys)),
        "backend": call["backend"],
        "endpoint": call["endpoint"],
        "model": call["model"],
        "raw_candidates": call["candidates"],
    }


def resolve_backends(cfg: dict, override: str | None = None) -> list[str]:
    """Ordered list of backends to try. 'auto' expands to fallback_order."""
    chosen = override or cfg.get("backend", "llamacpp")
    if chosen == "auto":
        return list(cfg.get("fallback_order", ["llamacpp"]))
    return [chosen]
