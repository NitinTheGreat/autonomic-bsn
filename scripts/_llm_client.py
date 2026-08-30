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
import time
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
def load_dotenv(path: str | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ (no new dependency).

    Real environment variables always win, so an exported key overrides the
    file. Missing .env is fine -- keys can be exported instead.
    """
    path = path or os.path.join(REPO_ROOT, ".env")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val


def load_config(path: str | None = None) -> dict:
    load_dotenv()
    path = path or os.path.join(REPO_ROOT, "configs", "models.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Remember each backend's shipped endpoint so call_backend can tell an
    # untouched default (which a *_BASE_URL env var may override) from one a
    # caller has deliberately repointed (which must win).
    for b in (cfg.get("backends") or {}).values():
        if isinstance(b, dict) and "endpoint" in b:
            b["_default_endpoint"] = b["endpoint"]
    return cfg


def require_api_key(backend: str, key_env: str) -> str:
    """Read an API key from the environment, or fail with the exact fix."""
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise LogprobError(
            backend + ": environment variable " + key_env + " is not set.\n"
            "Copy .env.example to .env and put your key there, or export it:\n"
            "  export " + key_env + "=your-key-here\n"
            "(PowerShell:  $env:" + key_env + " = 'your-key-here')")
    return api_key


# 429 bodies that mean "this will never succeed on its own" rather than
# "slow down". Retrying these only delays an unavoidable, actionable error.
_PERMANENT_429 = ("insufficient_quota", "credit_balance_exhausted",
                  "no credits remaining", "billing_hard_limit_reached",
                  "exceeded your current quota")


def _is_permanent_quota_failure(body: str) -> bool:
    low = (body or "").lower()
    return any(m in low for m in _PERMANENT_429)


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


def _extract_gemini(resp: dict) -> list[tuple[str, float]]:
    """Google Gemini generateContent with responseLogprobs.

    IMPORTANT MODEL RESTRICTION
    ---------------------------
    logprobs works on gemini-2.5-flash / 2.5-pro. The Gemini 3.x family
    (3-flash, 3-pro and later) does NOT support it: the API either rejects the
    request with "Logprobs is not supported for this model" or returns a
    candidate with logprobsResult absent / null. That is a silent-confidence
    failure of exactly the kind this phase exists to catch, so we fail loudly
    and name the fix rather than degrading to free-text parsing.
    """
    if "error" in resp:
        err = resp["error"]
        raise LogprobError(
            "Gemini API error " + str(err.get("code", "?")) + ": " +
            str(err.get("message", err))[:400] +
            "\nIf this mentions logprobs not being supported, the model does "
            "not expose them -- switch to gemini-2.5-flash."
        )
    cands = resp.get("candidates")
    if not cands:
        raise LogprobError(
            "Gemini response has no `candidates` (blocked by a safety filter?): "
            + repr(sorted(resp))[:300])

    cand = cands[0]
    lr = cand.get("logprobsResult")
    if not lr:
        finish = cand.get("finishReason", "?")
        raise LogprobError(
            "Gemini returned NO `logprobsResult` (finishReason=" + str(finish) +
            ").\nThis model does not support logprobs. The Gemini 3.x family "
            "(gemini-3-flash, gemini-3-pro) silently omits this field or "
            "returns null.\nUse a model that does support it -- "
            "gemini-2.5-flash is the documented, working choice -- or fall "
            "back to the local llama.cpp backend.\n"
            "Set backends.gemini.model in configs/models.yaml."
        )
    top = lr.get("topCandidates")
    if not top:
        raise LogprobError(
            "Gemini logprobsResult has no `topCandidates` (only the chosen "
            "token, which is not a distribution). Send logprobs: N with N>=2. "
            "Keys were: " + repr(sorted(lr)))

    first = top[0].get("candidates")
    if not first:
        raise LogprobError("Gemini topCandidates[0] has no `candidates` list")

    out: list[tuple[str, float]] = []
    for c in first:
        tok, lp = c.get("token"), c.get("logProbability")
        if tok is not None and lp is not None:
            out.append((tok, float(lp)))
    if not out:
        raise LogprobError("Gemini returned candidates but none had a "
                           "logProbability")
    return out


def _extract_openai(resp: dict) -> list[tuple[str, float]]:
    """OpenAI chat/completions with logprobs: true, top_logprobs: N.

    `top_logprobs` is capped at 20 by the API, which is why 20 is our default
    and why the 8-label set (A-H) fits comfortably inside one request.
    """
    choices = resp.get("choices")
    if not choices:
        raise LogprobError(
            "OpenAI response has no `choices`: " + repr(sorted(resp))[:300])

    lp_obj = choices[0].get("logprobs")
    if not lp_obj:
        raise LogprobError(
            "OpenAI response is missing the `logprobs` field.\n"
            "Was `\"logprobs\": true` (and `top_logprobs`) sent, and does this "
            "model support them? Refusing to continue -- returning a uniform "
            "or single-token distribution here would silently destroy the "
            "confidence signal this project measures."
        )

    content = lp_obj.get("content")
    if not content:
        raise LogprobError(
            "OpenAI `logprobs` object has no `content` array (the model may "
            "have returned no tokens). Keys were: " + repr(sorted(lp_obj)))

    top = content[0].get("top_logprobs")
    if not top:
        raise LogprobError(
            "OpenAI logprobs.content[0] has no `top_logprobs` -- only the "
            "chosen token, which is not a distribution. Send top_logprobs: N.")

    out = [(e["token"], float(e["logprob"])) for e in top
           if e.get("token") is not None and e.get("logprob") is not None]
    if not out:
        raise LogprobError("OpenAI returned top_logprobs but none had a logprob")
    return out


_EXTRACTORS = {
    "llamacpp": _extract_llamacpp,
    "vllm": _extract_vllm,
    "ollama_native": _extract_ollama_native,
    "gemini": _extract_gemini,
    # Vertex returns the same generateContent body as the Developer API, so the
    # response parsing is identical; only the URL and the auth differ.
    "vertex": _extract_gemini,
    "openai": _extract_openai,
}

# Styles that yield a genuine model-reported log-probability distribution.
# `anthropic` is deliberately absent: see _score_self_consistency().
LOGPROB_STYLES = frozenset(_EXTRACTORS)


# --------------------------------------------------------------------------- #
# Vertex AI auth
# --------------------------------------------------------------------------- #
_VERTEX_CREDS = None      # cached across calls: a run makes ~150 requests


def _vertex_token() -> str:
    """OAuth2 access token for Vertex AI, from a service account or ADC.

    Resolution order is google-auth's own default():
      1. GOOGLE_APPLICATION_CREDENTIALS -> service-account JSON key
      2. gcloud application-default credentials
      3. the attached service account, on GCP compute
    The credential object is cached and refreshed only when expired, so a
    150-window accuracy run does not re-mint a token per request.
    """
    global _VERTEX_CREDS
    try:
        import google.auth
        import google.auth.transport.requests
        from google.auth.exceptions import DefaultCredentialsError
    except ImportError as exc:
        raise LogprobError(
            "vertex: google-auth is not installed. Run:\n"
            "  pip install -r requirements.txt") from exc

    if _VERTEX_CREDS is None:
        try:
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"])
        except DefaultCredentialsError as exc:
            raise LogprobError(
                "vertex: no Google Cloud credentials found.\n"
                "Set up ONE of these:\n"
                "  A) Service account (no gcloud CLI needed):\n"
                "     - create a key in the GCP console, save the JSON\n"
                "     - grant it the 'Vertex AI User' role\n"
                "     - set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json "
                "in .env\n"
                "  B) gcloud CLI:\n"
                "     gcloud auth application-default login\n"
                "\nOriginal error: " + str(exc)[:200]) from exc
        _VERTEX_CREDS = creds

    if not _VERTEX_CREDS.valid:
        _VERTEX_CREDS.refresh(
            __import__("google.auth.transport.requests",
                       fromlist=["Request"]).Request())
    return _VERTEX_CREDS.token


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
    if style == "openai":
        return {
            "model": cfg_b["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": temperature,
            "logprobs": True,
            # capped at 20 by the API; 20 comfortably covers the 8 label tokens
            "top_logprobs": min(int(n_probs), 20),
        }
    if style == "anthropic":
        # One sample of a self-consistency vote. Temperature comes from the
        # backend config, NOT the global temperature: voting at temperature 0
        # would return the same token k times and produce a degenerate
        # all-or-nothing distribution.
        return {
            "model": cfg_b["model"],
            "max_tokens": 1,
            "temperature": float(cfg_b.get("sample_temperature", 0.7)),
            "messages": [{"role": "user", "content": prompt}],
        }
    if style in ("gemini", "vertex"):
        # The API caps `logprobs` at 20; 8 label letters fit comfortably.
        gen: dict = {
            "temperature": temperature,
            "maxOutputTokens": 1,
            "responseLogprobs": True,
            "logprobs": min(int(n_probs), 20),
        }
        # 2.5-series models spend output tokens on internal "thinking" by
        # default, which would consume our single allowed token and return an
        # empty candidate. Explicitly disable it.
        budget = cfg_b.get("thinking_budget", 0)
        if budget is not None:
            gen["thinkingConfig"] = {"thinkingBudget": int(budget)}
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen,
        }
    raise LogprobError("unknown backend style: " + repr(style))


def call_backend(cfg: dict, backend: str, prompt: str, n_probs: int) -> dict:
    """Call one backend. Returns {'candidates': [(tok, logprob)], 'raw': ...}."""
    if backend not in cfg.get("backends", {}):
        raise LogprobError("backend " + repr(backend) + " is not in models.yaml")
    cfg_b = cfg["backends"][backend]
    style = cfg_b.get("style", backend)

    # Env overrides for hosted providers, so a base URL or model can be
    # switched in .env without editing a tracked file.
    if cfg_b.get("model_env"):
        env_model = os.environ.get(cfg_b["model_env"], "").strip()
        if env_model:
            cfg_b = {**cfg_b, "model": env_model}
    if cfg_b.get("base_url_env"):
        env_base = os.environ.get(cfg_b["base_url_env"], "").strip()
        # Only override the endpoint the YAML shipped with. A caller that has
        # explicitly repointed `endpoint` at runtime -- a test mock, a local
        # proxy -- must win, or the env var silently drags requests back to the
        # real (paid) API behind their back.
        pinned = cfg_b.get("endpoint") != cfg_b.get("_default_endpoint",
                                                    cfg_b.get("endpoint"))
        if env_base and not pinned:
            cfg_b = {**cfg_b,
                     "endpoint": env_base.rstrip("/") + cfg_b.get("path", "")}

    # Hosted endpoints carry the model (and for Vertex, the project/location)
    # in their path. Values may come from config or the environment.
    project = (cfg_b.get("project")
               or os.environ.get("GOOGLE_CLOUD_PROJECT", "")).strip()
    location = (cfg_b.get("location")
                or os.environ.get("GOOGLE_CLOUD_LOCATION", "")
                or "us-central1").strip()
    endpoint = (cfg_b["endpoint"]
                .replace("{model}", str(cfg_b.get("model", "")))
                .replace("{project}", project)
                .replace("{location}", location))
    _reject_ollama_compat(endpoint)

    headers: dict[str, str] = {}

    if style == "vertex":
        if not project:
            raise LogprobError(
                "vertex: no GCP project set. Put your project id in .env as\n"
                "  GOOGLE_CLOUD_PROJECT=my-project-id\n"
                "or set backends.vertex.project in configs/models.yaml.")
        headers["Authorization"] = "Bearer " + _vertex_token()
    else:
        key_env = cfg_b.get("api_key_env")
        if key_env:
            api_key = require_api_key(backend, key_env)
            # Each provider wants the key in a different header.
            auth_style = cfg_b.get("auth_style", "x-goog-api-key")
            if auth_style == "bearer":
                headers["Authorization"] = "Bearer " + api_key
            elif auth_style == "x-api-key":
                headers["x-api-key"] = api_key
            else:
                headers["x-goog-api-key"] = api_key
    for k, v in (cfg_b.get("extra_headers") or {}).items():
        headers[str(k)] = str(v)

    payload = _build_payload(style, cfg_b, prompt, n_probs,
                             float(cfg["request"]["temperature"]))
    # Bounded retry on rate limits and transient server errors. A baseline run
    # makes ~150 sequential calls, and free tiers rate-limit well below that --
    # without this, a single 429 aborts the whole gate. Retries are capped and
    # never silent: exhausting them still raises.
    rq = cfg.get("request", {})
    max_retries = int(rq.get("max_retries", 5))
    backoff = float(rq.get("retry_backoff_s", 2.0))
    r = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(endpoint, json=payload, headers=headers,
                              timeout=float(rq["timeout_s"]))
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise LogprobError(backend + ": cannot reach " + endpoint +
                                   " -- " + str(exc)) from exc
            time.sleep(backoff * (2 ** attempt))
            continue
        # A 429 is usually a rate limit and worth retrying -- but an exhausted
        # credit balance also returns 429, and no amount of backoff fixes that.
        # Retrying it just burns six sleeps before reporting the same thing.
        if r.status_code == 429 and _is_permanent_quota_failure(r.text):
            break
        if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            # Honour Retry-After when the server sends one.
            wait = backoff * (2 ** attempt)
            hdr = r.headers.get("retry-after")
            if hdr:
                try:
                    wait = max(wait, float(hdr))
                except ValueError:
                    pass
            time.sleep(wait)
            continue
        break

    if r.status_code != 200:
        hint = ""
        if r.status_code == 429 and _is_permanent_quota_failure(r.text):
            hint = ("\nThis is a BILLING failure, not a rate limit -- retrying "
                    "will not help. Add credits, or switch provider with "
                    "LLM_PROVIDER=cerebras (free) or LLM_PROVIDER=llamacpp "
                    "(local).")
        # Surface the API's own message: it names the real cause (bad key,
        # quota, or "Logprobs is not supported for this model").
        raise LogprobError(backend + ": HTTP " + str(r.status_code) + " from " +
                           endpoint + ": " + r.text[:400] + hint)
    try:
        resp = r.json()
    except json.JSONDecodeError as exc:
        raise LogprobError(
            backend + ": non-JSON response: " + r.text[:300]) from exc

    # Sampling backends (anthropic) have no logprobs to extract by design --
    # the caller reads `raw` instead. Only logprob styles get an extractor.
    candidates = _EXTRACTORS[style](resp) if style in _EXTRACTORS else []
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
    """Probability distribution over `letters` for the model's answer token.

    Single entry point for every backend. Two confidence methods exist and the
    result always says which one produced it, via `confidence_method`:

      "logprob"          the model's own next-token log-probabilities,
                         softmax-renormalised over the label tokens.
      "self_consistency" k sampled answers, distribution = vote fractions.
                         Only the `anthropic` style, which has no logprobs.

    These are NOT interchangeable and must never be pooled in one analysis.
    """
    style = cfg.get("backends", {}).get(backend, {}).get("style", backend)
    if style == "anthropic":
        return _score_self_consistency(cfg, backend, prompt, letters)
    return _score_logprob(cfg, backend, prompt, letters, n_probs)


def _score_logprob(cfg: dict, backend: str, prompt: str, letters: list[str],
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
        "confidence_method": "logprob",
        "n_letters_seen": len(keys),
        "missing_letters": sorted(wanted - set(keys)),
        "backend": call["backend"],
        "endpoint": call["endpoint"],
        "model": call["model"],
        "raw_candidates": call["candidates"],
    }


def _score_self_consistency(cfg: dict, backend: str, prompt: str,
                            letters: list[str]) -> dict:
    """Sampling-based confidence for backends with NO logprobs (Anthropic).

    The Anthropic Messages API exposes no logprobs or top_logprobs -- there is
    no parameter for it and no model that enables it. So this backend CANNOT
    satisfy Gate 1, and is not a drop-in substitute for a logprob backend.

    Instead we estimate confidence by self-consistency: sample the same prompt
    k times at a non-zero temperature and use the vote fraction per label as
    the probability. This is a legitimate, literature-supported estimator and
    is empirically better calibrated than asking a model to verbalise its own
    confidence -- but it costs k x inference, and its resolution is bounded by
    1/k (with k=10 the finest distinguishable confidence step is 0.1, which
    matters when binning for ECE later).

    Treat it as a ROBUSTNESS CHECK against the logprob results, never as the
    primary method, and never pool the two in a single analysis.
    """
    cfg_b = cfg["backends"][backend]
    k = int(cfg_b.get("k_samples", 10))
    wanted = {l.upper() for l in letters}

    votes: dict[str, int] = {l.upper(): 0 for l in letters}
    raw_answers: list[str] = []
    unparsed = 0

    for _ in range(k):
        call = call_backend(cfg, backend, prompt, 1)
        text = _anthropic_text(call["raw"])
        raw_answers.append(text)
        key = norm_token(text)[:1]        # first char, normalised
        if key in wanted:
            votes[key] += 1
        else:
            unparsed += 1

    counted = k - unparsed
    if counted == 0:
        raise LogprobError(
            "anthropic: none of the " + str(k) + " sampled answers contained a "
            "valid label letter. Wanted " + repr(sorted(wanted)) + ", got " +
            repr(raw_answers[:10]) + ". The prompt is not steering the model "
            "to answer with a bare letter.")

    dist = {l: votes[l.upper()] / counted for l in letters}

    return {
        "distribution": dist,
        "argmax": max(dist, key=dist.get),
        "confidence_method": "self_consistency",
        "k_samples": k,
        # Resolution floor: vote fractions are multiples of 1/counted.
        "probability_resolution": 1.0 / counted,
        "n_valid_samples": counted,
        "n_unparsed_samples": unparsed,
        "sample_temperature": float(cfg_b.get("sample_temperature", 0.7)),
        "n_letters_seen": sum(1 for v in votes.values() if v),
        "missing_letters": sorted(l for l in wanted if not votes[l]),
        "backend": backend,
        "endpoint": cfg_b["endpoint"],
        "model": cfg_b.get("model", backend),
        "raw_answers": raw_answers,
    }


def _anthropic_text(resp: dict) -> str:
    """Pull the answer text out of an Anthropic Messages response."""
    if "error" in resp:
        raise LogprobError("anthropic API error: " + str(resp["error"])[:300])
    for b in (resp.get("content") or []):
        if b.get("type") == "text" and b.get("text"):
            return str(b["text"])
    return ""


def confidence_method_for(cfg: dict, backend: str) -> str:
    """Which confidence method a backend yields, without calling it."""
    style = cfg.get("backends", {}).get(backend, {}).get("style", backend)
    return "logprob" if style in LOGPROB_STYLES else "self_consistency"


def resolve_backends(cfg: dict, override: str | None = None) -> list[str]:
    """Ordered list of backends to try.

    Precedence: explicit argument (a --backend flag) > BSN_BACKEND env var >
    `backend:` in models.yaml. The env var lets the user switch stacks without
    editing a tracked file.

    'auto' expands to fallback_order, which contains ONLY logprob backends.
    `anthropic` is deliberately excluded from that chain: silently falling back
    from a logprob method to a sampling method would change what the numbers
    mean with no visible signal. It must be selected explicitly.
    """
    # LLM_PROVIDER and BSN_BACKEND are equivalent; either selects the backend.
    chosen = (override
              or os.environ.get("BSN_BACKEND", "").strip()
              or os.environ.get("LLM_PROVIDER", "").strip()
              or cfg.get("backend", "llamacpp"))

    if cfg.get("require_logprobs") and chosen != "auto":
        if confidence_method_for(cfg, chosen) != "logprob":
            raise LogprobError(
                "Backend '" + chosen + "' provides confidence by "
                "self-consistency sampling, not model log-probabilities, but "
                "`require_logprobs: true` is set in configs/models.yaml.\n"
                "Refusing to run: these two are not interchangeable and their "
                "numbers must never be pooled in one analysis.\n"
                "Either pick a logprob backend (openai, llamacpp, vllm), or "
                "set require_logprobs: false to accept sampling-based "
                "confidence deliberately.")

    if chosen == "auto":
        return list(cfg.get("fallback_order", ["llamacpp"]))
    return [chosen]


def describe_backend(cfg: dict, backend: str) -> str:
    """One-line provenance banner: which stack produced this run's numbers."""
    cfg_b = cfg.get("backends", {}).get(backend, {})
    return ("backend=%s  model=%s  confidence_method=%s"
            % (backend, cfg_b.get("model", "?"),
               confidence_method_for(cfg, backend)))
