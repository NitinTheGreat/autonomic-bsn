"""Thin adapter over Phase 1's verified logprob path.

It does NOT reimplement the softmax or the token normalisation. Those live in
`scripts/_llm_client.py`, were validated by Gate 1, and are shared by every
backend -- reimplementing them here would silently fork the confidence
definition and make Phase 5's numbers incomparable with Gate 1's.

What this adds is the extra confidence *readouts* Phase 5 needs, computed from
the same single call's candidate list:

    max_p        softmax maximum over the label tokens
    log_margin   top1_logprob - top2_logprob, in LOG space
    entropy      over the label distribution

`log_margin` matters because `max_p` saturates. Phase 1 measured gemma-4-31b at
max p 0.99999, where the softmax has no resolution left; the log-space margin is
still finite and still moves. The headroom probe decides which of these is a
given provider's primary signal, and that choice is recorded in every row.

`confidence_method` ("logprob" or "self_consistency") propagates into every
result. The two are not comparable and must never be pooled.
"""

from __future__ import annotations

import math
import os
import sys

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _llm_client import (  # noqa: E402
    LogprobError,
    call_backend,
    confidence_method_for,
    load_config,
    norm_token,
    resolve_backends,
    score_labels,
)


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


class LLMClientAdapter:
    """One resolved backend, exposing classify()."""

    def __init__(self, cfg: dict | None = None, backend: str | None = None):
        self.cfg = cfg or load_config()
        self.backend = resolve_backends(self.cfg, backend)[0]
        b = self.cfg["backends"][self.backend]
        model = b.get("model", self.backend)
        if b.get("model_env"):
            model = os.environ.get(b["model_env"], "").strip() or model
        self.model = model
        self.confidence_method = confidence_method_for(self.cfg, self.backend)
        self.n_probs = int(self.cfg["request"].get("step2_n_probs", 20))

    # ------------------------------------------------------------------ API --
    def classify(self, prompt: str, label_set: list[str]) -> dict:
        """One call -> label + the three confidence readouts."""
        if self.confidence_method != "logprob":
            # Sampling backends have no per-token logprobs, so log_margin and
            # entropy-over-logprobs do not exist. Report what does.
            r = score_labels(self.cfg, self.backend, prompt, label_set,
                             self.n_probs)
            dist = r["distribution"]
            ent = -sum(p * math.log(p) for p in dist.values() if p > 0)
            return {
                "distribution": dist, "predicted": r["argmax"],
                "max_p": max(dist.values()),
                "log_margin": None,          # undefined without logprobs
                "entropy": ent,
                "confidence_method": self.confidence_method,
                "provider": self.backend, "model": self.model,
                "k_samples": r.get("k_samples"),
                "probability_resolution": r.get("probability_resolution"),
            }

        call = call_backend(self.cfg, self.backend, prompt, self.n_probs)
        wanted = {l.upper() for l in label_set}

        per_letter: dict[str, list[float]] = {}
        for tok, lp in call["candidates"]:
            k = norm_token(tok)
            if k in wanted:
                per_letter.setdefault(k, []).append(lp)

        if not per_letter:
            # PORTABILITY GUARD. The shipped V0 prompt relied on the model
            # continuing after "Answer:"; gemma-4-31b instead answered with
            # 'Answer'/'To'/'Based'/'The'. Failing loudly with the tokens that
            # WERE returned makes that diagnosable; silently picking a
            # low-probability label would hide a broken prompt behind a
            # plausible number.
            seen = [t for t, _ in call["candidates"]][:12]
            raise LogprobError(
                "None of the label tokens appeared in the returned top-k. "
                "Wanted %r, model's top tokens were %r. The prompt is not "
                "steering this model to answer with a bare letter -- fix the "
                "prompt rather than trusting a fallback label."
                % (sorted(wanted), seen))

        merged = {k: _logsumexp(v) for k, v in per_letter.items()}
        ordered = sorted(merged.items(), key=lambda kv: -kv[1])
        top1 = ordered[0][1]
        top2 = ordered[1][1] if len(ordered) > 1 else None
        log_margin = (top1 - top2) if top2 is not None else None

        mx = max(merged.values())
        exps = {k: math.exp(v - mx) for k, v in merged.items()}
        tot = sum(exps.values())
        probs = {k: v / tot for k, v in exps.items()}
        # Labels the model never put in the top-k get an explicit 0.0 so the
        # distribution always covers the full label set.
        dist = {l: probs.get(l.upper(), 0.0) for l in label_set}
        ent = -sum(p * math.log(p) for p in dist.values() if p > 0)

        return {
            "distribution": dist,
            "predicted": ordered[0][0],
            "max_p": max(dist.values()),
            "log_margin": log_margin,
            "entropy": ent,
            "confidence_method": self.confidence_method,
            "provider": self.backend,
            "model": self.model,
            "n_labels_in_topk": len(merged),
            "missing_labels": sorted(wanted - set(merged)),
        }

    def describe(self) -> str:
        return ("provider=%s model=%s confidence_method=%s"
                % (self.backend, self.model, self.confidence_method))
