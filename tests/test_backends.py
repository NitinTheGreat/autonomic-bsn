#!/usr/bin/env python3
"""Phase 1 backend test suite.

Runs every backend's client code against mock_backends.py and asserts both the
success paths and -- more importantly -- that each failure mode fails LOUDLY
rather than silently returning a distribution that looks plausible.

No network access, no API keys, no research claims: this proves the client
code is correct, not that any model is good.

    python tests/test_backends.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import _llm_client as C  # noqa: E402

PORT = 8099
BASE = "http://127.0.0.1:%d" % PORT
LETTERS6 = ["A", "B", "C", "D", "E", "F"]

# Expected distribution from the shared fixture, computed once here so every
# logprob backend can be asserted against the SAME numbers.
EXPECTED_D = 0.783984

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


class Mock:
    """Run mock_backends.py in a given mode for the duration of a with-block."""

    def __init__(self, mode: str):
        self.mode = mode

    def __enter__(self):
        self.p = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "mock_backends.py"),
             self.mode, str(PORT)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(80):
            try:
                urllib.request.urlopen(
                    urllib.request.Request(BASE, data=b"{}"), timeout=1).read()
                break
            except urllib.error.HTTPError:
                break
            except Exception:
                time.sleep(0.05)
        return self

    def __exit__(self, *a):
        self.p.terminate()
        try:
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()
        return False


def cfg_for(backend: str, path: str, **over) -> dict:
    """Config pointed at the local mock, with the real backend definition."""
    cfg = C.load_config()
    b = cfg["backends"][backend]
    b["endpoint"] = BASE + path
    b.update(over)
    cfg["require_logprobs"] = False   # individual tests opt back in
    return cfg


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
def test_logprob_backends_agree():
    """Every logprob backend must yield the SAME distribution for the same
    fixture -- proving they share one softmax, not parallel implementations."""
    print("\n[1] logprob backends share one softmax path")
    os.environ["OPENAI_API_KEY"] = "test-key"
    results = {}
    for backend, mode, path in [
        ("llamacpp", "llamacpp_good", "/completion"),
        ("vllm", "vllm_good", "/v1/completions"),
        ("openai", "openai_good", "/v1/chat/completions"),
        ("gemini", "gemini_good", "/v1beta/models/{model}:generateContent"),
    ]:
        with Mock(mode):
            r = C.score_labels(cfg_for(backend, path), backend, "p", LETTERS6, 6)
        results[backend] = r
        check("%s parses and sums to 1.0" % backend,
              approx(sum(r["distribution"].values()), 1.0, 1e-6))
        check("%s argmax=D, p=%.6f" % (backend, r["distribution"]["D"]),
              r["argmax"] == "D" and approx(r["distribution"]["D"], EXPECTED_D))
        check("%s tagged confidence_method=logprob" % backend,
              r["confidence_method"] == "logprob")

    same = all(
        all(approx(results[b]["distribution"][k],
                   results["llamacpp"]["distribution"][k], 1e-12)
            for k in LETTERS6)
        for b in results)
    check("all four backends produce identical distributions", same,
          "shared softmax confirmed")


def test_openai_guards():
    print("\n[2] OpenAI failure guards")
    os.environ["OPENAI_API_KEY"] = "test-key"

    with Mock("openai_nologprobs"):
        try:
            C.score_labels(cfg_for("openai", "/v1/chat/completions"),
                           "openai", "p", LETTERS6, 6)
            check("missing `logprobs` fails loudly", False, "no error raised")
        except C.LogprobError as e:
            check("missing `logprobs` fails loudly", "logprobs" in str(e))
            check("error names the field, not a generic message",
                  "missing the `logprobs` field" in str(e))

    with Mock("openai_nolabels"):
        try:
            C.score_labels(cfg_for("openai", "/v1/chat/completions"),
                           "openai", "p", LETTERS6, 6)
            check("no label token in top_logprobs fails loudly", False)
        except C.LogprobError as e:
            msg = str(e)
            check("no label token in top_logprobs fails loudly",
                  "None of the label tokens" in msg)
            check("error lists the tokens actually returned",
                  "Answer" in msg or "The" in msg, "diagnosable prompt format")

    # missing key
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        C.score_labels(cfg_for("openai", "/v1/chat/completions"),
                       "openai", "p", LETTERS6, 6)
        check("missing OPENAI_API_KEY is actionable", False)
    except C.LogprobError as e:
        check("missing OPENAI_API_KEY is actionable",
              "OPENAI_API_KEY" in str(e) and "not set" in str(e))
    if saved:
        os.environ["OPENAI_API_KEY"] = saved


def test_openai_request_shape():
    print("\n[3] OpenAI request shape")
    os.environ["OPENAI_API_KEY"] = "test-key"
    with Mock("openai_good") as m:
        C.score_labels(cfg_for("openai", "/v1/chat/completions"),
                       "openai", "p", LETTERS6, 6)
        m.p.terminate()
        err = m.p.stderr.read().decode()
    import json as _j
    reqs = [_j.loads(l[4:]) for l in err.splitlines() if l.startswith("REQ ")]
    r = reqs[-1]
    check("sends logprobs: true", r["logprobs"] is True)
    check("sends top_logprobs", r["top_logprobs"] == 6)
    check("sends max_tokens: 1", r["max_tokens"] == 1)
    check("sends temperature 0", r["temperature"] == 0.0)
    check("uses bearer auth", r["auth"] is True)


def test_top_logprobs_cap():
    print("\n[4] top_logprobs capped at the API maximum of 20")
    os.environ["OPENAI_API_KEY"] = "test-key"
    with Mock("openai_good") as m:
        C.score_labels(cfg_for("openai", "/v1/chat/completions"),
                       "openai", "p", LETTERS6, 99)   # ask for far too many
        m.p.terminate()
        err = m.p.stderr.read().decode()
    import json as _j
    reqs = [_j.loads(l[4:]) for l in err.splitlines() if l.startswith("REQ ")]
    check("n_probs=99 is clamped to 20", reqs[-1]["top_logprobs"] == 20)


def test_anthropic_self_consistency():
    print("\n[5] Anthropic self-consistency voting")
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    with Mock("anthropic_votes") as m:
        cfg = cfg_for("anthropic", "/v1/messages", k_samples=10)
        r = C.score_labels(cfg, "anthropic", "p", LETTERS6, 1)
        m.p.terminate()
        err = m.p.stderr.read().decode()

    d = r["distribution"]
    check("vote fractions correct (7D/2E/1A of 10)",
          approx(d["D"], 0.7) and approx(d["E"], 0.2) and approx(d["A"], 0.1),
          "D=%.2f E=%.2f A=%.2f" % (d["D"], d["E"], d["A"]))
    check("distribution sums to 1.0", approx(sum(d.values()), 1.0, 1e-9))
    check("tagged confidence_method=self_consistency",
          r["confidence_method"] == "self_consistency")
    check("records k", r["k_samples"] == 10)
    check("records probability resolution 1/k",
          approx(r["probability_resolution"], 0.1),
          "finest resolvable step, matters for ECE binning")
    check("argmax is the modal vote", r["argmax"] == "D")

    import json as _j
    reqs = [_j.loads(l[4:]) for l in err.splitlines() if l.startswith("REQ ")]
    calls = [q for q in reqs if q.get("max_tokens") == 1]
    check("made k separate calls", len(calls) == 10, "%d calls" % len(calls))
    check("sampled at non-zero temperature", calls[-1]["temperature"] == 0.7,
          "temperature 0 would make all k samples identical")
    check("uses x-api-key auth", calls[-1]["x_api_key"] is True)
    check("sends anthropic-version header",
          calls[-1]["anthropic_version"] == "2023-06-01")

    with Mock("anthropic_garbage"):
        try:
            C.score_labels(cfg_for("anthropic", "/v1/messages", k_samples=5),
                           "anthropic", "p", LETTERS6, 1)
            check("all-invalid answers fail loudly", False)
        except C.LogprobError as e:
            check("all-invalid answers fail loudly",
                  "valid label letter" in str(e))


def test_require_logprobs_refusal():
    print("\n[6] require_logprobs refuses sampling backends")
    cfg = C.load_config()
    cfg["require_logprobs"] = True
    try:
        C.resolve_backends(cfg, "anthropic")
        check("anthropic refused when require_logprobs=true", False)
    except C.LogprobError as e:
        check("anthropic refused when require_logprobs=true",
              "self-consistency" in str(e))
        check("refusal explains why pooling is invalid",
              "never be pooled" in str(e))
    check("openai still allowed when require_logprobs=true",
          C.resolve_backends(cfg, "openai") == ["openai"])


def test_backend_selection():
    print("\n[7] backend selection and fallback chain")
    cfg = C.load_config()
    os.environ.pop("BSN_BACKEND", None)
    check("models.yaml default is openai", C.resolve_backends(cfg) == ["openai"])

    os.environ["BSN_BACKEND"] = "llamacpp"
    check("BSN_BACKEND overrides models.yaml",
          C.resolve_backends(cfg) == ["llamacpp"])
    check("explicit --backend beats BSN_BACKEND",
          C.resolve_backends(cfg, "vllm") == ["vllm"])
    os.environ["BSN_BACKEND"] = "auto"
    chain = C.resolve_backends(cfg)
    os.environ.pop("BSN_BACKEND", None)

    check("auto chain is all-logprob", chain == ["openai", "llamacpp", "vllm"],
          str(chain))
    check("anthropic NOT in fallback chain", "anthropic" not in chain,
          "silent logprob->sampling fallback would corrupt results")
    check("paused vertex NOT in fallback chain", "vertex" not in chain)
    check("paused gemini NOT in fallback chain", "gemini" not in chain)
    check("confidence_method_for is correct",
          C.confidence_method_for(cfg, "openai") == "logprob" and
          C.confidence_method_for(cfg, "anthropic") == "self_consistency")


def test_existing_guards_still_fire():
    print("\n[8] pre-existing guards not weakened")
    with Mock("llamacpp_uniform"):
        r = C.score_labels(cfg_for("llamacpp", "/completion"),
                           "llamacpp", "p", LETTERS6, 6)
        mx = max(r["distribution"].values())
        check("uniform distribution is detectable (anti-noise guard input)",
              approx(mx, 1.0 / 6, 1e-6), "max_prob=%.6f" % mx)

    with Mock("llamacpp_nofield"):
        try:
            C.score_labels(cfg_for("llamacpp", "/completion"),
                           "llamacpp", "p", LETTERS6, 6)
            check("llama.cpp missing field fails loudly", False)
        except C.LogprobError:
            check("llama.cpp missing field fails loudly", True)

    with Mock("gemini_nologprobs"):
        try:
            C.score_labels(cfg_for("gemini", "/v1beta/models/{model}:generateContent"),
                           "gemini", "p", LETTERS6, 6)
            check("gemini missing logprobsResult fails loudly", False)
        except C.LogprobError as e:
            check("gemini missing logprobsResult fails loudly",
                  "logprobsResult" in str(e))

    with Mock("ollama_nologprob"):
        try:
            C.score_labels(cfg_for("ollama_native", "/api/generate"),
                           "ollama_native", "p", LETTERS6, 6)
            check("ollama missing logprobs fails loudly", False)
        except C.LogprobError as e:
            check("ollama missing logprobs fails loudly", "logprobs" in str(e))

    cfg = C.load_config()
    cfg["backends"]["ollama_native"]["endpoint"] = \
        "http://localhost:11434/v1/chat/completions"
    try:
        C.score_labels(cfg, "ollama_native", "p", ["A"], 6)
        check("ollama /v1 compat endpoint hard-blocked", False)
    except C.LogprobError as e:
        check("ollama /v1 compat endpoint hard-blocked",
              "Refusing" in str(e) and "16117" in str(e))

    check("token normalisation still merges variants",
          C.norm_token(" A") == "A" and C.norm_token("▁A") == "A"
          and C.norm_token("a") == "A")


def test_uniform_guard_new_backends():
    """The anti-noise guard must fire for the NEW backends too, through the
    real check_logprobs.evaluate() -- not just at the score_labels level."""
    print()
    print("[9] anti-noise guard fires for the new backends")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import check_logprobs as CL

    os.environ["OPENAI_API_KEY"] = "test-key"
    with Mock("openai_uniform"):
        r = C.score_labels(cfg_for("openai", "/v1/chat/completions"),
                           "openai", "p", LETTERS6, 6)
    ok, reasons = CL.evaluate(r["distribution"], 0.05)
    check("openai uniform distribution REJECTED by the gate", not ok,
          "; ".join(reasons)[:70])
    check("rejection names it as noise, not real confidence",
          any("uniform" in x for x in reasons))

    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    with Mock("anthropic_uniform"):
        r = C.score_labels(cfg_for("anthropic", "/v1/messages", k_samples=6),
                           "anthropic", "p", LETTERS6, 1)
    ok, reasons = CL.evaluate(r["distribution"], 0.05)
    check("anthropic uniform vote REJECTED by the gate", not ok,
          "max=%.4f" % max(r["distribution"].values()))

    with Mock("openai_good"):
        r = C.score_labels(cfg_for("openai", "/v1/chat/completions"),
                           "openai", "p", LETTERS6, 6)
    ok, _ = CL.evaluate(r["distribution"], 0.05)
    check("a genuine peaked distribution still PASSES the gate", ok)


def main() -> int:
    print("=" * 74)
    print("PHASE 1 BACKEND TEST SUITE (mocks only -- no network, no keys)")
    print("=" * 74)
    C.load_dotenv()
    test_logprob_backends_agree()
    test_openai_guards()
    test_openai_request_shape()
    test_top_logprobs_cap()
    test_anthropic_self_consistency()
    test_require_logprobs_refusal()
    test_backend_selection()
    test_existing_guards_still_fire()
    test_uniform_guard_new_backends()

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
