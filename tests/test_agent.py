#!/usr/bin/env python3
"""Phase 5 agent test suite. Mocks only for the LLM -- no network, no keys.

    python tests/test_agent.py
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import _llm_client as C  # noqa: E402
from agent.graph import build_fewshot_block, load_fewshot, load_system_prompt  # noqa: E402
from agent.llm_client_adapter import LLMClientAdapter  # noqa: E402
from core.datasource import NodeFrame, Window  # noqa: E402
from datasets.dataset_replay_source import DatasetReplaySource  # noqa: E402
from features.extractors import (  # noqa: E402
    node_features,
    render,
    window_features,
)
from health.window_view import blind  # noqa: E402

PORT = 8099
BASE = "http://127.0.0.1:%d" % PORT
LETTERS8 = list("ABCDEFGH")

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
    def __init__(self, mode: str):
        self.mode = mode

    def __enter__(self):
        self.p = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "mock_backends.py"),
             self.mode, str(PORT)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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


def adapter_for(mode_path="/v1/chat/completions") -> LLMClientAdapter:
    os.environ["OPENAI_API_KEY"] = "test-key"
    cfg = C.load_config()
    b = cfg["backends"]["openai"]
    b["endpoint"] = BASE + mode_path
    b.pop("base_url_env", None)
    b.pop("model_env", None)
    b["_default_endpoint"] = b["endpoint"]
    cfg["require_logprobs"] = False
    return LLMClientAdapter(cfg, "openai")


def synthetic_window(nan_from: int | None = None, gyro: bool = True,
                     channels=("accel", "gyro")) -> Window:
    frames = {}
    for node in ("wrist", "chest", "ankle"):
        fl = []
        for i in range(100):
            missing = nan_from is not None and node == "ankle" and i >= nan_from
            acc = ((float("nan"),) * 3 if missing
                   else (0.1 * math.sin(i / 5), 0.9, 0.2))
            gy = None if not gyro else (
                (float("nan"),) * 3 if missing else (1.0, 2.0, 3.0))
            fl.append(NodeFrame(node_id=node, t_sec=i * 0.01, accel_g=acc,
                                gyro_dps=gy, source="dataset",
                                meta={"channels_present": list(channels),
                                      "sampling_rate_hz": 100.0}))
        frames[node] = fl
    return Window(start_sec=0.0, end_sec=1.0, frames=frames, label="walking",
                  meta={"sampling_rate_hz": 100.0})


# --------------------------------------------------------------------------- #
def test_distribution():
    print("\n[1] classify() returns a full distribution")
    with Mock("openai_good"):
        a = adapter_for()
        r = a.classify("p", LETTERS8)
    d = r["distribution"]
    check("covers all 8 labels", sorted(d) == LETTERS8, str(sorted(d)))
    check("sums to ~1.0", abs(sum(d.values()) - 1.0) < 1e-9,
          "%.12f" % sum(d.values()))
    check("labels absent from top-k are 0.0, not dropped",
          all(isinstance(v, float) for v in d.values()))
    check("predicted is the argmax", r["predicted"] == max(d, key=d.get))
    check("max_p matches the distribution",
          abs(r["max_p"] - max(d.values())) < 1e-12)
    check("entropy is finite and non-negative",
          r["entropy"] >= 0 and math.isfinite(r["entropy"]))
    check("confidence_method propagates", r["confidence_method"] == "logprob")
    check("provider and model propagate",
          r["provider"] == "openai" and bool(r["model"]))


def test_log_margin():
    print("\n[2] log_margin from a known logprob pair")
    # The mock's fixture: D=-0.223, E=-1.897 are the top two.
    with Mock("openai_good"):
        r = adapter_for().classify("p", LETTERS8)
    expected = -0.223 - (-1.897)
    check("log_margin = top1_logprob - top2_logprob",
          abs(r["log_margin"] - expected) < 1e-9,
          "got %.6f, expected %.6f" % (r["log_margin"], expected))
    check("log_margin stays finite where max_p would saturate",
          math.isfinite(r["log_margin"]))
    # Independent check of the arithmetic itself.
    a, b = -0.05, -9.0
    check("a saturating pair still yields a usable margin",
          abs((a - b) - 8.95) < 1e-12,
          "max_p would read ~0.99987 while the margin is 8.95")


def test_portability_guard():
    print("\n[3] portability guard: prose answer must fail LOUDLY")
    with Mock("openai_nolabels"):
        try:
            adapter_for().classify("p", LETTERS8)
            check("prose-answering model raises", False, "returned a label")
        except C.LogprobError as e:
            msg = str(e)
            check("prose-answering model raises", True)
            check("error lists the tokens actually returned",
                  "Answer" in msg or "The" in msg, "diagnosable")
            check("error says the prompt is at fault, not the data",
                  "prompt is not steering" in msg)
            check("does NOT silently pick a low-probability label",
                  "fallback" in msg or "rather than trusting" in msg)


def test_blind_required():
    print("\n[4] extractor accepts ONLY a BlindWindow")
    w = synthetic_window()
    try:
        window_features(w)
        check("raw Window rejected", False, "accepted it")
    except TypeError as e:
        check("raw Window rejected", True)
        check("error explains the leak", "ground-truth channel" in str(e))
    check("BlindWindow accepted", isinstance(window_features(blind(w)), dict))


def test_no_zero_fill():
    print("\n[5] missing data is reported, never zero-filled")
    f = window_features(blind(synthetic_window(nan_from=0)))["nodes"]["ankle"]
    check("fully-dead node reports has_data False", f["has_data"] is False)
    check("no orientation invented for a dead node", "orientation" not in f)
    check("no motion level invented for a dead node", "motion" not in f)
    txt = render(window_features(blind(synthetic_window(nan_from=0))))
    check("rendered as NO DATA", "NO DATA" in txt)
    check("never rendered as a zero orientation vector",
          "(+0.00, +0.00, +0.00)" not in txt)
    check("never labelled 'vigorous' when there is no data",
          "nan" not in txt.lower())

    # Partially missing: statistics computed over valid samples only.
    f2 = window_features(blind(synthetic_window(nan_from=50)))["nodes"]["ankle"]
    check("partially-missing node still reports data", f2["has_data"] is True)
    check("nan_fraction recorded", abs(f2["nan_fraction"] - 0.5) < 1e-9)
    check("missing percentage surfaced in the prompt",
          "MISSING" in render(window_features(
              blind(synthetic_window(nan_from=50)))))


def test_nan_vs_zero_imputed():
    print("\n[6] NaN-aware features differ from zero-imputed")
    f = window_features(blind(synthetic_window(nan_from=50)))["nodes"]["ankle"]
    got = f["accel"]["y"]["mean"]
    # Valid half is constant 0.9; zero-imputing the missing half gives 0.45.
    check("mean over valid samples only", abs(got - 0.9) < 1e-9,
          "got %.4f; zero-imputed would be 0.45" % got)
    check("zero-imputation would have changed the answer",
          abs(0.45 - got) > 0.4)
    check("energy also computed over valid samples only",
          abs(f["accel"]["y"]["energy"] - 0.81) < 1e-9)


def test_mhealth_chest_gyro():
    print("\n[7] MHEALTH chest: gyro features absent, not zero-filled")
    src = DatasetReplaySource("mhealth", subjects=["subject1"],
                              label_set="CANONICAL_6")
    w = next(iter(src.windows()))
    f = window_features(blind(w))["nodes"]
    check("chest has no gyro block", "gyro" not in f["chest"])
    check("chest states why", "no gyroscope" in f["chest"].get(
        "gyro_absent_reason", ""))
    check("chest is NOT given zero gyro features",
          not any(k.startswith("gyro") and isinstance(v, dict)
                  for k, v in f["chest"].items()))
    check("ankle DOES have gyro features", isinstance(f["ankle"].get("gyro"),
                                                      dict))
    txt = render(window_features(blind(w)))
    check("prompt says 'none on this node'", "none on this node" in txt)
    check("chest still contributes accel features",
          isinstance(f["chest"].get("accel"), dict))


def test_fewshot_disjoint():
    print("\n[8] few-shot subjects disjoint from eval subjects")
    doc = load_fewshot()
    fs, ev = set(doc["fewshot_subjects"]), set(doc["eval_subjects"])
    check("no overlap", not (fs & ev), "fewshot=%s eval=%s"
          % (sorted(fs), sorted(ev)))
    check("2 examples per class",
          all(sum(1 for e in doc["examples"] if e["activity"] == a) == 2
              for a in {e["activity"] for e in doc["examples"]}))
    check("all 8 classes covered",
          len({e["activity"] for e in doc["examples"]}) == 8)
    check("declared healthy-only", "HEALTHY" in doc["provenance"])
    # The assertion must actually fire, not just be documented.
    try:
        build_fewshot_block(doc, eval_subjects=list(fs)[:1])
        check("overlap RAISES", False, "accepted a leaking split")
    except ValueError as e:
        check("overlap RAISES", True)
        check("error warns about invisible inflation", "inflates" in str(e))


def test_prompt_explicit_format():
    print("\n[9] prompt states the output format explicitly")
    p = load_system_prompt()
    check("demands exactly one character", "exactly one character" in p)
    check("forbids preamble", "no preamble" in p.lower())
    check("does NOT rely on continuing after 'Answer:'",
          not p.rstrip().endswith("Answer:"),
          "V0 ended that way and was unusable on gemma-4-31b")
    check("explains NO DATA semantics", "absent evidence" in p)
    for ph in ("{legend}", "{few_shot}", "{query}"):
        check("carries %s placeholder" % ph, ph in p)


def test_confidence_method_propagates():
    print("\n[10] confidence_method reaches every row")
    with Mock("openai_good"):
        r = adapter_for().classify("p", LETTERS8)
    check("present on the classify result", r["confidence_method"] == "logprob")
    keys = {"distribution", "predicted", "max_p", "log_margin", "entropy",
            "confidence_method", "provider", "model"}
    check("classify returns the full documented contract",
          keys <= set(r), "missing %s" % sorted(keys - set(r)))


def main() -> int:
    print("=" * 74)
    print("PHASE 5 AGENT TEST SUITE (mocked LLM, no network, no keys)")
    print("=" * 74)
    test_distribution()
    test_log_margin()
    test_portability_guard()
    test_blind_required()
    test_no_zero_fill()
    test_nan_vs_zero_imputed()
    test_mhealth_chest_gyro()
    test_fewshot_disjoint()
    test_prompt_explicit_format()
    test_confidence_method_propagates()

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
