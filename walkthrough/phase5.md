# Phase 5 — Baseline Agent (S1) + Confidence

**Status:** S1 built and running end-to-end. **The headroom probe returned a
blocking finding for Phase 6 — read §2 before planning the sweep.**

**Date:** 2026-08-30

---

## 1. What was built

```
core/results_paths.py         provider-dimensioned result paths
features/extractors.py        BlindWindow -> features (never zero-fills)
agent/llm_client_adapter.py   thin wrapper over Phase 1's verified path
agent/graph.py                S1: ingest -> context_build -> har_agent -> confidence
agent/prompts/system_v1.txt   explicit output format (portable across models)
agent/prompts/fewshot.yaml    2/class, healthy windows, disjoint subjects
scripts/measure_confidence_headroom.py
scripts/run_phase5_samples.py
frontend/phase5_playground.html
tests/test_agent.py           51 assertions
results/paper/by_provider/<provider>_<model>/
results/paper/cross_provider/compare.py
```

S1 is deliberately minimal: **exactly one LLM call, no mitigation**. It *shows*
the agent each node's health state but does not act on it. Separating
perception from action is what lets Phase 7 add S2/S3 as feature-flagged edges
(`FLAGS` in `agent/graph.py`) rather than a rewrite. `trust_weight` is carried
through every result row and left deliberately unused.

---

## 2. HEADROOM VERDICT — **NO HEADROOM** on `gemma-4-31b`

20 paired windows, clean vs a fully dead ankle (dropout sev4):

| Signal | Clean | Injected | Paired Δ | Cohen's d | Consistency | Moves? |
|---|---|---|---|---|---|---|
| `max_p` | 0.95596 | 0.98007 | **+0.024** | +0.21 | 60 % | no |
| `log_margin` | 10.147 | 8.716 | **−1.431** | −0.28 | 60 % | no |
| `entropy` | 0.1029 | 0.0807 | −0.022 | −0.09 | 60 % | no |

Criterion: |Cohen's d| ≥ 0.5 **and** ≥ 70 % of pairs shifting the same way.

**Nothing clears it.** `log_margin` has the largest effect and moves in the
expected direction, but only 12 of 20 pairs move that way — barely above
chance. `max_p` moves the **wrong** way: the model becomes *slightly more*
confident when a node dies.

> **Consequence for Phase 6.** This provider cannot express degraded confidence
> on this task. Running the sweep on it would produce a flat chart whose
> "finding" is an artefact of saturation, and the Overconfidence Gap would
> collapse to (1 − accuracy) by construction. **Phase 6's headline must come
> from another provider** — gpt-4o once credits are restored, or
> self-consistency via `anthropic`.

The analysis is **paired**: each window is measured both ways, so the
per-window delta is the unit. An unpaired comparison of group means (the first
pass) is far less sensitive, because between-window variation across activities
dwarfs the within-window effect of injection.

---

## 3. A renderer bug that mattered more than the probe

The first headroom run reported `max_p` 0.9993 → 0.9997 and looked hopelessly
saturated. It was **partly our own fault**.

The Gate 2 V1 renderer, applied to a fully-NaN node, emitted:

```
ankle  ORIENTATION (+0.00, +0.00, +0.00)  magnitude nan g
       MOTION      nan g  (vigorous)
```

A zero orientation vector — indistinguishable from a real reading of zero — and
a **"vigorous" motion label for a dead node**, because `nan < 0.08` is False and
`nan < 0.35` is False, so the band check fell through to the last branch. That
is fabricated evidence handed straight to the model, and it is exactly the
class of error Phases 2–4 were built to prevent.

`features/extractors.py` now reports:

```
ankle  NO DATA -- node reported no usable data (100% of samples missing)
```

With the fabrication removed, clean `max_p` fell **0.9993 → 0.956**. A
meaningful part of the apparent saturation was our prompt, not the model. The
verdict above was reached only *after* this fix.

---

## 4. Other bugs found by running it

**4.1 Every sampled window was the same activity.** `run_phase5_samples` took
the first N windows; windows arrive in activity-segment order, so all 8 were
`lying` and accuracy read **0.00**. Sampling is now stratified across classes,
as Gate 2's was. This is the second time in the project that "take the first N"
has produced a misleading result.

**4.2 Few-shot collapses this model's distribution.** Diagnosed while chasing
4.1 and worth recording:

| Prompt | Top tokens | Distinct labels in top-20 |
|---|---|---|
| no few-shot | C −0.06, B −2.89, A −5.57 | **8** |
| 1 example/class | E −0.00, ' E' −13.97 | 5 |
| 2 examples/class | E −0.00, ' E' −13.99 | **1** |

With few-shot present the distribution collapses onto a single token at
p ≈ 1.0; without it the model produces a genuinely graded distribution. Length
is not the cause — a compacted 2/class block collapses identically. **This is a
strong candidate explanation for the saturation** and should be investigated
before Phase 6, since it is a property of our prompt rather than of the model's
knowledge.

---

## 5. Determinism (methods-section number)

| Provider | Protocol | mean abs Δ max_p | Bit-identical |
|---|---|---|---|
| `cerebras/gemma-4-31b` | 16 windows × 2 | **0.000000** | yes |
| `openai/gpt-4o` | 5 windows × 2 | **0.046313** | **no** |

Cerebras is exactly reproducible; gpt-4o is not, even at temperature 0. Neither
provider is simultaneously **reproducible and expressive** — that constraint
belongs in the methods section.

---

## 6. S1 end-to-end results (n = 16, `gemma-4-31b`)

| Condition | n | Accuracy | mean `max_p` | mean `log_margin` |
|---|---|---|---|---|
| healthy | 8 | 0.38 | 0.9998 | 12.653 |
| injected | 8 | 0.50 | 0.9997 | 11.115 |

Qualitative check (what the DoD asks for): Δ mean `max_p` = **−0.00014**
(nothing); Δ mean `log_margin` = **−1.539**, consistent in sign and magnitude
with the headroom probe's −1.43.

> **Accuracy caveat, stated rather than buried.** 0.38 is well below Gate 2's
> V1 result of 0.7083 on the same model. Three differences could account for
> it: n = 8 per condition is tiny; the S1 prompt adds per-node health
> annotations; and the richer extractor changed the feature rendering. **This
> is not yet explained and should be resolved before Phase 6** — an S1 baseline
> weaker than the Gate 2 pipeline it descends from would understate every
> downstream comparison. The obvious first ablation is health-annotations
> on/off.

---

## 7. Results are dimensioned by provider

```
results/paper/
  by_provider/cerebras_gemma-4-31b/   gate1, gate2, ablation, headroom,
                                      determinism, phase5_samples
  by_provider/openai_gpt-4o/          gate1, gate2(V0), determinism,
                                      not_run.json
  cross_provider/compare.py           reads whatever exists
  cross_provider/comparison.md
```

Scripts resolve their folder from the **resolved backend**, so switching
`LLM_PROVIDER` writes elsewhere automatically and two providers can never
overwrite each other. `openai_gpt-4o/not_run.json` records which cells are
**unrun (no credits)** — unrun is not the same as failed, and the comparison
tables keep them distinct.

---

## 8. Verification

| Check | Result |
|---|---|
| `tests/test_agent.py` | **51/51** |
| Phase 1 `test_backends.py` | 59/59 — unweakened |
| Phase 2 `test_datasource.py` | 45/45 — unweakened |
| Phase 4 `test_health.py` | 70/70 — unweakened |
| Playground (real data / empty) | 15/15, 6/6 |
| Ground-truth leakage into `agent/`, `features/` | none (grep) |

The extractor accepts a `BlindWindow` only and raises on a raw `Window`; the
graph blinds once at `ingest`. MHEALTH's chest emits **no** gyro features
rather than zeros, and statistics are computed over non-NaN samples only.

---

## 9. How to approach Phase 6

**Do not start the sweep on `gemma-4-31b`.** §2 is the blocker. In order:

1. **Investigate the few-shot collapse (§4.2).** It is the cheapest and most
   likely explanation for the missing headroom, and it is our prompt's
   behaviour rather than the model's. Compare zero-shot vs few-shot headroom
   directly.
2. **Explain the S1 accuracy gap (§6).** Ablate health annotations on/off
   against the Gate 2 V1 configuration.
3. **Restore OpenAI credits** and re-run `measure_confidence_headroom.py` there
   — the scripts already write to `by_provider/openai_gpt-4o/`. gpt-4o showed a
   genuinely graded distribution at Gate 1 (0.9416 / 0.0583), so it is the
   likelier source of a measurable Overconfidence Gap.
4. Only then run the sweep, and report which provider produced the headline.

Carried forward for Phase 7: `FLAGS` in `agent/graph.py` (`trust_reweighting`,
`abstention`, `requery`) are the declared extension points, and every result
row already carries `trust_weights` and `confidence_method`.
