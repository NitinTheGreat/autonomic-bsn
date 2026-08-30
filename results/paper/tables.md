# Autonomic Agentic BSN — paper tables

_Generated 2026-08-30 by `scripts/build_paper_artifacts.py`. Every number is read from a `results/` JSON written by a phase script; nothing here is recomputed or estimated._

## Table 1 — Datasets

| Dataset | Subjects | Rate | Rows | Hours | Windows | Classes | Per-node channels |
|---|---|---|---|---|---|---|---|
| MHEALTH | 10 | 50 Hz | 1,215,745 | 6.8 | 2,760 | 6 | chest: accel+ecg; ankle: accel+gyro+mag; wrist: accel+gyro+mag |
| PAMAP2 | 9 | 100 Hz | 2,872,533 | 8.0 | 9,909 | 8 | wrist: accel+gyro+mag; chest: accel+gyro+mag; ankle: accel+gyro+mag |

## Table 2 — Providers verified to expose token log-probabilities

| Provider | Model | Real logprobs | max p | 2nd p | Confidence headroom | Role |
|---|---|---|---|---|---|---|
| openai | `gpt-4o` | yes | 0.94160 | 0.05834 | usable | paid -- paper results |
| cerebras | `gemma-4-31b` | yes | 0.99999 | 0.00001 | **saturated** | free -- development, demos |

> Both providers were verified empirically before use. Two other surfaces were tested and **rejected**: Ollama's OpenAI-compat layer silently drops `logprobs`, and the Gemini Developer API returns `400 "Logprobs is not enabled for this model"` on every model while serving ordinary generation normally.

## Table 2b — Temperature-0 determinism

| Provider | Repeats | mean max p | sd | spread | Bit-identical |
|---|---|---|---|---|---|
| openai | 5 | 0.95087 | 0.01853 | 0.04631 | **no** |
| cerebras | 5 | 0.99999 | 0.00000 | 0.00000 | yes |

> The **same** prompt at temperature 0. `gpt-4o` is not bit-reproducible: repeated identical probes returned differing confidence values. A study whose dependent variable IS the confidence number must therefore report repeated measurements with variance, not a single run. `gemma-4-31b` on Cerebras was bit-identical across repeats but is saturated (Table 2), so neither provider is simultaneously reproducible and expressive -- a constraint worth stating explicitly in the methods.

## Table 3 — Gate 2 baseline accuracy: feature and format ablation

| Provider | Model | Variant | Classes | n | Accuracy | Gate 2 |
|---|---|---|---|---|---|---|
| openai | gpt-4o | V0 (shipped) | 8 | 144 | 0.4722 | FAIL |
| openai | gpt-4o | V0 (shipped) | 6 | 96 | 0.5833 | FAIL |
| openai | `openai` | V0 | 8 | -- | not run (no credits) | -- |
| openai | `openai` | V0F | 8 | -- | not run (no credits) | -- |
| openai | `openai` | V1 | 8 | -- | not run (no credits) | -- |
| cerebras | `cerebras` | V0 | 6 | -- | unusable (no bare-letter answer) | -- |
| cerebras | `cerebras` | V0 | 8 | -- | unusable (no bare-letter answer) | -- |
| cerebras | `gemma-4-31b` | V0F | 6 | 48 | 0.8125 | PASS |
| cerebras | `gemma-4-31b` | V0F | 8 | 48 | 0.6667 | PASS |
| cerebras | `gemma-4-31b` | V1 | 8 | 48 | 0.7083 | PASS |

> V0 is the shipped rendering (45 raw numbers in m/s², orientation implicit). V0F changes only the answer-format instruction. V1 re-presents the **same 45 features** with an explicit per-node orientation unit vector and motion level, in g. No new sensor channels are introduced in any variant.

## Table 4 — Health monitor detection

| Failure | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| clock_desync | 0.923 | 1.000 | **0.960** | 240 | 20 | 0 |
| displacement | 0.846 | 0.458 | **0.595** | 110 | 20 | 130 |
| dropout | 0.923 | 1.000 | **0.960** | 240 | 20 | 0 |
| packet_loss | 0.897 | 0.729 | **0.805** | 175 | 20 | 65 |
| rate_degradation | 0.923 | 1.000 | **0.960** | 240 | 20 | 0 |

False-positive rate on clean windows: **0.0667** (12 of 180 node-windows). Diagnosis accuracy across the 6 classes: **0.8797**.

## Table 5 — Diagnosis confusion matrix

| true \ predicted | healthy | dropout | packet_loss | rate_degradation | clock_desync | displacement |
|---|---|---|---|---|---|---|
| **healthy** | 168 |  |  |  |  | 12 |
| **dropout** |  | 240 |  |  |  |  |
| **packet_loss** | 17 | 6 | 216 |  |  | 1 |
| **rate_degradation** |  |  |  | 240 |  |  |
| **clock_desync** |  |  |  |  | 240 |  |
| **displacement** | 130 |  |  |  |  | 110 |

## Table 6 — Displacement detectability

| Node / activity | n | Recall | Diagnosis acc. | Mean observed (deg) | Mean alignment | Verdict |
|---|---|---|---|---|---|---|
| ankle/lying | 120 | 0.42 | 0.42 | 34.9 | 0.315 | **undetectable** |
| ankle/standing | 120 | 0.50 | 0.50 | 43.5 | 0.243 | detectable |

Temporal-baseline probe: best onset recall (K=1) **1.000** vs the population baseline's 0.458 pooled, at worst clean FPR **0.1974** vs 0.0667. The temporal reference detects the transient at onset then absorbs it, so it relocates the limit rather than removing it.

## Table 7 — Gilbert–Elliott packet-loss calibration

| Severity | target L | realised L | |dL| | target B | realised B | %err B | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 0.05 | 0.0491 | 0.0009 | 3.0 | 2.973 | 0.91% | within |
| 2 | 0.15 | 0.1475 | 0.0025 | 4.0 | 3.990 | 0.24% | within |
| 3 | 0.30 | 0.3029 | 0.0029 | 6.0 | 6.043 | 0.72% | within |
| 4 | 0.50 | 0.5009 | 0.0009 | 8.0 | 8.052 | 0.65% | within |
