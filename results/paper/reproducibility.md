# Reproducibility appendix

## Confidence extraction

Backend `openai`, model `gpt-4o`, `confidence_method=logprob`.

Next-token log-probabilities are renormalised over the label tokens only: `p_i = exp(lp_i) / sum_j exp(lp_j)`. Probability mass from tokenizer variants that normalise to the same letter (`"A"`, `" A"`, sentencepiece `"_A"`) is summed in log space before renormalising.

### Gate 1 probe prompt

```
You are classifying human activity from body-worn sensor features.

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
Answer:
```

## Gate 2 protocol

- Windows: 2.56 s, 50 % overlap, cut only inside contiguous single-activity segments, >=60 % sample coverage
- Features: 3 nodes x 3 axes x 5 statistics = 45 per window
- Test subjects: [101, 105, 106]
- Few-shot subjects: [102, 103, 104, 107, 108] (disjoint, enforced at runtime)
- Few-shot: 2 examples per class
- Prediction: argmax over the label-token logprobs, never free-text parsing

### Exact prompt template (V0, as shipped)

```
You are an expert at recognising human activity from body-worn inertial sensors.

Each window is 2.56 seconds of tri-axial accelerometer data (units m/s^2, gravity included) from three body-worn nodes: wrist, chest and ankle. For each node and axis you are given mean, standard deviation, minimum, maximum and energy (mean of squares).

Legend:
{legend}

{few_shot}Now classify this window.

{query}

Answer with exactly one letter from the legend.
Answer:
```

## Health monitor

The monitor receives a `BlindWindow`: `injected_failure`, the true activity label and all injection metadata are stripped and **raise on access**. A grep test asserts the ground-truth identifiers appear nowhere in `health/signals.py` or `health/diagnose.py`. Ground truth is read only by `health/score_detection.py`, after the fact.

- Seed: 20260830
- Windows per condition: 30
- Scoring uses `meta['realised']`, never `meta['requested']`

- mhealth: calibration subject `subject2` (held out), evaluation subject `subject1`, target node `ankle`
- pamap2: calibration subject `subject102` (held out), evaluation subject `subject101`, target node `ankle`

## Failure injection

Five failure types as `DataSource` decorators. Each stochastic injector uses its own seeded `numpy.random.Generator`, never global state, so results do not depend on call order.

Missing samples are always NaN tuples, never zeros (a zero reading is a real stationary measurement) and never `None` (`None` is reserved for a structurally absent sensor, such as MHEALTH's chest gyroscope).

## Software

- Python 3.11, numpy, pandas, requests, PyYAML
- No agent framework, no learned model anywhere in Phases 1-4
- Every phase script is deterministic given its seed
- Test suites: `tests/test_backends.py`, `test_datasource.py`, `test_injection.py`, `test_health.py`
