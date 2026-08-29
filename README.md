# Autonomic Agentic BSN

Research question: **do LLM agents doing human activity recognition from
body-worn sensors become "confidently wrong" when the sensor network
degrades?**

The project is built in phases. Each phase ends with a walkthrough document in
[`walkthrough/`](walkthrough/) and its own page on a cumulative dashboard.

---

## Quick start

```bash
# 1. install
python -m venv .venv
.venv\Scripts\activate                 # PowerShell;  source .venv/bin/activate on Unix
pip install -r requirements.txt

# 2. set up Vertex AI auth (see "Model backends" below)
copy .env.example .env                 # PowerShell;  cp .env.example .env on Unix
#    then edit .env: GOOGLE_CLOUD_PROJECT, and either
#    GOOGLE_APPLICATION_CREDENTIALS (service-account JSON) or `gcloud auth
#    application-default login`

# 3. get the data  (see data/raw/README.md for the download commands)
#    expected: data/raw/pamap2/Protocol/subject101.dat ... subject109.dat

# 4. run the checks, in order
python scripts/profile_dataset.py           # dataset profile (no API key needed)
python scripts/check_logprobs.py            # gate 1 - must PASS first
python scripts/check_baseline_accuracy.py   # gate 2 - must reach >= 0.65

# 5. view the dashboard
cd frontend
python -m http.server 8000
#    then open http://localhost:8000/
```

Every script exits **0 on PASS, 1 on FAIL**.

---

## Phase 1 — foundational de-risking

Two yes/no questions, answered before the wider system is built. If either
fails we stop and fix it rather than writing more code.

| # | Question | Script | Gate |
|---|---|---|---|
| 1 | Can we extract **real token log-probabilities**? | `scripts/check_logprobs.py` | non-uniform distribution summing to 1.0 |
| 2 | Is zero-shot HAR **accurate enough on clean data**? | `scripts/check_baseline_accuracy.py` | overall accuracy ≥ **0.65** over 8 classes |

Question 1 gates everything: without a genuine confidence signal, "confidently
wrong" is not measurable. Question 2 gates dynamic range: if the model is
already near chance on pristine sensors, later degradation results are
unattributable.

> **Phase 1 is deliberately disposable.** The PAMAP2 parsing here is throwaway;
> only the *verified column indices* carry forward. Phase 2 rebuilds the real
> parser behind `DataSource`. Failure injection, the health monitor and
> LangGraph belong to later phases — do not add them here.

Full detail, including everything verified and everything still open:
**[`walkthrough/phase1.md`](walkthrough/phase1.md)**.

---

## Phase 2 — pluggable data layer & dataset explorer

Implements the `DataSource` contract every later phase plugs into — including
Phase 9's ESP32/BLE hardware source, which must satisfy it unchanged. Backed by
real PAMAP2 and MHEALTH data. **Makes zero LLM calls**, so it is not blocked by
Phase 1's unpassed gates.

```bash
python scripts/verify_labels.py           # label-map cross-check, both datasets
python scripts/profile_datasets.py        # -> results/phase2/dataset_stats.json
python scripts/export_phase2_samples.py   # -> frontend sample waveforms
python tests/test_datasource.py           # 47 assertions against real data
```

| | PAMAP2 | MHEALTH |
|---|---|---|
| Subjects / rate | 9 @ 100 Hz | 10 @ 50 Hz |
| Label set | PAMAP2_8 | CANONICAL_6 |
| Rows / duration | 2,872,533 / 8.0 h | 1,215,745 / 6.8 h |
| Windows (2.56 s, 50 %) | **9,909** | **2,760** |
| Timestamp | measured | **derived** from row index |
| chest channels | accel + gyro + mag | **accel + ECG only** |

The rebuilt parser reproduces Phase 1's verified 9,909 windows **exactly**
(delta +0, 0.000 %).

> **Absent sensor is not a zero reading.** `gyro_dps` is `None` when a node has
> no gyroscope (MHEALTH's chest) and is *never* zero-filled. A node that has a
> gyro but dropped a sample carries `NaN` — a third, distinct state. Collapsing
> these would fabricate a "sensor reads zero" signal inside the degradation
> study this project exists to run.

Full detail: **[`walkthrough/phase2.md`](walkthrough/phase2.md)**.

---

## Phase 3 — failure injection lab

Five failure modes, each a `DataSource` **decorator**: it wraps any source and
is itself a valid source, so injectors compose with the dataset replay, with
each other, and with Phase 9's hardware unchanged. **Zero LLM calls.**

```bash
python scripts/verify_injectors.py         # -> results/phase3/injector_verification.json
python scripts/export_phase3_samples.py    # -> 100 paired clean/injected traces
python tests/test_injection.py             # 71 assertions
```

| | Failure | Severity 1 → 4 |
|---|---|---|
| F1 | `dropout` | blanks last 25 / 50 / 75 / 100 % of the window |
| F2 | `clock_desync` | 50 / 200 / 500 / 2000 ms offset |
| F3 | `packet_loss` | Gilbert-Elliott (L, B) = (.05,3) (.15,4) (.30,6) (.50,8) |
| F4 | `rate_degradation` | native ÷ 2 / 4 / 8 / 16, zero-order hold |
| F5 | `displacement` | 15 / 30 / 45 / 90° sensor-frame rotation |

> **Three states, kept distinct.** Valid sample → `(x,y,z)`; dropped sample →
> tuple of **NaN**; no such sensor → **`None`**. No injector may ever write
> `None`, and blanked samples are never zero — zero is a real stationary
> reading a model may believe. `rate_degradation` emits **zero NaNs** (stale
> but present) while `packet_loss` emits gaps (absent); Phase 4 must tell them
> apart.

Two calibration findings are documented in the walkthrough: displacement's
realised angle is **posture-dependent** (a requested 15° realises as 2.3–14.9°
depending on gravity/axis alignment), and the 10,000-step Gilbert-Elliott check
is **statistically underpowered** at severity 4.

Full detail: **[`walkthrough/phase3.md`](walkthrough/phase3.md)**.

---

## Phase 4 — health monitor

Detects and diagnoses node degradation from observable signal statistics
**alone**. **Zero LLM calls.**

```bash
python scripts/run_detection_eval.py       # -> results/phase4/detection_metrics.json
python scripts/export_phase4_samples.py    # -> per-sample monitor verdicts
python tests/test_health.py                # 70 assertions
```

| Failure | F1 | | Failure | F1 |
|---|---|---|---|---|
| `dropout` | **0.960** | | `packet_loss` | **0.805** |
| `rate_degradation` | **0.960** | | `displacement` | 0.595 |
| `clock_desync` | **0.960** | | **FPR (clean)** | **0.067** |

Diagnosis accuracy across the 6 classes: **0.880**.

> **The blind is mechanical, not a convention.** The monitor receives a
> `BlindWindow`: `injected_failure`, the true activity label and every
> injection field are stripped and **raise** on access — returning `None`
> would let buggy code read nothing and still look correct. A grep test
> confirms the ground-truth identifiers appear nowhere in `signals.py` or
> `diagnose.py`. Only `score_detection.py` reads truth, after the fact.

**Displacement is weakly detected, and that is a finding.** A rotation about a
node's long axis is only partly observable in gravity
(`cos(obs) = cos²φ + sin²φ·cos θ`), and natural posture swing exceeds the
rotation's own effect — so it is *not reliably detectable from gravity alone*.
Detecting it needs a temporal baseline or a second modality.

Full detail: **[`walkthrough/phase4.md`](walkthrough/phase4.md)**.

---

## Model backends — read this before changing the model

The only thing this project needs from a model is a **real next-token
log-probability distribution**. Several popular endpoints return `200 OK` with
no usable confidence data, which fails *invisibly*. Two such traps are guarded
in code:

### ⚠️ The Gemini Developer API does not expose logprobs

**Verified against the live API, not just the docs.** All 12 probed models --
`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-flash-latest`,
`gemini-3-flash-preview`, `gemini-3.5-flash`, `gemini-3.7-flash`,
`gemma-4-31b-it` and others -- return:

```
400  "Logprobs is not enabled for this model"
```

while returning `200 OK` for ordinary generation. The failure persists without
`thinkingConfig`, so logprobs itself is the blocker.

This gates the **Developer API surface** (`generativelanguage.googleapis.com`,
AI Studio keys). Google's logprobs guide targets **Vertex AI**
(`aiplatform.googleapis.com`) -- a different surface needing a GCP project,
billing and service-account auth.

**A model without logprobs cannot be used for this project**, however good its
text output is. The default backend is therefore `vertex`, not `gemini`.

### Vertex AI setup (the default backend)

Vertex AI is the Google Cloud surface where Gemini logprobs are available. It
needs a GCP project rather than an AI Studio key.

1. Create or choose a GCP project and **enable billing**.
2. Enable the [Vertex AI API](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com).
3. Authenticate with **either**:
   - **Service account** (no CLI needed) — IAM &amp; Admin → Service Accounts →
     create one, grant it **Vertex AI User**, create a JSON key, save it
     *outside* the repo, and set `GOOGLE_APPLICATION_CREDENTIALS` in `.env`.
   - **gcloud CLI** — `gcloud auth application-default login`.
4. Put your project id in `.env` as `GOOGLE_CLOUD_PROJECT`.

Then run `python scripts/check_logprobs.py`. It exists precisely to prove
whether logprobs really work on your account — believe its output over any
documentation, including this README.

### ⚠️ Never use Ollama's OpenAI-compat endpoint

`/v1/chat/completions` (and `/v1/completions`) on port 11434 **silently drops
`logprobs`** even when requested — unsupported per Ollama's own
OpenAI-compatibility docs, tracked by
[ollama/ollama#16117](https://github.com/ollama/ollama/issues/16117).
`scripts/_llm_client.py` hard-blocks any endpoint matching `:11434` + `/v1/`.
Ollama's *native* `/api/generate` is supported but is **not** in the default
fallback chain.

### Configured backends

Set `backend:` in [`configs/models.yaml`](configs/models.yaml); use `auto` to
walk `fallback_order` until one returns real logprobs.

| Backend | Endpoint | Notes |
|---|---|---|
| `vertex` *(default)* | `aiplatform.googleapis.com` | GCP project + OAuth2; `logprobs` capped at 20 |
| `llamacpp` | `POST /completion` | local, CPU-friendly |
| `vllm` | `POST /v1/completions` | local; genuinely supports logprobs |
| `gemini` | `generativelanguage.googleapis.com` | **verified non-working** — rejects logprobs |
| `ollama_native` | `POST /api/generate` | reference only, not in the chain |

Whichever backend passes is recorded as `backend` in
`results/phase1/logprob_check.json`. **Every later phase must use that same
one** — confidence numbers are not comparable across serving stacks.

#### Running a local model instead

```bash
llama-server -m <path-to-gguf> -c 4096 --port 8080
```
Then set `backend: llamacpp`. `-c 4096` fits Step 2's 16-example prompt
(~2.1k tokens), but raising `few_shot_per_class` needs `-c 8192` or the prompt
is silently truncated from the left, which quietly destroys accuracy.

---

## Layout

```
configs/models.yaml     backend selection + all tunable parameters
core/
  datasource.py         THE DataSource contract (NodeFrame, Window, Protocol)
  labels.py             label sets, ID maps, deliberate exclusions
datasets/
  pamap2_loader.py      full 54-column map, rad/s -> deg/s
  mhealth_loader.py     24-column map, derived clock, absent-gyro handling
  dataset_replay_source.py   reference DataSource implementation
health/
  window_view.py        BlindWindow -- the mechanical blind, built first
  signals.py            observable signals (no ground truth, grep-verified)
  diagnose.py           auditable if/elif ladder + evidence
  score_detection.py    the ONLY module that reads ground truth
injection/
  base.py               FailureInjector decorator + strategy ABC
  dropout.py clock_desync.py packet_loss.py
  rate_degradation.py displacement.py
  registry.py           type -> class map; Phase 6 iterates this
scripts/
  _llm_client.py        THE shared logprob extraction path (all backends)
  check_logprobs.py     gate 1
  check_baseline_accuracy.py  gate 2
  profile_dataset.py    Phase 1 dataset profile (no LLM needed)
  verify_labels.py      Phase 2 label-map cross-check
  profile_datasets.py   Phase 2 Table 1
  export_phase2_samples.py   frontend sample waveforms
frontend/               dashboard (no frameworks, no build step)
walkthrough/            one handoff document per phase
data/raw/               datasets (gitignored; see data/raw/README.md)
results/                generated results (gitignored)
```

`_llm_client.py` is the load-bearing file: both check scripts call
`score_labels()` from it, so the accuracy run provably uses the same extraction
path that the logprob check validates.

---

## The dashboard

```bash
cd frontend && python -m http.server 8000
```

Open <http://localhost:8000/>. The page reads the generated JSON live, so
**re-run a script and refresh the browser** — there is no build step and
nothing to restart.

Phase 1 shows the dataset (stat tiles, windows per class, a per-subject ×
activity grid, raw accelerometer sparklines), both PASS/FAIL gates, the parsed
probability distribution, per-class accuracy against the 65 % threshold, the
8×8 confusion matrix, and the exact prompt template for the reproducibility
appendix.

Every panel is independent: if a result file is missing, that panel shows a
"run the script first" notice while the others still render.

> **Why results are mirrored.** `http.server` refuses to serve paths above its
> root, so a page served from `frontend/` cannot reach `../results/`. Scripts
> write results twice — canonical in `results/`, mirrored in
> `frontend/results/` — and the page prefers the mirror, falling back to
> `../results/` if you serve from the project root instead. Both are
> gitignored.

The dashboard is cumulative: `frontend/index.html` holds a single `PHASES`
array, so shipping a phase means adding one entry. Phases 2–10 render greyed
out with a "not yet built" tag; their names are placeholders.

---

## What each Phase 1 check asserts

### Step 1 — `check_logprobs.py`

1. Send a **dummy** feature summary with 6 labels A–F, `n_predict: 1`,
   `n_probs: 6`, `post_sampling_probs: false`, `temperature: 0`.
2. Renormalise via softmax over the logprobs:
   `p_i = exp(lp_i) / Σ_j exp(lp_j)`.
3. Assert the result **sums to 1.0 within 1e-6** and is **not uniform**
   (`max_prob` must beat 1/6 by ≥ `uniform_margin`, default 0.05).
   *A uniform distribution means we are reading noise, not real confidence* —
   the single most important assertion in the phase.
4. Write `results/phase1/logprob_check.json`.

Mass from tokenizer variants that normalise to the same letter (`"A"`, `" A"`,
sentencepiece `"▁A"`) is summed in log space, so a model that emits a leading
space is not mistaken for one that never answered.

### Step 2 — `check_baseline_accuracy.py`

1. Read `subject10{1..9}.dat` using only the verified columns.
2. **Verify the label map first.** Missing expected ID → prominent warning;
   *undocumented* ID → prominent warning; documented-but-excluded PAMAP2 ID
   (7/16/17/24) → INFO, since those legitimately appear in the Protocol files
   and warning on them every run would just train you to ignore warnings.
3. Filter to the 8 target IDs, drop `activityID == 0`, drop NaN accel rows.
4. Window into **2.56 s windows, 50 % overlap**, cut only inside contiguous
   single-activity segments so no window straddles an activity change or gap.
5. Per window: 3 nodes × 3 axes × 5 stats (mean, std, min, max, energy) = **45
   features**.
6. Prompt at temperature 0 with **2 few-shot examples per class drawn from
   subjects disjoint from the test set** — enforced; the script aborts on
   overlap.
7. Predict by **argmax over the 8 label-token logprobs**, never free-text
   parsing.
8. Report overall accuracy, per-class accuracy and an 8×8 confusion matrix.
9. Write `results/phase1/baseline_accuracy.json` including the **exact prompt
   template and a fully rendered example prompt**.
10. Print PASS (≥ 0.65) or FAIL. **On FAIL** it prints the 3 most-confused
    class pairs and ranks the candidate fixes from what the confusion matrix
    actually shows.

`step2_n_probs` defaults to **20**, not 8: with only 8 candidate slots,
probability mass on `"\n"` or `<eos>` can push a real label token out of the
top-k and silently zero it.

---

## Dataset

PAMAP2, 9 subjects, 3 IMUs (wrist/chest/ankle) at 100 Hz.
Download commands and the full 54-column format:
[`data/raw/README.md`](data/raw/README.md).

Current profile: **2,872,533 rows · 8.0 hours · 9,909 windows**, all 8 target
activities present, no undocumented activity IDs. Subject109 contains only
8,477 rows across 2 activities and yields 0 windows, so it is excluded from
both the test and few-shot sets.

Verified column map (**the only part of Phase 1's parsing that carries
forward**):

| Node | IMU block start | accel16 x,y,z |
|---|---|---|
| wrist (hand) | 3 | **4, 5, 6** |
| chest | 20 | **21, 22, 23** |
| ankle | 37 | **38, 39, 40** |

---

## Contributing notes

- Commit **incrementally within a phase** — one focused commit per change, not
  a single bulky commit per phase.
- Write the phase's `walkthrough/phaseN.md` before moving on, and keep it
  honest about what was *not* proven.
- Update `.env.example` whenever a new environment variable is needed.
- Never commit `.env`, datasets, results or model weights — all are gitignored.
