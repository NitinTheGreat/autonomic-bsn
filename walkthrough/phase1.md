# Phase 1 — Foundational de-risking

**Status:** BOTH GATES NOW RUN AGAINST LIVE MODELS (2026-08-30).
**Gate 1 PASSES. Gate 2 FAILS at 0.4722 against a 0.65 threshold.**
The STOP condition is in force — see §6a.

**Date:** 2026-08-29

---

## 1. Goal

The project studies whether LLM agents doing human activity recognition (HAR)
from body-worn sensors become **confidently wrong** when the sensor network
degrades. Two things must be true before any of that is worth building:

| # | Question | Gate |
|---|---|---|
| 1 | Can we extract **real token log-probabilities** from the model? | distribution sums to 1.0 (±1e-6) and is **not uniform** |
| 2 | Is zero-shot HAR **accurate enough on clean data**? | overall accuracy **≥ 0.65** over 8 classes |

Question 1 gates everything: without a genuine confidence signal, "confidently
wrong" cannot be measured at all. Question 2 gates dynamic range: if the model
is near chance on pristine sensors, later degradation results are
unattributable.

**Phase 1 is deliberately disposable.** The PAMAP2 parsing here is throwaway.
Only the *verified column indices* carry forward. Phase 2 rebuilds the real
parser behind `DataSource`.

---

## 2. What was built

```
configs/models.yaml          backend selection + all tunable parameters
scripts/_llm_client.py       THE shared logprob extraction path (all backends)
scripts/check_logprobs.py    Step 1 — logprob verification
scripts/check_baseline_accuracy.py  Step 2 — windowing, features, accuracy
scripts/profile_dataset.py   dataset profile for the dashboard (no LLM needed)
frontend/index.html          cumulative shell; one PHASES array drives the nav
frontend/phase1_derisk.html  Phase 1 dashboard
frontend/shared/{style.css,app.js}
walkthrough/                 these handoff documents
.env.example                 required environment variables
```

`_llm_client.py` is the important one. Both check scripts call `score_labels()`
from it, so Step 2's predictions provably use the same extraction path Step 1
validates. Any future phase that needs confidence must go through it too.

### Backends

Selected by `backend:` in `configs/models.yaml`; `auto` walks `fallback_order`.

| Backend | Endpoint | How logprobs are requested |
|---|---|---|
| `openai` *(default)* | `POST /v1/chat/completions` | `logprobs: true`, `top_logprobs: N` (API caps N at 20) |
| `anthropic` | `POST /v1/messages` | **no logprobs exist** — k-sample self-consistency instead |
| `vertex` | Vertex `generateContent` | **PAUSED** pending credentials |
| `gemini` | Developer `generateContent` | **PAUSED** — verified non-working, surface rejects logprobs |
| `llamacpp` | `POST /completion` | `n_probs`, `post_sampling_probs: false` |
| `vllm` | `POST /v1/completions` | `logprobs: N`, `echo: true` |
| `ollama_native` | `POST /api/generate` | kept for reference, **removed from the fallback chain** |

---

## 3. Two model traps found — both are silent-failure traps

These matter more than any code in this phase, because each would produce a
system that *looks* like it works while returning no usable confidence.

### 3.1 Ollama's OpenAI-compat layer

`/v1/chat/completions` on port 11434 **silently drops `logprobs`** even when
requested — listed as unsupported in Ollama's own OpenAI-compat docs, tracked
by [ollama/ollama#16117](https://github.com/ollama/ollama/issues/16117). It
answers `200 OK` with a normal-looking body and no confidence data.

`_llm_client.py` **hard-blocks** any endpoint matching `:11434` + `/v1/`.
Ollama's *native* `/api/generate` gained logprobs only recently, so the client
verifies the field is present and fails loudly rather than degrading.

### 3.2 The Gemini Developer API does not expose logprobs AT ALL

Originally flagged from documentation as a Gemini-3-only limitation. **Tested
against the live API on 2026-08-29 with a real key, this is far broader:**

Every one of 12 probed models -- including `gemini-2.5-flash`,
`gemini-2.5-pro`, `gemini-flash-latest`, `gemini-3-flash-preview`,
`gemini-3.5-flash`, `gemini-3.7-flash` and `gemma-4-31b-it` -- rejects the
request with:

```
400  "Logprobs is not enabled for this model"
```

The same models return `200 OK` for plain generation, and the failure persists
with `thinkingConfig` removed, so **logprobs specifically is the blocker**, not
the request shape or the thinking budget.

This is a gate on the Gemini **Developer API surface**
(`generativelanguage.googleapis.com`, AI Studio keys), not a per-model quirk.
Google's own logprobs walkthrough is published for **Vertex AI**
(`aiplatform.googleapis.com`), which is a different surface requiring a GCP
project, billing and service-account auth.

**Consequence: the Gemini backend as configured cannot satisfy Gate 1.** The
code is correct and fails loudly with the API's own message -- which is exactly
what this phase was built to catch -- but a different backend is required.

Since this project measures model confidence, a model without logprobs is
unusable here **regardless of how good its text output is**.

> **Decision taken (2026-08-29): Vertex AI.** A `vertex` backend was added and
> is now the default. It reuses the Gemini response parsing (same
> generateContent body) but authenticates with an OAuth2 bearer token from a
> service account or ADC, and carries project/location in the URL.
> **Whether this account is entitled to logprobs on Vertex is still unverified**
> — `check_logprobs.py` answers that, and its output is the authority.

The Gemini backend code is kept and is correct -- if the account is later
enabled for logprobs, or is pointed at Vertex, it will work unchanged. Two
details in it that are easy to get wrong:
- `thinkingConfig.thinkingBudget: 0`, because 2.5-series models otherwise spend
  the single allowed output token on internal reasoning and return an empty
  candidate.
- `logprobs` is capped at **20**, which comfortably covers the 8 label tokens.

---

## 4. What was verified, and how

**Be precise about this distinction — it is the point of the phase.**

### Verified against REAL data ✅

| Claim | Evidence |
|---|---|
| PAMAP2 present and well-formed | 9 Protocol files, 54 columns, 2,872,533 rows, 8.0 hours |
| Label map is correct | all 8 expected activityIDs present; **no undocumented IDs** |
| Excluded activities behave as designed | IDs 7/16/17/24 reported as INFO, not false alarms |
| Windowing produces usable data | **9,909 windows**, all 8 classes represented |
| Column indices are right | wrist 4/5/6, chest 21/22/23, ankle 38/39/40 — consistent with 17-column IMU blocks at 3/20/37 |
| Feature maths | energy and std cross-checked against independent numpy to 6 dp |

### Verified against MOCKS / FIXTURES ⚙️ (code correct; says nothing about the science)

| Path | Result |
|---|---|
| llama.cpp good response | parses, `max_prob` 0.783984, sums to 1.0 |
| vLLM `echo`+`top_logprobs` | identical distribution |
| Gemini good response | identical distribution — shared softmax confirmed |
| Uniform distribution | **rejected** (the anti-noise guard fires) |
| Missing logprob field | fails loudly |
| Gemini-style missing `logprobsResult` | fails loudly, names the fix |
| Ollama `/v1` endpoint | hard-refused |
| Missing `GEMINI_API_KEY` | actionable message |
| Corrupted label map (ID removed + undocumented ID injected) | both caught, exit 1 |
| Frontend render | 13/13 results panels, 13/13 dataset panels, 7/7 empty-state, malformed-profile recovery |

### Tested against the LIVE Gemini API ❌ (key supplied 2026-08-29)

| Claim | Outcome |
|---|---|
| Gemini returns usable logprobs | **NO** — all 12 probed models return `400 "Logprobs is not enabled for this model"` |
| Gemini generates text at all | yes, `200 OK` — isolating logprobs as the blocker |
| The client handles it correctly | yes — surfaces the API's own message and refuses to proceed |

### Still NOT answered ❌

- **Gate 1 (logprob extraction)** — needs Vertex AI or a local backend.
- **Gate 2 (baseline accuracy)** — blocked behind Gate 1.

`results/phase1/logprob_check.json` holds a genuine `pass: false` recording the
real API rejection. **No results were fabricated at any point.**

---

## 5. Design decisions that depart from the original brief

Each was a deliberate call; revisit if you disagree.

1. **Label-map warnings are tiered.** The brief said to warn prominently on any
   unexpected activityID with a large row count. PAMAP2's Protocol files
   legitimately contain IDs 7/16/17/24 with 40k–240k rows each, so the literal
   rule fires on every correct dataset and trains you to ignore warnings. Now:
   missing-expected → **warning**; *undocumented* ID → **warning**;
   documented-but-excluded → **INFO**. Confirmed correct on real data.

2. **`step2_n_probs` defaults to 20, not 8.** With only 8 candidate slots, mass
   on `"\n"` or `<eos>` can push a real label token out of the top-k and
   silently zero it.

3. **Windows never straddle** an activity change or a recording gap. Windows are
   cut only inside contiguous single-activity segments; otherwise labels and
   statistics are meaningless.

4. **Results are mirrored into `frontend/results/`.** `python -m http.server`
   refuses to serve paths above its root (verified), so a page served from
   `frontend/` cannot reach `../results/`. Both locations are gitignored.

5. **Token normalisation.** Mass from `"A"`, `" A"` and sentencepiece `"▁A"` is
   summed in log space before renormalising, so a model that emits a leading
   space is not mistaken for one that never answered.

6. **`.gitignore` uses `data/raw/*` + `!data/raw/README.md`.** Git cannot
   re-include a file under an excluded directory, so the original pattern would
   have silently dropped the dataset documentation.

7. **Fast CSV parser.** PAMAP2 is single-space separated, so `sep=" "` uses the
   C engine (~1.6 s/subject vs ~15 s with the regex engine).

---

## 6. Results so far

```
Dataset          2,872,533 rows · 8.0 hours · 9 subjects · 9,909 windows
Label map        PASS — all 8 IDs present, no undocumented IDs
Logprob check    NOT YET RUN against a live model (current file: pass=false)
Baseline accuracy NOT YET RUN
```

Windows per class: lying 1487 · sitting 1424 · standing 1441 · walking 1837 ·
running 757 · cycling 1277 · ascending_stairs 892 · descending_stairs 794.

**Subject109 has only 8,477 rows across 2 activities and yields 0 windows** —
it is excluded from both the test and few-shot sets. Test subjects are
101/105/106; few-shot comes from 102/103/104/107/108 (disjointness is enforced
at runtime — the script aborts if they overlap).

---

## 6a. GATE RESULTS — run live 2026-08-30

Two logprob providers were credentialed. Both were **verified empirically
before use**, not trusted — the same check that previously caught Ollama and
the Gemini Developer API silently lacking logprobs.

| Provider | Model | Logprobs | Role |
|---|---|---|---|
| OpenAI | `gpt-4o` | **verified live** | paid — paper results |
| Cerebras | `gemma-4-31b` | **verified live** | free — development, demos |

### Gate 1 — logprob extraction: **PASS**

Both return a real, non-uniform distribution summing to 1.0.

| Provider | max_prob | 2nd | Shape |
|---|---|---|---|
| gpt-4o | 0.9019 | 0.0981 | informative spread |
| gemma-4-31b | 0.99999 | 0.00001 | **near-saturated** |

**Cerebras' saturation is a finding, not a detail.** This project measures
whether a model becomes *confidently wrong*. A model pinned at p = 0.99999 has
almost no headroom to express degraded confidence, so calibration work on it
would compress into the top bin. Develop on Cerebras; measure on OpenAI.

### Gate 2 — baseline accuracy: **FAIL (0.4722 < 0.65)**

144 windows, subjects 101/105/106, gpt-4o, 8 classes (chance 0.125).

| Class | Accuracy | | Class | Accuracy |
|---|---|---|---|---|
| cycling | **1.000** | | lying | 0.278 |
| running | 0.889 | | sitting | 0.278 |
| descending_stairs | 0.778 | | standing | 0.056 |
| ascending_stairs | 0.500 | | **walking** | **0.000** |

Three most-confused pairs — **100 % involve stairs**:

- `standing → ascending_stairs` — 14 windows (77.8 % of the class)
- `walking → ascending_stairs` — 13 windows (72.2 %)
- `ascending_stairs → descending_stairs` — 9 windows (50.0 %)

Walking scoring **0.000** is the striking result: every walking window was
absorbed into a stair class.

### Probe: does dropping stairs fix it? Partly — and it relocates the problem

The script's own ranked diagnosis put "(c) drop the two stair classes" first,
so that was **measured rather than assumed**
(`scripts/probe_label_set.py` → `results/phase1/label_set_probe.json`).

| Label set | Chance | Accuracy | Verdict |
|---|---|---|---|
| 8-class | 0.125 | 0.4722 | FAIL |
| **6-class** | 0.167 | **0.5833** | **still FAIL** (+0.111) |

Dropping stairs fixes the *dynamic* classes outright — walking 0.000 → 0.750,
running 1.000, cycling 0.938 — but the failure moves to the **static
postures**: sitting 0.062, lying 0.375, standing 0.375, now confusing
`sitting → standing` (62.5 %) and `standing → cycling` (43.8 %).

Static postures differ almost entirely by **gravity orientation**, which the
45-number feature summary does contain (per-axis mean) but does not
foreground. The remaining gap is therefore a **feature/presentation problem,
not a class-count problem** — and the re-run diagnosis says exactly that.

### What this means

**The STOP condition from the Phase 1 brief is in force.** Phase 5 stays
blocked — now for a substantive reason rather than a credentials one.

Ranked by what the data actually shows:

1. **(b) Foreground orientation.** Add explicit tilt / mean-gravity-direction
   features, or restate the per-axis means as an orientation summary. This is
   where both runs' residual error now lives.
2. **(a) Few-shot examples emphasising orientation**, for the three static
   postures specifically.
3. **(c) The 6-class set** helps (+0.111) and is already measured — but it
   costs the two hardest classes, which are the most informative about
   degradation-induced confusion. A real trade, not a free win.
4. **(d) A larger model** is last: gpt-4o already scores 1.000 on cycling and
   0.889 on running, so the ceiling is not obviously model capacity.

---

## 7. Current blockers

1. ~~An API key for a logprob backend.~~ **RESOLVED 2026-08-30** — OpenAI and
   Cerebras are both credentialed and verified. Gate 1 passes.

1b. **Gate 2 accuracy: 0.4722, needs 0.65.** This is now the blocker. See §6a
   for the measured, ranked diagnosis.

   **Vertex is PARKED, not abandoned.** The `vertex` and `gemini` backends are
   left in place, unchanged and correct; they are simply removed from
   `fallback_order` so `auto` does not burn a timeout on an unauthenticated
   Vertex call on every run. Both carry a "PAUSED pending credentials" note in
   `configs/models.yaml`. Re-enable by setting `BSN_BACKEND=vertex` once GCP
   credentials exist — no code change needed.

2. Then run, in order:
   ```bash
   python scripts/check_logprobs.py            # must PASS before anything else
   python scripts/check_baseline_accuracy.py   # must reach >= 0.65
   ```

**If baseline accuracy < 0.65: STOP.** Do not start Phase 2. The script prints
the 3 most-confused class pairs and ranks the candidate fixes from what the
confusion matrix actually shows — (a) better few-shot, (b) richer features
(gyro, cross-axis correlation), (c) drop to the 6-class set, (d) larger model.
Act on that ranking rather than guessing.

---

## 7a. Two confidence methods now exist — do not mix them

`score_labels()` always returns a `confidence_method` field:

| Value | Backends | How it is produced |
|---|---|---|
| `logprob` | openai, llamacpp, vllm, vertex, gemini | the model's own next-token log-probabilities, softmax-renormalised over the label tokens |
| `self_consistency` | anthropic | k sampled answers (default k=10, temperature 0.7); distribution = vote fractions |

The Anthropic Messages API exposes **no logprobs or top_logprobs** — there is no
parameter and no model that enables it. Self-consistency is a legitimate,
literature-supported estimator (better calibrated than verbalized confidence),
but:

- its **resolution is bounded by 1/k** — at k=10 the finest distinguishable
  confidence step is 0.1, which matters for ECE binning;
- it costs **k× inference**;
- it is a **robustness check, not the primary method**.

Three structural guards enforce this rather than relying on a comment:
1. `anthropic` is **excluded from `fallback_order`**, so `auto` can never
   silently switch from a logprob method to a sampling one.
2. `require_logprobs: true` (the default) makes any sampling backend **refuse
   to run** with an explanation.
3. Every run prints a provenance banner —
   `backend=… model=… confidence_method=…` — and the value is written into the
   result JSON.

**Never pool the two in one analysis.**

---

## 8. Open questions for the user

1. **No Gemini model on the Developer API can provide logprobs** (tested, not
   assumed). Choose: Vertex AI (GCP project, billing, service-account auth) or
   a local model via llama.cpp / vLLM (free, already implemented, and what the
   original brief specified).
2. **Hosted vs local.** The original brief specified a *local* LLM. Moving to a
   hosted API changes cost, reproducibility and rate-limiting characteristics,
   and means ~150 sequential calls per accuracy run. Worth an explicit decision
   before the degradation sweeps, which will be far larger.
3. **Phase 2–10 names** in `frontend/index.html` are placeholders.

---

## 9. How to approach Phase 2

**Do not start until both Phase 1 gates pass.**

Phase 2 builds the real `DataSource`-based parser. Carry forward:

- **The verified column map, and only that** (documented in
  `data/raw/README.md`): timestamp 0, activityID 1, accel16 at 4/5/6 (wrist),
  21/22/23 (chest), 38/39/40 (ankle). Full 54-column semantics — gyro at block
  offsets +7..+9, magnetometer +10..+12, orientation +13..+16 (**invalid in
  this collection, never use**) — belong to Phase 2.
- **The windowing contract**: 2.56 s, 50 % overlap, contiguous single-activity
  segments only, ≥60 % sample coverage. Phase 1's window counts are the
  reference; a rebuilt parser should reproduce ~9,909 windows.
- **`_llm_client.score_labels()`** unchanged — the backend recorded in
  `logprob_check.json` must be reused, since confidence numbers are not
  comparable across serving stacks.

Treat `scripts/check_baseline_accuracy.py` as **reference, not a base class**.
It is intentionally disposable; do not extend it into the real pipeline.

Still explicitly out of scope until their own phases: failure injection, the
health monitor, and LangGraph.

### Adding the Phase 2 page

`frontend/index.html` holds a single `PHASES` array. Shipping a phase means
adding one entry (or flipping `page` from `null` to a filename). Follow the
existing panel conventions: single-hue sequential ramps for magnitude, status
colour always paired with a text label, and every panel degrading to a
"run the script first" notice when its JSON is absent.
