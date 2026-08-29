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

# 2. add your API key
copy .env.example .env                 # PowerShell;  cp .env.example .env on Unix
#    then edit .env and set GEMINI_API_KEY

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

## Model backends — read this before changing the model

The only thing this project needs from a model is a **real next-token
log-probability distribution**. Several popular endpoints return `200 OK` with
no usable confidence data, which fails *invisibly*. Two such traps are guarded
in code:

### ⚠️ Gemini 3.x does not support logprobs

`responseLogprobs` works on **`gemini-2.5-flash`** and `gemini-2.5-pro`.
It does **not** work on the **Gemini 3.x family** (`gemini-3-flash`,
`gemini-3-pro`), which either rejects the request with *"Logprobs is not
supported for this model"* or returns a candidate with `logprobsResult`
absent/null.

A model without logprobs is unusable for this project regardless of how good
its text output is. The default is therefore `gemini-2.5-flash`, and the client
fails loudly naming the fix if the field is missing.

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
| `gemini` *(default)* | `generateContent` | needs `GEMINI_API_KEY`; `logprobs` capped at 20 |
| `llamacpp` | `POST /completion` | local, CPU-friendly |
| `vllm` | `POST /v1/completions` | local; genuinely supports logprobs |
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
scripts/
  _llm_client.py        THE shared logprob extraction path (all backends)
  check_logprobs.py     gate 1
  check_baseline_accuracy.py  gate 2
  profile_dataset.py    dataset profile for the dashboard (no LLM needed)
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
