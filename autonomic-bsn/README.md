# Autonomic Agentic BSN — Phase 1: foundational de-risking

Research question: **do LLM agents doing human activity recognition from
body-worn sensors become "confidently wrong" when the sensor network
degrades?**

Before any of that system gets built, this phase answers two yes/no questions.
If either fails, we stop and fix it rather than writing more code.

| # | Question | Script | Gate |
|---|---|---|---|
| 1 | Can we extract **real token log-probabilities** from a local LLM? | `scripts/check_logprobs.py` | non-uniform distribution summing to 1.0 |
| 2 | Is zero-shot HAR **accurate enough on clean data** to leave headroom? | `scripts/check_baseline_accuracy.py` | overall accuracy ≥ **0.65** over 8 classes |

Question 1 gates everything: without a genuine confidence signal, "confidently
wrong" is not measurable. Question 2 gates the experiment's dynamic range: if
the model is already near chance on pristine sensors, later degradation results
are unattributable.

> **Phase 1 is deliberately disposable.** The PAMAP2 parsing here is throwaway;
> only the *verified column indices* carry forward. Phase 2 rebuilds the real
> parser behind `DataSource`. Do not add failure injection, the health monitor,
> or LangGraph to this phase.

---

## 1. Setup

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
```

## 2. Start a model server

**Primary target: llama.cpp** — runs on CPU, no GPU required.

```bash
llama-server -m <path-to-gguf> -c 4096 --port 8080
```

> **Context size:** `-c 4096` is the documented minimum and is enough for
> Step 1. Step 2's prompt carries 16 few-shot examples (2 per class × 8
> classes) and lands around ~2.1k tokens, which fits — but if you raise
> `few_shot_per_class` in `configs/models.yaml`, start the server with
> `-c 8192` or the prompt will be silently truncated from the left, which
> quietly destroys accuracy.

Then confirm it is up:

```bash
curl http://localhost:8080/health
```

### Backends and the Ollama trap

Selected via `backend:` in `configs/models.yaml`; use `auto` to try each of
`fallback_order` until one returns real logprobs.

| Backend | Endpoint | Logprobs |
|---|---|---|
| `llamacpp` *(primary)* | `POST /completion` | `n_probs` + `post_sampling_probs: false` |
| `vllm` *(fallback 1)* | `POST /v1/completions` | `logprobs: N` + `echo: true` — genuinely supported |
| `ollama_native` *(fallback 2)* | `POST /api/generate` | `logprobs: true` + `top_logprobs: N`, **verified at runtime** |

> ⚠️ **Never use Ollama's OpenAI-compatible endpoint** (`/v1/chat/completions`
> or `/v1/completions` on port 11434). It **silently drops the `logprobs`
> field even when you request it** — it is listed as unsupported in Ollama's
> own OpenAI-compatibility documentation, and
> [ollama/ollama#16117](https://github.com/ollama/ollama/issues/16117) is the
> open request for it. It returns `200 OK` with a perfectly normal-looking
> body and no usable confidence data, so building against it would *appear*
> to work while returning nothing. `_llm_client.py` hard-blocks any endpoint
> matching `:11434` + `/v1/` and refuses to run.
>
> Ollama's **native** `/api/generate` only gained logprob support recently, so
> the client explicitly verifies the field is present in the response and
> fails loudly if it is not, rather than silently degrading to free-text
> parsing.

Whichever backend passes is recorded as `backend` in
`results/phase1/logprob_check.json`. **Every later phase must use that same
one** — confidence numbers are not comparable across serving stacks.

## 3. Get the data

See [`data/raw/README.md`](data/raw/README.md) for the exact download commands
and the full 54-column format. Phase 1 expects:

```
data/raw/pamap2/Protocol/subject101.dat … subject109.dat
```

## 4. Run the checks

```bash
# Step 1 — logprob extraction
python scripts/check_logprobs.py
python scripts/check_logprobs.py --backend auto     # try the fallback chain

# Step 2 — label-map sanity check only (no LLM calls, fast)
python scripts/check_baseline_accuracy.py --verify-labels-only

# Step 2 — full baseline accuracy (~150 windows)
python scripts/check_baseline_accuracy.py
python scripts/check_baseline_accuracy.py --n-windows 24   # quick smoke run
```

Both scripts exit **0 on PASS, 1 on FAIL**, and write to `results/phase1/`
(plus a mirror into `frontend/results/phase1/`, see below).

## 5. View the dashboard

```bash
cd frontend && python -m http.server 8000
```

Then open <http://localhost:8000/>.

> **Why results are mirrored:** `http.server` refuses to serve paths above its
> root, so a page served from `frontend/` cannot reach `../results/`. The
> scripts therefore write results twice — the canonical copy in `results/` and
> a mirror in `frontend/results/` — and the page tries the mirror first,
> falling back to `../results/` if you instead serve from the project root.
> Both locations are gitignored.

The dashboard is cumulative. `frontend/index.html` holds a single `PHASES`
array; shipping a phase means adding one entry (or flipping its `page` from
`null`). Phases 2–10 render greyed out with a "not yet built" tag, and their
names are placeholders — rename them as the roadmap firms up.

---

## What each check actually asserts

### Step 1 — `check_logprobs.py`

1. `POST /completion` with a **dummy** feature summary and 6 labels A–F,
   `{"n_predict": 1, "n_probs": 6, "post_sampling_probs": false,
   "temperature": 0}`.
2. Parse `completion_probabilities[0].probs`, renormalise via softmax over the
   logprobs: `p_i = exp(lp_i) / Σ_j exp(lp_j)`.
3. Assert the result **sums to 1.0 within 1e-6** and is **not uniform**
   (`max_prob` must exceed 1/6 by ≥ `uniform_margin`, default 0.05). *A uniform
   distribution means we are reading noise, not real model confidence* — this
   is the single most important assertion in the phase.
4. Write `results/phase1/logprob_check.json` with
   `{backend, endpoint, model, pass, distribution, max_prob, notes}`.

Tokenizer note: mass from variants that normalise to the same letter (`"A"`,
`" A"`, sentencepiece `"▁A"`) is summed in log space before renormalising, so a
model that emits a leading space is not mistaken for one that never answered.

### Step 2 — `check_baseline_accuracy.py`

1. Read `subject10{1..9}.dat`, extracting only the verified columns
   (timestamp 0, activityID 1, accel16 at 4/5/6, 21/22/23, 38/39/40).
2. **Verify the label map first** — print unique activityIDs with per-subject
   row counts, then classify each: missing expected ID → prominent warning;
   undocumented ID with volume → prominent warning; documented-but-excluded
   PAMAP2 ID (7/16/17/24) → INFO, since those legitimately appear in the
   Protocol files and warning on them every run would just train you to ignore
   warnings.
3. Filter to the 8 target IDs, drop `activityID == 0`, drop NaN accel rows.
4. Window by timestamp into **2.56 s windows with 50 % overlap** — cut only
   inside contiguous single-activity segments, so no window straddles an
   activity change or a recording gap.
5. Per window: 3 nodes × 3 axes × 5 stats (mean, std, min, max, energy) = **45
   features**.
6. Render each window as a compact natural-language summary; prompt at
   temperature 0 with **2 few-shot examples per class drawn from subjects
   disjoint from the test set** (enforced — the script aborts if they overlap).
7. Predict by **argmax over the 8 label-token logprobs**, using the same
   extraction path as Step 1 — never free-text parsing.
8. Report overall accuracy, per-class accuracy, and an 8×8 confusion matrix.
9. Write `results/phase1/baseline_accuracy.json`, including the **exact prompt
   template and a fully rendered example prompt** for the paper's
   reproducibility appendix.
10. Print PASS (≥ 0.65) or FAIL. **On FAIL** it prints the 3 most-confused
    class pairs and ranks the candidate fixes *from what the confusion matrix
    actually shows* — (a) better few-shot, (b) richer features, (c) drop to the
    6-class set, (d) larger model — rather than guessing.

`step2_n_probs` defaults to **20**, not 8: with only 8 candidate slots,
probability mass parked on `"\n"` or `"<eos>"` can push a real label token out
of the top-k and silently zero it.

---

## Definition of done

- [ ] `results/phase1/logprob_check.json` exists with `pass: true`, a
      non-uniform distribution, and a named backend that actually worked
- [ ] `results/phase1/baseline_accuracy.json` exists with
      `overall_accuracy >= 0.65`
- [ ] the activityID verification ran and reported no unexplained mismatches
- [ ] `frontend/phase1_derisk.html` renders both results when served locally
- [x] `.gitignore` excludes `data/raw/`, `results/`, `*.gguf`, `.venv/`,
      `__pycache__/`

**If baseline accuracy < 0.65: STOP. Do not proceed to Phase 2.** The script
prints the ranked diagnosis; act on that before writing any more code.
