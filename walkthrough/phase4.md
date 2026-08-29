# Phase 4 — Health Monitor

**Status:** complete and verified. **Zero LLM calls.** Phase 5 remains blocked
(Phase 1 Gates 1 & 2 still unpassed — no credentialed logprob backend).

**Date:** 2026-08-30

---

## 1. Goal

Detect and diagnose node degradation from observable signal statistics alone —
never from ground truth. These numbers become a Results subsection.

---

## 2. The blind, enforced mechanically

`health/window_view.py` was built first, and everything depends on it.

| Stripped | Preserved |
|---|---|
| `NodeFrame.injected_failure` | `channels_present` |
| `Window.label` (the true activity) | `timestamp_derived`, `clock_source` |
| `failure_type`, `severity`, `target_node`, `seed` | `dataset_name`, `subject_id` |
| `requested`, `realised`, `gravity_axis_alignment`, `tag` | `sampling_rate_hz` |

Access **raises `GroundTruthAccessError`** rather than returning `None`.
Returning `None` would let buggy monitor code read nothing and still appear to
work — a far harder bug to catch than a crash.

`signals.py` and `diagnose.py` accept only a `BlindWindow` and reject a raw
`Window` with an explanation. A grep test confirms `injected_failure`,
`failure_type`, `severity` and `realised` appear **nowhere** in either file —
including comments, which forced renaming `expected_realised_angle` →
`expected_observable_angle`.

**One addition beyond the brief:** `label` is stripped too. In deployment the
activity is exactly what the system infers, so a monitor conditioning on the
true activity uses information it would never have. Scoring reads the original
`Window`, so stratification by activity is unaffected.

---

## 3. Results

### Detection F1 by failure type (pooled across severities)

| Failure | Precision | Recall | **F1** | Target |
|---|---|---|---|---|
| `dropout` | 0.923 | 1.000 | **0.960** | > 0.8 ✅ |
| `rate_degradation` | 0.923 | 1.000 | **0.960** | > 0.8 ✅ |
| `clock_desync` | 0.923 | 1.000 | **0.960** | reported honestly ✅ |
| `packet_loss` | 0.897 | 0.729 | **0.805** | > 0.8 ✅ |
| `displacement` | 0.846 | 0.458 | **0.595** | reported honestly ⚠️ |

All three required targets clear 0.8. `clock_desync` came out strong;
`displacement` is weak **for a geometric reason, documented below**.

### False positives on clean windows

**FPR = 0.0667** — 12 of 180 clean node-windows flagged, all 12 misdiagnosed as
`displacement`. No clean window was ever mistaken for dropout, packet loss,
rate degradation or desync.

### Diagnosis confusion (6 classes, n = 1380, accuracy **0.880**)

```
                  healthy dropout packet_l rate_deg clock_de displace
healthy               168       0        0        0        0       12
dropout                 0     240        0        0        0        0
packet_loss            17       6      216        0        0        1
rate_degradation        0       0        0      240        0        0
clock_desync            0       0        0        0      240        0
displacement          130       0        0        0        0      110
```

Where types are confused:

- **`displacement` → `healthy` (130 of 240).** The dominant error, and it is
  geometric — see §4.
- **`packet_loss` → `healthy` (17).** At the mildest setting some windows draw
  under the 2 % missingness threshold. Genuinely marginal.
- **`packet_loss` → `dropout` (6).** A burst that happens to land at the window
  end looks terminal. Honest ambiguity: within one window those two are
  indistinguishable by construction.
- **`dropout`, `rate_degradation`, `clock_desync`: perfect, 240/240 each.** The
  three signatures Phase 3 was built to keep separable stayed separable.

---

## 4. Displacement by (node, activity) — an explicit limit

| Node / activity | n | Recall | Diagnosis acc. | Mean observed | Mean alignment | Verdict |
|---|---|---|---|---|---|---|
| ankle / lying | 120 | 0.42 | 0.42 | 34.9° | 0.315 | **undetectable** |
| ankle / standing | 120 | 0.50 | 0.50 | 43.5° | 0.243 | borderline |

**Two compounding effects make gravity-based displacement detection weak:**

1. **Observability geometry** (Phase 3 §6.1). A rotation θ about the node's long
   axis changes the observed gravity direction by
   `cos(obs) = cos²φ + sin²φ·cos(θ)`. Where gravity lies near the axis,
   almost nothing is observable. The monitor's model reproduces Phase 3's
   independent measurement to within 0.02° (predicts 2.31°, measured 2.33°).

2. **Natural posture swing swamps the effect.** A body-worn node's gravity
   direction moves with posture far more than a 15–90° sensor rotation moves
   it. Calibrating on clean held-out windows shows the natural spread exceeds
   what the rotation itself produces, so any threshold that clears the spread
   also rejects most genuine displacements.

**Statement for the paper:** displacement of a body-worn node is *not reliably
detectable from gravity direction alone*. Detecting it needs either a temporal
baseline of the same node (comparing against its own recent history rather than
a population reference) or a second modality. This is a limit of the method,
not a defect in the implementation.

---

## 5. Bugs found during evaluation

Each made the monitor look better or worse than it was.

**5.1 Diagnosed but never flagged.** A 2000 ms desync was named correctly while
the node stayed `HEALTHY` — no sub-score reflected cross-node offset. Detection
recall read **0.417** against a *perfect* 240/240 diagnosis. Adding a synchrony
sub-score took `clock_desync` F1 to **0.960**. Worth remembering: a high
diagnosis score with a low detection score is the signature of exactly this
class of bug.

**5.2 Stillness penalised as staleness.** `unique_value_ratio` on a lying ankle
is 0.82 through sensor quantisation, with a max repeat *run* of 1. The monitor
was calling a still limb stale. Re-keyed on **consecutive** repeats, which
throttling produces and quantisation never does.

**5.3 Mildest throttling unnamed.** Holding every 2nd sample gives a repeat run
of 2, below the run threshold, so 45 windows were flagged but called "healthy".
The ladder now also keys on repeat *fraction* → 240/240.

**5.4 Calibration produced an 82 % false-positive rate.** A single
cross-activity mean gravity vector from a held-out subject was a poor reference,
for the reason in §4.2. Calibration now measures the natural spread of the
gravity direction on clean windows and the threshold must clear it.
**FPR 0.82 → 0.33 → 0.067.**

---

## 6. Verification

| Suite | Result |
|---|---|
| `tests/test_health.py` | **70/70** |
| Phase 4 page (DOM shim) | 19/19 |
| Phase 3 health-panel integration | 6/6 |
| Phase 4 empty state | 8/8 |
| Phase 3 `test_injection.py` | 71/71 — unweakened |
| Phase 2 `test_datasource.py` | 45/45 — unweakened |
| Phase 1 `test_backends.py` | 56/56 — unweakened |
| Phase 1 + 2 frontend | 32/32 — unweakened |

**Zero LLM calls and zero network imports**, confirmed by grep.

Housekeeping: Phase 3's sample payload dropped duplicated untouched-node
traces, **6.9 MB → 4.7 MB**; the Phase 3 page re-verified afterwards.

---

## 7. How to approach Phase 5 — STILL BLOCKED

**Do not start Phase 5 until Phase 1 Gate 1 and Gate 2 actually pass.** They
have never run against a credentialed backend. Set `OPENAI_API_KEY` (or run a
local `llama-server` and set `BSN_BACKEND=llamacpp`), then:

```bash
python scripts/check_logprobs.py            # Gate 1
python scripts/check_baseline_accuracy.py   # Gate 2, needs >= 0.65
```

What Phase 5 inherits:

- **`trust_weight` per node** — 1.0 / 0.6 / 0.25 / 0.0 by state. This is the
  hook the agent uses to down-weight a degraded node's contribution.
- **The blind still applies.** If the agent prompt is built from window data,
  build it from a `BlindWindow`, or the agent inherits the ground-truth channel
  the monitor was carefully denied.
- **Diagnosis carries evidence.** `verdict["evidence"]` is a dict with a named
  rule — usable directly as prompt context, and auditable in the paper.
- **Do not pool confidence methods.** Phase 1's `confidence_method` field is
  `logprob` or `self_consistency`; they are not comparable and `anthropic` is
  excluded from the fallback chain for that reason.
- **Displacement is weakly detected.** An agent that trusts the monitor to catch
  sensor displacement will be wrong roughly half the time. Phases 6–8 should
  either avoid depending on it or add the temporal baseline described in §4.
