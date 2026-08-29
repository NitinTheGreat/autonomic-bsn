# Phase 3 — Failure Injection Lab

**Status:** complete and verified. Makes **zero LLM calls**, so Phase 1's
unpassed gates did not block it. **Phase 5 remains blocked.**

**Date:** 2026-08-30

---

## 1. Goal

Implement the five-failure taxonomy as a `DataSource` **decorator** that wraps
any `DataSource` and is itself one — so it composes with `DatasetReplaySource`
today and Phase 9's `HardwareLiveSource` unchanged, and with other injectors.

---

## 2. What was built

```
injection/base.py               FailureInjector + InjectorStrategy ABC
injection/dropout.py            F1 battery exhaustion
injection/clock_desync.py       F2 timestamp drift
injection/packet_loss.py        F3 Gilbert-Elliott bursty loss
injection/rate_degradation.py   F4 throttling (zero-order hold)
injection/displacement.py       F5 sensor rotation
injection/registry.py           type -> class map + make_injector factory
scripts/verify_injectors.py     -> results/phase3/injector_verification.json
scripts/export_phase3_samples.py -> 100 paired {clean, injected} traces
tests/test_injection.py         71 assertions
frontend/phase3_injection.html  before/after explorer + calibration table
```

Phase 6's sweep iterates `registry.STRATEGIES`, so a sixth failure type will
need no sweep-code change.

---

## 3. One additive change to the Phase 2 contract

`Window` gained a `meta: dict = field(default_factory=dict)`. Phase 2 said not
to alter the shape; this is **additive with a default**, so every existing
construction and every `DataSource` implementation — including Phase 9's
hardware source — remains valid unchanged. Injectors record `failure_type`,
`severity`, `target_node`, `seed`, `requested` and `realised` there. Plain
sources leave it empty.

---

## 4. Invariants, enforced structurally

| Invariant | Why |
|---|---|
| **No injector ever writes `None`** | `None` means "no such sensor". A dead node still *has* its sensors. Writing `None` would make an injected failure indistinguishable from MHEALTH's absent chest gyro. |
| **Blanked samples are NaN, not zero** | Zero is a real stationary reading a model may reasonably believe. NaN says "no measurement". |
| **`rate_degradation` emits zero NaNs; `packet_loss` emits gaps** | Phase 4 must separate "stale but present" from "absent", so Phase 3 generates them distinguishably. |
| **Capability check RAISES, never no-ops** | A silent no-op means the requested injection rate ≠ the realised one, and Phase 6's severity axis is quietly wrong. |
| **Own `numpy.Generator` per injector, seeded** | Global state would make output depend on call order across the sweep. Tested by interleaving two generators. |
| **`injected_failure` is never read by the monitor** | Stated in the module docstring. If Phase 4's monitor can see the ground truth it must infer, its metrics measure only its ability to read an answer key. |

---

## 5. Results

### Gilbert-Elliott calibration — all four severities within tolerance

20 chains × 10,000 steps per severity:

| sev | target L | realised L | ΔL | target B | realised B | ΔB | verdict |
|---|---|---|---|---|---|---|---|
| 1 | 0.05 | 0.0491 | 0.0009 | 3 | 2.973 | 0.91 % | ✅ |
| 2 | 0.15 | 0.1475 | 0.0025 | 4 | 3.990 | 0.24 % | ✅ |
| 3 | 0.30 | 0.3029 | 0.0029 | 6 | 6.043 | 0.72 % | ✅ |
| 4 | 0.50 | 0.5009 | 0.0009 | 8 | 8.052 | 0.65 % | ✅ |

Tolerance ±0.02 absolute on L, ±15 % on B. The model is exact: π_Bad = L to
**1e-12** for every severity.

### Monotonicity — all five, both datasets

```
dropout            frac_nan            0.253  0.503  0.753  1.000
clock_desync       delta_ms             50     200    500   2000
packet_loss        loss_rate           0.040  0.129  0.295  0.502
rate_degradation   target_hz            50     25     12.5   6.25   (decreasing)
displacement       realised_angle_deg  11.03  22.05  33.03  65.64
```

---

## 6. Where a stated severity did NOT produce the expected realised effect

Two cases. **Neither is a formula error**, and both matter for Phase 6.

### 6.1 Displacement: realised angle is posture-dependent

The rotation *is* applied at exactly θ. But the realised metric — the angle
between pre- and post-rotation **mean gravity** — depends on how closely
gravity lies along the rotation axis:

    cos(realised) = cos²φ + sin²φ·cos(θ)      φ = angle(gravity, axis)

At φ = 90° realised equals θ; at φ = 0 it is zero, because spinning a vector
about itself changes nothing. Measured on PAMAP2 subject101, requested 15°:

| node | activity | realised @15° | realised @90° | \|g·axis\| |
|---|---|---|---|---|
| ankle | lying | **14.89°** | 89.15° | 0.122 |
| ankle | walking | 3.49° | 18.99° | 0.972 |
| chest | standing | **2.33°** | 12.63° | 0.988 |
| wrist | running | 12.92° | 75.10° | 0.507 |

A requested 15° realises anywhere from 2.3° to 14.9° depending on posture — a
**6× spread for the same severity**.

The severity *ordering* is preserved everywhere (monotonic within any given
window), so the axis is still usable. But the magnitude is **not comparable
across activities or nodes**. The injector now records
`gravity_axis_alignment` in every window's realised metrics.

> **Phase 6 must not treat displacement severity as a fixed physical
> magnitude.** Compare within an activity, or condition on the alignment.

Arguably the realised metric is the *more* meaningful quantity: it measures how
much of the rotation is observable in the gravity signal, which is what
actually reaches a classifier. But it must be reported, not assumed equal to θ.

### 6.2 The 10,000-step Gilbert-Elliott check is underpowered

The first verification run **failed** at severity 4: realised L 0.4674 against
a target of 0.5000 (ΔL 0.0326, outside the ±0.02 band).

Diagnosis: **not the formula.** π_Bad = L exactly, and across 200 seeds the
mean bias is −0.0008. It is sampling variance — bursty chains are highly
autocorrelated, so a single 10k chain has sd ≈ 0.013 at severity 4:

| chain length | sd | seeds within ±0.02 |
|---|---|---|
| 10,000 | 0.0108 | 95 % |
| 50,000 | 0.0053 | 100 % |
| 500,000 | 0.0018 | 100 % |

Verification now averages 20 independent 10k chains and records the per-chain
spread and single-chain pass rate, so the underpowering stays visible.

### 6.3 A verification criterion that was wrong (mine, not the injector's)

"Assert the target node measurably differs" failed for MHEALTH `packet_loss`
severity 1. At L = 0.05, B = 3 over a 128-sample window, the chance of drawing
**no loss at all** is ≈ (1−p)¹²⁸ ≈ 10 %, so some windows legitimately come back
unmodified. Demanding that *every* window differ fails on correct behaviour.
Stochastic types are now held to an aggregate criterion; deterministic types
still must change every window.

---

## 7. Verification

| Suite | Result |
|---|---|
| `tests/test_injection.py` | **71/71** |
| `scripts/verify_injectors.py` | ALL CHECKS PASSED (both datasets, 5 × 4) |
| Phase 3 frontend — default view | 17/17 |
| Phase 3 frontend — NaN gaps | 8/8 |
| Phase 3 frontend — rate degradation | 2/2 |
| Phase 3 frontend — empty state | 5/5 |
| Phase 2 `test_datasource.py` | 45/45 — **unweakened** |
| Phase 1 `test_backends.py` | 56/56 — **unweakened** |
| Phase 1 + 2 frontend | 45/45 — unweakened |

**Zero LLM calls**, confirmed by grep across all Phase 3 files for
`_llm_client|score_labels|openai|anthropic|gemini|vertex|llamacpp|vllm|logprob|api_key|generateContent|chat/completions`
— 0 matches. No network imports — 0 matches.

---

## 8. How to approach Phase 4 (health monitor)

**Phase 5 stays blocked** until Phase 1's Gate 1 and Gate 2 pass. Phase 4 is
not blocked.

The single most important constraint:

> **The health monitor must NEVER read `NodeFrame.injected_failure` or
> `Window.meta["failure_type"] / ["severity"] / ["realised"]`.**
>
> Those fields are ground truth. A monitor that reads them is scored on its
> ability to read an answer key, and every detection metric becomes circular
> and meaningless. Keep scoring in a **separate module** that sees both the
> monitor's output and the ground truth; the monitor itself sees only frames.
>
> Consider enforcing this mechanically — e.g. hand the monitor a view of the
> window with those fields stripped — rather than relying on discipline.

What the monitor has to be able to separate, and what Phase 3 guarantees is
distinguishable:

| Signature | Failure |
|---|---|
| NaN gaps, bursty | `packet_loss` |
| NaN from a point onward, never recovering | `dropout` |
| **No NaN**, repeated values | `rate_degradation` |
| Values unchanged, timestamps offset | `clock_desync` |
| Values rotated, no gaps, no staleness | `displacement` |

Also carry forward:

- **Realised, not requested.** Score against `meta["realised"]`; for
  displacement the two differ by up to 6× (§6.1).
- **MHEALTH's clock is derived.** `clock_source` is recorded per window.
  Desync results on MHEALTH are not comparable to PAMAP2's.
- **MHEALTH's chest has no gyro.** A gyro-based health feature must branch on
  `channels_present`, not assume every node has one.
- **NaN ≠ 0.** A monitor that imputes zeros before computing features will
  destroy the very signal it needs.

### Adding the Phase 4 page

`frontend/index.html` holds a single `PHASES` array — flip `page` from `null`.
Follow the existing conventions: single-hue sequential ramps for magnitude,
status colour always paired with a text label, and every panel degrading to a
"run the script first" notice when its JSON is absent.
