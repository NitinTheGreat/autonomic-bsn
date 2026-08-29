# Phase 2 — Pluggable Data Layer & Dataset Explorer

**Status:** complete and verified against real data. Makes **zero LLM calls**,
so Phase 1's unpassed gates did not block it.

**Date:** 2026-08-29

---

## 1. Goal

Implement the `DataSource` contract that every later phase plugs into —
including Phase 9's ESP32/BLE hardware source — backed by real PAMAP2 and
MHEALTH data, plus a frontend explorer that renders real waveforms.

Phase 1's gates (logprob extraction, baseline accuracy) are **still unpassed**
for want of a credentialed backend. That blocks Phase 5, not this phase.

---

## 2. What was built

```
core/datasource.py                  the contract: NodeFrame, Window, DataSource
core/labels.py                      PAMAP2_8 / CANONICAL_6, ID maps, exclusions
datasets/pamap2_loader.py           full 54-column map, rad/s -> deg/s
datasets/mhealth_loader.py          24-column map, derived clock, absent gyro
datasets/dataset_replay_source.py   reference DataSource implementation
scripts/verify_labels.py            tiered label-map cross-check
scripts/profile_datasets.py         -> results/phase2/dataset_stats.json
scripts/export_phase2_samples.py    -> frontend/data/phase2_samples/
tests/test_datasource.py            47 assertions against real data
frontend/phase2_explorer.html       waveform explorer + comparison table
```

`core/datasource.py` is the load-bearing file. Phase 9's `HardwareLiveSource`
must satisfy the same Protocol unchanged, so nothing downstream can tell a
replayed dataset from live nodes except by `NodeFrame.source`.

---

## 3. The two invariants this phase exists to protect

### 3.1 Absent sensor ≠ zero reading

`gyro_dps` is `None` when a node has no gyroscope at all. It is **never** a
zero tuple. There are three distinct states, and the code keeps them distinct:

| State | Representation |
|---|---|
| Node has a gyro, sample valid | `(x, y, z)` |
| Node has a gyro, sample dropped | tuple containing `NaN` |
| Node has **no** gyro | `None` |

A zero gyro reading is a real measurement meaning "not rotating". Zero-filling
an absent sensor would fabricate that signal inside precisely the degradation
study this project runs, and would make a healthy node indistinguishable from a
stuck one. Every frame also carries `meta["channels_present"]` so consumers
branch on declared availability rather than inferring it from `None`-ness.

Asserted in tests over 5,122 MHEALTH chest frames: `gyro_dps is None` on every
one, and zero tuples counted explicitly at **0**.

### 3.2 `injected_failure` must not be read by the health monitor

Written only by Phase 3's injector, read only by Phase 4's separate scoring
module. The module docstring states this explicitly: if the monitor can see the
ground-truth label it is supposed to infer, its detection metrics become
circular — it would be scored on reading an answer key. It stays `None`
throughout Phase 2, asserted in tests.

---

## 4. Results

### Reference check — exact

PAMAP2, all 9 subjects, PAMAP2_8:

```
actual 9,909   reference 9,909   delta +0 (0.000%)   tolerance ±1%   PASS
```

Per-subject counts match Phase 1 one for one (1305 / 1305 / 967 / 1221 / 1388 /
1260 / 1177 / 1286 / 0). The rebuilt parser did **not** diverge.

The detail that makes it exact: a row is dropped when any **accelerometer**
channel is NaN, matching Phase 1. Gyro NaN does not drop the row — the node has
a gyroscope, it merely dropped a sample. Dropping on gyro too would have
silently changed the count.

`subject109` yields 0 windows (8,477 rows across 2 activities) — expected, not
an error, and asserted as such.

### Table 1 (`results/phase2/dataset_stats.json`)

| | PAMAP2 | MHEALTH |
|---|---|---|
| Subjects | 9 | 10 |
| Sampling rate | 100 Hz | 50 Hz |
| Label set | PAMAP2_8 (8) | CANONICAL_6 (6) |
| Raw rows | 2,872,533 | 1,215,745 |
| Duration | 8.0 h | 6.8 h |
| Windows (2.56 s / 50 %) | **9,909** | **2,760** |
| Timestamp | measured (col 0) | **derived** from row index |
| chest channels | accel + gyro + mag | **accel + ECG only** |

MHEALTH's classes are exactly balanced at 460 windows each — unusual, and worth
remembering: it needs no class balancing when sampling. PAMAP2 is not balanced
(walking 1837 … running 757).

### Label verification

Both datasets pass with **no unexplained mismatches**. All mapped IDs present;
no undocumented IDs. Excluded-by-design IDs report as INFO with their reason:
PAMAP2 7/16/17/24, MHEALTH 5/6/7/8/10/12.

---

## 5. Where the verified column maps met reality — and one place they did not

Everything positional was correct. Verified end-to-end against raw file values:

- PAMAP2 col 4 → `wrist_acc_x`: raw 2.37223 m/s² → 0.24190 g ✅
- PAMAP2 col 10 → `wrist_gyr_x`: raw −0.09222 rad/s → −5.28367 deg/s ✅
- MHEALTH col 8 → `ankle_gyr_x`: passed through unconverted (already deg/s) ✅
- MHEALTH col 0 → `chest_acc_x`: m/s² → g ✅
- Frame counts: PAMAP2 256/window, MHEALTH 128/window — exactly rate × 2.56 s

**The one claim that did not survive testing** was mine, not the dataset's. I
had written that the ±6 g accelerometer channel "saturates during vigorous
motion". Measured on subject101 running (wrist, 21,007 rows):

| Channel | observed range | at the ±6 g rail |
|---|---|---|
| accel16 | −47.30 … 66.18 m/s² (6.75 g) | — |
| accel6 | −47.17 … 62.06 m/s² (6.33 g) | **0.10 % of samples** |

accel6 does **not** hard-clip; it reports beyond its nominal rail and tracks
accel16 closely. The real reason to prefer accel16 is narrower: peaks genuinely
exceed 6 g (6.75 g observed), so accel6 cannot represent the extremes — and
those peaks are what separate running from walking. Still the right channel,
for a smaller reason than stated. The docstring has been corrected to the
measured fact.

One minor observed detail: window frame counts are occasionally 257 rather than
256 (128 → 129 for MHEALTH), because `searchsorted` includes a boundary-aligned
sample. This matches Phase 1's behaviour exactly, so it does not affect the
reference count.

---

## 6. Design decisions

1. **Windowing carried forward, not re-derived.** Phase 1's figures are the
   reference; re-deriving would have risked silent divergence.
2. **Units normalised to g and deg/s, not SI.** The planned ESP32 nodes report
   those natively, so Phase 9's hardware source needs no conversion layer.
3. **Median-motion sample selection in the exporter.** Taking the positional
   middle looked reasonable but picked the single quietest of 113
   `descending_stairs` windows (ankle sd 0.0076 vs class median 0.711), which
   rendered as a near-flat line. Ranking by motion energy fixes it.
4. **Sample windows are committed** (488 KB) so the explorer works immediately
   after clone, without the 1.6 GB PAMAP2 download.
5. **`resolve_label_set` raises** rather than returning six classes where eight
   were requested — a silent downgrade would corrupt every downstream accuracy
   and confusion figure with no visible signal.

---

## 7. Verification

| Suite | Result |
|---|---|
| `tests/test_datasource.py` | **47/47** against real data |
| Phase 2 explorer, default view | 19/19 (DOM shim, real exported data) |
| Phase 2 explorer, MHEALTH + gyro on | 9/9 (no-gyro notice, chest limited to 3 series) |
| Phase 2 explorer, empty state | 5/5 |
| Phase 1 `tests/test_backends.py` | 56/56 — **unchanged, not weakened** |
| Phase 1 frontend panels | 26/26 — unchanged |

**Zero LLM calls**, confirmed by grep across all Phase 2 files for
`_llm_client|score_labels|openai|anthropic|gemini|vertex|llamacpp|vllm|logprob|api_key|generateContent|chat/completions`
— 0 matches. No network imports either (`requests|urllib|http.client|socket`) — 0 matches.

---

## 8. How to approach Phase 3 (failure injection)

**Do not start Phase 5** until Phase 1's Gate 1 and Gate 2 actually pass.
Phase 3 and 4 are not blocked by that.

Phase 3 wraps a `DataSource` and writes `injected_failure`. Carry forward:

- **Wrap, don't fork.** The injector should consume a `DataSource` and emit a
  `DataSource`, so it composes with `DatasetReplaySource` today and
  `HardwareLiveSource` in Phase 9 without change.
- **Never zero-fill to simulate a dead sensor.** Use `None` for dropout only
  where it means "no such channel"; a dead-but-present sensor is a different
  failure and needs its own representation. Getting this wrong would collapse
  the distinction Phase 2 exists to preserve.
- **MHEALTH's clock is synthetic.** `meta["timestamp_derived"] is True`. A
  clock-desync injector must know it is perturbing a perfectly regular derived
  clock, not a recorded one — desync statistics from MHEALTH are therefore not
  comparable to PAMAP2's, whose timestamps were measured.
- **MHEALTH's chest node has no gyro to degrade.** Any gyro-targeting failure
  mode must skip it rather than silently no-op, or the injection rate will not
  match what was requested.
- `injected_failure` is a string tag; Phase 4's monitor must never read it.

### Adding the Phase 3 page

`frontend/index.html` holds a single `PHASES` array — flip `page` from `null`.
Follow the existing conventions: single-hue sequential ramps for magnitude,
status colour always paired with a text label, and every panel degrading to a
"run the script first" notice when its JSON is absent.
