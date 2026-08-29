"""Observable health signals — pure deterministic Python.

No LLM, no learned model, no ground truth. Every quantity here is computable
from the frames a real deployment would actually receive.

Accepts ONLY a BlindWindow (see health.window_view). The ground-truth
identifiers do not appear anywhere in this file; a test greps for them.

NaN handling
------------
Features are computed over the non-NaN samples using nan-aware operations.
Zeros are NEVER imputed first: a zero accelerometer reading is a real
stationary measurement, so imputing zeros would replace "no data" with "not
moving" and destroy the exact signal being measured. Asserted in tests.

Channel availability
--------------------
Every gyro-derived quantity branches on meta["channels_present"]. A node
without a gyroscope reports `unavailable` and is NOT penalised for a sensor it
never had -- MHEALTH's chest node carries accel + ECG only.

Deliberately NOT computed
-------------------------
`clock_offset_ms` and `skew_ppm` in the hardware sense need paired device/host
timestamps, which only Phase 9's real nodes provide. Replayed data has a single
timestamp per sample, so those are reported as unavailable rather than
fabricated from a number that would look plausible and mean nothing.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from health.window_view import BlindWindow, require_blind

UNAVAILABLE = "unavailable"
NOT_APPLICABLE = "not_applicable"

# Physically sane band for mean |accel| in g. At rest a body-worn node reads
# ~1 g; vigorous motion peaks a few g. Far outside this means a broken sensor.
PLAUSIBLE_MAG_MIN = 0.2
PLAUSIBLE_MAG_MAX = 8.0

# Full-scale rail for the +/-16 g accelerometers both datasets use.
FULL_SCALE_G = 16.0

# A clean node's samples are essentially all distinct. Rate degradation drives
# this ratio down hard (Phase 3 measured 16 unique of 239 at its
# strongest step).
STALENESS_FLOOR = 0.15      # unique ratio at/below this scores 0

# Cross-node timing. Nodes within this offset are considered in step; beyond
# it the sub-score falls linearly, reaching 0 at SYNC_TOLERANCE + SYNC_SCALE.
SYNC_TOLERANCE_MS = 25.0
SYNC_SCALE_MS = 200.0


def _accel_array(frames) -> np.ndarray:
    return np.array([f.accel_g for f in frames], dtype=float)


def _nan_mask(a: np.ndarray) -> np.ndarray:
    """True where the sample is missing (any axis NaN)."""
    return np.isnan(a).any(axis=1)


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return int(best)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# --------------------------------------------------------------------------- #
# per-node signal computation
# --------------------------------------------------------------------------- #
def node_signals(bw: BlindWindow, node: str,
                 cross_node_offset_ms) -> dict:
    """All observable signals for one node, plus sub-scores in [0,1]."""
    frames = bw.frames[node]
    n = len(frames)
    out: dict = {"node": node, "n_frames": n}

    if n == 0:
        return {**out, "sub_scores": {"completeness": 0.0}, "empty": True}

    acc = _accel_array(frames)
    miss = _nan_mask(acc)
    n_missing = int(miss.sum())
    valid = acc[~miss]

    channels = list(frames[0].meta.get("channels_present", []))
    has_gyro = frames[0].gyro_dps is not None

    # ---------------------------------------------- liveness / completeness --
    nan_fraction = n_missing / n
    nan_run_max = _longest_run(miss)
    # Terminal = the run reaches the final sample and never recovers. This is
    # what separates a flat battery from bursty radio loss.
    nan_terminal = bool(n_missing and miss[-1])
    trailing = 0
    for v in miss[::-1]:
        if not v:
            break
        trailing += 1

    rate = float(bw.meta.get("sampling_rate_hz")
                 or frames[0].meta.get("sampling_rate_hz") or 0.0)
    expected = int(round(rate * bw.duration_sec)) if rate else n
    out.update({
        "nan_fraction": nan_fraction,
        "nan_run_max": nan_run_max,
        "nan_terminal": nan_terminal,
        "nan_trailing_run": trailing,
        "frames_received": n,
        "frames_expected": expected,
        "frame_completeness": (n - n_missing) / expected if expected else 0.0,
    })

    # ------------------------------------------------------------ staleness --
    # Distinct samples over total. Rate degradation holds values, so this drops
    # sharply while nan_fraction stays at zero -- the pair separates the two.
    if len(valid):
        uniq = len({tuple(r) for r in valid})
        unique_value_ratio = uniq / len(valid)
        rep = 1
        max_rep = 1
        n_consecutive_dupes = 0
        for i in range(1, len(valid)):
            if np.array_equal(valid[i], valid[i - 1]):
                rep += 1
                max_rep = max(max_rep, rep)
                n_consecutive_dupes += 1
            else:
                rep = 1
        repeat_fraction = n_consecutive_dupes / max(1, len(valid) - 1)
        eff_hz = uniq / bw.duration_sec if bw.duration_sec > 0 else 0.0
    else:
        unique_value_ratio, max_rep, eff_hz, uniq = 0.0, 0, 0.0, 0
        repeat_fraction = 0.0
    out.update({
        "unique_values": uniq,
        "unique_value_ratio": unique_value_ratio,
        "max_repeat_run": max_rep,
        "repeat_fraction": repeat_fraction,
        "effective_unique_rate_hz": eff_hz,
    })

    # -------------------------------------------------- temporal integrity --
    t = np.array([f.t_sec for f in frames], dtype=float)
    dt = np.diff(t)
    monotonic = bool(np.all(dt > 0)) if len(dt) else True
    cv = float(np.std(dt) / np.mean(dt)) if len(dt) and np.mean(dt) > 0 else 0.0
    out.update({
        "timestamp_monotonic": monotonic,
        "inter_sample_interval_cv": cv,
        "cross_node_offset_ms": cross_node_offset_ms,
        # Needs paired device/host stamps -- Phase 9 hardware only.
        "clock_offset_ms": UNAVAILABLE,
        "skew_ppm": UNAVAILABLE,
    })

    # ----------------------------------------------------------- sanity ----
    if len(valid) > 1:
        axis_var = [float(np.var(valid[:, i])) for i in range(3)]
        mags = np.linalg.norm(valid, axis=1)
        mean_mag = float(np.mean(mags))
        sat = float(np.mean(np.abs(valid) >= FULL_SCALE_G * 0.99))
    else:
        axis_var, mean_mag, sat = [0.0, 0.0, 0.0], 0.0, 0.0
    plausible = PLAUSIBLE_MAG_MIN <= mean_mag <= PLAUSIBLE_MAG_MAX
    out.update({
        "axis_variance": axis_var,
        "total_variance": float(sum(axis_var)),
        "saturation_frac": sat,
        "mean_accel_magnitude_g": mean_mag,
        "magnitude_plausible": bool(plausible),
    })

    # --------------------------------------------------------- placement ----
    if len(valid):
        g = np.mean(valid, axis=0)
        norm = float(np.linalg.norm(g))
        gravity = (g / norm).tolist() if norm > 1e-9 else None
        # Stability of the gravity estimate: a jittery estimate makes any
        # angle comparison unreliable, so it is reported alongside.
        gnorms = np.linalg.norm(valid, axis=1)
        stability = float(1.0 / (1.0 + np.std(gnorms))) if len(gnorms) else 0.0
    else:
        gravity, stability, norm = None, 0.0, 0.0
    out.update({
        "gravity_vector": gravity,
        "gravity_magnitude_g": norm,
        "gravity_stability": stability,
    })

    # ------------------------------------------------------------- gyro ----
    if has_gyro and "gyro" in channels:
        gy = np.array([f.gyro_dps for f in frames], dtype=float)
        gmiss = np.isnan(gy).any(axis=1)
        gvalid = gy[~gmiss]
        out["gyro"] = {
            "available": True,
            "nan_fraction": float(gmiss.mean()),
            "mean_abs_dps": float(np.mean(np.abs(gvalid))) if len(gvalid) else 0.0,
            "variance": float(np.var(gvalid)) if len(gvalid) > 1 else 0.0,
        }
    else:
        # Never a default value, never a penalty: this node has no gyroscope.
        out["gyro"] = {"available": False, "reason": "no gyroscope on this node",
                       "nan_fraction": UNAVAILABLE, "mean_abs_dps": UNAVAILABLE,
                       "variance": UNAVAILABLE}
    out["channels_present"] = channels

    # -------------------------------------------------------- sub-scores ----
    # Each maps an observable to [0,1] where 1 is healthy.
    completeness = _clip01(1.0 - nan_fraction)

    # Keyed on CONSECUTIVE repeats, not raw uniqueness. A genuinely still limb
    # produces scattered duplicate values through sensor quantisation -- on
    # PAMAP2 a lying ankle shows unique_value_ratio ~0.82 with a max repeat run
    # of 1 -- and must not be called stale for holding still. A throttled node
    # holds the SAME value for consecutive samples, which quantisation never
    # does.
    staleness = _clip01(1.0 - repeat_fraction) if len(valid) > 1 else 1.0

    temporal = 1.0
    if not monotonic:
        temporal = 0.0
    elif cv > 0.5:
        temporal = _clip01(1.0 - (cv - 0.5))

    sanity = 1.0
    if not plausible:
        sanity = min(sanity, 0.3)
    if sat > 0.05:
        sanity = min(sanity, _clip01(1.0 - sat))
    if len(valid) > 1 and sum(axis_var) < 1e-9:
        sanity = min(sanity, 0.1)          # perfect flatline

    subs = {
        "completeness": completeness,
        "staleness": staleness,
        "temporal": temporal,
        "sanity": sanity,
    }
    # Cross-node synchrony. Without this the ladder can NAME a desync while the
    # node's state stays HEALTHY, so it is diagnosed but never flagged --
    # detection recall collapses while diagnosis looks perfect. Omitted, not
    # defaulted, when there is no second node to compare against.
    if isinstance(cross_node_offset_ms, (int, float)):
        off = abs(float(cross_node_offset_ms))
        subs["synchrony"] = _clip01(
            1.0 - max(0.0, off - SYNC_TOLERANCE_MS) / SYNC_SCALE_MS)
    out["sub_scores"] = subs
    return out


def window_signals(w) -> dict:
    """Signals for every node in the window, including cross-node timing."""
    bw = require_blind(w, "health.signals.window_signals")

    # Cross-node offset: each node's first timestamp against the median across
    # nodes. With a single node there is nothing to compare against.
    firsts = {n: fs[0].t_sec for n, fs in bw.frames.items() if fs}
    if len(firsts) >= 2:
        med = float(np.median(list(firsts.values())))
        offsets = {n: (t - med) * 1000.0 for n, t in firsts.items()}
    else:
        offsets = {n: NOT_APPLICABLE for n in bw.frames}

    return {
        "start_sec": bw.start_sec,
        "end_sec": bw.end_sec,
        "duration_sec": bw.duration_sec,
        "n_nodes": len(bw.frames),
        "cross_node_reference": "median of per-node first timestamps",
        "nodes": {n: node_signals(bw, n, offsets.get(n))
                  for n in bw.node_ids},
    }
