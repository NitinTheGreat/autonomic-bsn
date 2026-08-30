"""Window -> feature extraction for the S1 agent.

Consumes a **BlindWindow only**. Phase 4's constraint applies with full force
here: if the agent's prompt is built from ground-truth-bearing data, the agent
inherits the exact channel the health monitor was denied, and every downstream
confidence number becomes circular.

Feature set (richer than Gate 2's 45)
-------------------------------------
per accel axis          mean, std, min, max, energy
per gyro axis           the same five, WHERE THE NODE HAS A GYROSCOPE
cross-axis correlation  accel x-y, y-z, x-z
orientation             unit mean-acceleration vector + magnitude, and a motion
                        level -- carried forward from the Gate 2 V1 ablation,
                        where they were worth +0.042 and earned their place

Three rules that are not negotiable
-----------------------------------
1. **Never zero-fill.** A zero gyro feature is a real measurement meaning "not
   rotating"; an absent gyro means "no such sensor". They are different facts
   and must not collapse.

2. **Never fabricate a value for missing data.** An earlier version of this
   renderer emitted `ORIENTATION (+0.00, +0.00, +0.00)` and `MOTION (vigorous)`
   for a node whose samples were entirely NaN -- describing a dead node as
   vigorously moving with a zero orientation. That is invented evidence handed
   straight to the model, and it plausibly explains why confidence failed to
   drop under injection. Missing data is now reported AS missing.

3. **NaN-aware, never zero-imputed.** Statistics are computed over the non-NaN
   samples. Imputing zeros first would replace "no measurement" with "not
   moving" and destroy the degradation signal Phases 3 and 4 exist to create.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from health.window_view import BlindWindow, require_blind

AXES = ("x", "y", "z")
STATS = ("mean", "std", "min", "max", "energy")

# Motion-level bands, in g of per-axis standard deviation.
MOTION_STILL = 0.08
MOTION_MODERATE = 0.35

# A node with fewer than this fraction of valid samples is reported as having
# no usable data rather than having its statistics computed from a handful.
MIN_VALID_FRACTION = 0.20

NO_DATA = "no_data"


def _axis_stats(v: np.ndarray) -> Optional[dict]:
    """Five statistics over the valid samples, or None if there are none."""
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return None
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "energy": float(np.mean(v ** 2)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Pearson correlation over samples where BOTH axes are valid."""
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return None
    x, y = a[m], b[m]
    sx, sy = np.std(x), np.std(y)
    if sx < 1e-12 or sy < 1e-12:
        return None            # constant axis: correlation is undefined
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / (sx * sy))


def node_features(bw: BlindWindow, node: str) -> dict:
    """Every feature that EXISTS for this node. Absent things stay absent."""
    frames = bw.frames[node]
    out: dict = {"node": node, "n_frames": len(frames)}
    if not frames:
        return {**out, "has_data": False, "reason": "no frames",
                "channels_present": []}

    channels = list(frames[0].meta.get("channels_present", []))
    out["channels_present"] = channels

    acc = np.array([f.accel_g for f in frames], dtype=float)
    valid = ~np.isnan(acc).any(axis=1)
    frac_valid = float(valid.mean())
    out["valid_fraction"] = frac_valid
    out["nan_fraction"] = 1.0 - frac_valid

    # Report a dead or near-dead node as having no data. Do NOT compute
    # statistics from a residue of samples and present them as the node's state.
    if frac_valid < MIN_VALID_FRACTION:
        return {**out, "has_data": False,
                "reason": "node reported no usable data (%.0f%% of samples "
                          "missing)" % (100 * (1 - frac_valid))}

    out["has_data"] = True

    # ---- accelerometer ---------------------------------------------------- #
    out["accel"] = {}
    for i, a in enumerate(AXES):
        s = _axis_stats(acc[:, i])
        if s is not None:
            out["accel"][a] = s

    # ---- cross-axis correlation ------------------------------------------- #
    corr = {}
    for (i, j), name in ((0, 1), "xy"), ((1, 2), "yz"), ((0, 2), "xz"):
        c = _corr(acc[:, i], acc[:, j])
        if c is not None:
            corr[name] = c
    if corr:
        out["accel_corr"] = corr

    # ---- orientation + motion (the Gate 2 V1 features) -------------------- #
    av = acc[valid]
    means = np.mean(av, axis=0)
    mag = float(np.linalg.norm(means))
    if mag > 1e-9 and not math.isnan(mag):
        out["orientation"] = {
            "unit": [float(x) for x in (means / mag)],
            "magnitude_g": mag,
        }
    # else: omitted entirely. Never a zero vector -- that would read as a real
    # measurement of zero rather than as an absence.

    stds = np.std(av, axis=0)
    motion = float(np.mean(stds))
    if not math.isnan(motion):
        out["motion"] = {
            "mean_axis_sd_g": motion,
            "level": ("still" if motion < MOTION_STILL else
                      "moderate" if motion < MOTION_MODERATE else "vigorous"),
        }

    # ---- gyroscope, only where the node actually has one ------------------ #
    has_gyro = frames[0].gyro_dps is not None and "gyro" in channels
    if has_gyro:
        gy = np.array([f.gyro_dps for f in frames], dtype=float)
        g_out = {}
        for i, a in enumerate(AXES):
            s = _axis_stats(gy[:, i])
            if s is not None:
                g_out[a] = s
        gvalid = float((~np.isnan(gy).any(axis=1)).mean())
        if g_out:
            out["gyro"] = g_out
            out["gyro_valid_fraction"] = gvalid
    else:
        # Structural absence, stated -- not a zero, not a default.
        out["gyro_absent_reason"] = "this node has no gyroscope"
    return out


def window_features(w) -> dict:
    """Features for every node. BlindWindow only."""
    bw = require_blind(w, "features.extractors.window_features")
    return {
        "start_sec": bw.start_sec,
        "end_sec": bw.end_sec,
        "duration_sec": bw.duration_sec,
        "nodes": {n: node_features(bw, n) for n in bw.node_ids},
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_node(f: dict) -> list[str]:
    """One node as prompt lines. Missing things are SAID to be missing."""
    name = f["node"]
    if not f.get("has_data"):
        return ["%-6s NO DATA -- %s" % (name, f.get("reason", "unavailable"))]

    lines = []
    o = f.get("orientation")
    if o:
        lines.append("%-6s ORIENTATION (%+.2f, %+.2f, %+.2f)  magnitude %.2f g"
                     % (name, o["unit"][0], o["unit"][1], o["unit"][2],
                        o["magnitude_g"]))
    else:
        lines.append("%-6s ORIENTATION unavailable" % name)

    m = f.get("motion")
    lines.append("       MOTION      %.3f g  (%s)"
                 % (m["mean_axis_sd_g"], m["level"]) if m
                 else "       MOTION      unavailable")

    if f.get("nan_fraction", 0) > 0.01:
        lines.append("       MISSING     %.0f%% of samples absent"
                     % (100 * f["nan_fraction"]))

    acc = f.get("accel", {})
    if acc:
        det = " | ".join(
            "%s mean %+.2f std %.2f min %+.2f max %+.2f"
            % (a, acc[a]["mean"], acc[a]["std"], acc[a]["min"], acc[a]["max"])
            for a in AXES if a in acc)
        lines.append("       ACCEL       " + det)

    c = f.get("accel_corr")
    if c:
        lines.append("       CORR        " + " ".join(
            "%s %+.2f" % (k, v) for k, v in c.items()))

    g = f.get("gyro")
    if g:
        det = " | ".join(
            "%s mean %+.1f std %.1f" % (a, g[a]["mean"], g[a]["std"])
            for a in AXES if a in g)
        lines.append("       GYRO        " + det + " (deg/s)")
    elif f.get("gyro_absent_reason"):
        lines.append("       GYRO        none on this node")
    return lines


def render(features: dict) -> str:
    out = []
    for node in ("wrist", "chest", "ankle"):
        f = features["nodes"].get(node)
        if f:
            out.extend(render_node(f))
    return "\n".join(out)


def feature_count(features: dict) -> int:
    """How many scalar features actually exist (absent ones are not counted)."""
    n = 0
    for f in features["nodes"].values():
        for key in ("accel", "gyro"):
            for ax in (f.get(key) or {}).values():
                n += len(ax)
        n += len(f.get("accel_corr") or {})
        if f.get("orientation"):
            n += 4
        if f.get("motion"):
            n += 1
    return n
