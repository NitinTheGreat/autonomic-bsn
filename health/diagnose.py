"""Diagnosis — an explicit, auditable ladder over observable signals.

Accepts ONLY a BlindWindow. No ground truth reaches this module; a test greps
this file for the ground-truth identifiers.

Aggregation: MIN, not a weighted sum
------------------------------------
A node's health is the MINIMUM of its sub-scores. An arithmetic mean would let
a node that is entirely offline (completeness 0) average back up to "fine" on
the strength of three unrelated sub-scores that still look healthy -- a silent
timestamp channel and an unremarkable variance say nothing about a node that
has stopped reporting. Health is a conjunction: a node is healthy only if
NOTHING is wrong with it. The geometric mean is recorded alongside as a softer
summary, but the state decision uses the min.

Displacement thresholds are ALIGNMENT-CONDITIONED
-------------------------------------------------
Phase 3 section 6.1 established that a rotation of theta about a node's long
axis changes the observed gravity direction by

    cos(observed) = cos^2(phi) + sin^2(phi) * cos(theta)

where phi is the angle between gravity and the rotation axis. The same
requested 15 degrees shows up as 14.89 deg on a lying ankle (|g.axis| = 0.12)
and 2.33 deg on a standing chest (|g.axis| = 0.99). A single global threshold
would therefore miss low-observability cases or false-positive on
high-observability ones.

The monitor cannot read `gravity_axis_alignment` -- that is ground truth and
BlindWindow strips it. But phi is INVARIANT under a rotation about the axis, so
the monitor can compute it from the observed gravity vector and the node's
known mounting axis, which is calibration config rather than ground truth about
this window. The detection threshold is then scaled by the expected
observability at that alignment.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from health.signals import NOT_APPLICABLE, UNAVAILABLE, window_signals
from health.window_view import require_blind

# --------------------------------------------------------------------------- #
# states and trust weights
# --------------------------------------------------------------------------- #
STATES = ("HEALTHY", "DEGRADED", "UNRELIABLE", "OFFLINE")

TRUST_WEIGHT = {"HEALTHY": 1.0, "DEGRADED": 0.6, "UNRELIABLE": 0.25,
                "OFFLINE": 0.0}

STATE_BOUNDS = [(0.9, "HEALTHY"), (0.6, "DEGRADED"), (0.2, "UNRELIABLE")]

DIAGNOSES = ("healthy", "dropout", "packet_loss", "rate_degradation",
             "clock_desync", "displacement")

# --------------------------------------------------------------------------- #
# ladder thresholds -- named so the paper can quote them
# --------------------------------------------------------------------------- #
TH_NAN_PRESENT = 0.02          # any meaningful missingness
TH_DROPOUT_TRAILING = 0.10     # trailing NaN run as a fraction of the window
TH_STALE_UNIQUE_RATIO = 0.75   # below this, values are being held
TH_STALE_REPEAT_RUN = 3        # consecutive identical samples
# Fraction of samples equal to their predecessor. This is the discriminator
# that works at the mildest throttling: holding every 2nd sample gives a max
# repeat RUN of only 2 (below the run threshold) but a repeat FRACTION of ~0.5.
# Sensor quantisation on a still limb produces scattered duplicates with a run
# length of 1 and a repeat fraction near 0, so the two do not collide.
TH_REPEAT_FRACTION = 0.20
TH_CROSS_NODE_OFFSET_MS = 25.0 # beyond this a node is out of step
TH_DISPLACEMENT_FLOOR_DEG = 4.0    # noise floor for any angle claim
TH_DISPLACEMENT_FRACTION = 0.5     # fraction of the smallest detectable effect
SMALLEST_TARGET_THETA_DEG = 15.0   # the smallest rotation worth catching

# Mounting axis per node -- CALIBRATION CONFIG, not ground truth about a
# window. Index into (x, y, z); the limb's long axis, the direction a strap
# lets a sensor twist around.
NODE_ROTATION_AXIS = {"wrist": 0, "ankle": 0, "chest": 1}


def expected_observable_angle(theta_deg: float, phi_deg: float) -> float:
    """Observable gravity-direction change for rotation theta at alignment phi."""
    th, ph = math.radians(theta_deg), math.radians(phi_deg)
    c = math.cos(ph) ** 2 + math.sin(ph) ** 2 * math.cos(th)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def angle_between(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return math.degrees(math.acos(
        float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def state_for(min_score: float) -> str:
    for bound, name in STATE_BOUNDS:
        if min_score > bound:
            return name
    return "OFFLINE"


def _geometric_mean(vals) -> float:
    vals = [max(v, 1e-9) for v in vals]
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


# --------------------------------------------------------------------------- #
# displacement
# --------------------------------------------------------------------------- #
def displacement_assessment(sig: dict, node: str,
                            reference) -> dict:
    """Angle vs the calibrated reference, with an observability-scaled threshold.

    `reference` is either a bare 3-vector or a calibration dict carrying the
    reference direction AND the NATURAL SPREAD of that direction across clean
    held-out windows.

    The spread term matters more than the reference itself. A body-worn node's
    gravity direction moves with posture: an ankle points very differently when
    lying than when standing. Measured on PAMAP2, that natural variation is
    far larger than the change a 15-90 degree sensor rotation produces, so a
    threshold set without reference to it flags almost every clean window.
    Calibrating the spread on held-out clean data is still blind -- it says
    nothing about whether THIS window was perturbed.
    """
    g = sig.get("gravity_vector")
    ref_vec, natural_spread = None, None
    if isinstance(reference, dict):
        ref_vec = reference.get("vector")
        natural_spread = reference.get("clean_angle_p95")
    elif reference is not None:
        ref_vec = list(reference)

    if g is None or ref_vec is None:
        return {"available": False,
                "reason": ("no reference gravity for this (node, dataset)"
                           if g is not None else "no valid samples"),
                "angle_deg": None, "threshold_deg": None, "exceeded": False}

    angle = angle_between(g, ref_vec)
    reference_gravity = ref_vec

    # phi is invariant under rotation about the axis, so it is observable.
    axis = np.zeros(3)
    axis[NODE_ROTATION_AXIS.get(node, 0)] = 1.0
    alignment = abs(float(np.dot(np.asarray(g, float)
                                 / np.linalg.norm(g), axis)))
    phi_deg = math.degrees(math.acos(max(-1.0, min(1.0, alignment))))

    # The smallest effect worth catching at this alignment, halved so a genuine
    # smallest-step rotation clears the bar, floored against sensor noise.
    detectable = expected_observable_angle(SMALLEST_TARGET_THETA_DEG, phi_deg)
    threshold = max(TH_DISPLACEMENT_FLOOR_DEG,
                    TH_DISPLACEMENT_FRACTION * detectable)

    # A claim of displacement must clear the node's own natural posture-driven
    # variation, or every clean window trips it. Where that spread exceeds what
    # the rotation itself can produce, displacement is simply NOT DETECTABLE
    # from gravity direction alone -- reported, not hidden.
    swamped = False
    if natural_spread is not None:
        if natural_spread > threshold:
            swamped = True
        threshold = max(threshold, natural_spread)

    return {
        "available": True,
        "angle_deg": angle,
        "reference_gravity": list(reference_gravity),
        "observed_gravity": list(g),
        "estimated_alignment": alignment,      # |g . axis|, observable
        "estimated_phi_deg": phi_deg,
        "expected_min_detectable_deg": detectable,
        "natural_spread_deg": natural_spread,
        "threshold_deg": threshold,
        "swamped_by_natural_variation": swamped,
        "exceeded": angle > threshold,
        "observability": "low" if alignment > 0.9 else
                         ("medium" if alignment > 0.6 else "high"),
    }


# --------------------------------------------------------------------------- #
# the ladder
# --------------------------------------------------------------------------- #
def diagnose_node(sig: dict, node: str,
                  reference_gravity: Optional[list] = None) -> dict:
    """Explicit if/elif ladder. Returns a verdict AND the evidence behind it."""
    subs = dict(sig.get("sub_scores", {}))
    nan_frac = sig.get("nan_fraction", 0.0)
    trailing_frac = (sig.get("nan_trailing_run", 0) / sig["n_frames"]
                     if sig.get("n_frames") else 0.0)
    uniq_ratio = sig.get("unique_value_ratio", 1.0)
    repeat_run = sig.get("max_repeat_run", 1)
    repeat_frac = sig.get("repeat_fraction", 0.0)
    offset = sig.get("cross_node_offset_ms")
    offset_abs = (abs(offset) if isinstance(offset, (int, float))
                  else None)

    disp = displacement_assessment(sig, node, reference_gravity)

    # Displacement contributes a placement sub-score only where it is
    # assessable; a node with no reference is not penalised for that.
    if disp["available"]:
        subs["placement"] = 0.0 if disp["exceeded"] else 1.0
    sub_vals = list(subs.values()) or [1.0]
    min_score = float(min(sub_vals))
    state = state_for(min_score)

    ev: dict = {}
    # --- the ladder ------------------------------------------------------- #
    if nan_frac > TH_NAN_PRESENT and sig.get("nan_terminal") \
            and trailing_frac > TH_DROPOUT_TRAILING:
        dx = "dropout"
        ev = {"rule": "terminal NaN run reaching the end of the window",
              "nan_fraction": nan_frac,
              "nan_terminal": True,
              "trailing_run_fraction": trailing_frac}
    elif nan_frac > TH_NAN_PRESENT:
        dx = "packet_loss"
        ev = {"rule": "NaN present but not terminal -- bursty loss",
              "nan_fraction": nan_frac,
              "nan_run_max": sig.get("nan_run_max"),
              "nan_terminal": bool(sig.get("nan_terminal")),
              "trailing_run_fraction": trailing_frac}
    elif (repeat_frac >= TH_REPEAT_FRACTION
          or (uniq_ratio < TH_STALE_UNIQUE_RATIO
              and repeat_run >= TH_STALE_REPEAT_RUN)):
        dx = "rate_degradation"
        ev = {"rule": "no NaN, but consecutive samples repeat -- values held",
              "nan_fraction": nan_frac,
              "repeat_fraction": repeat_frac,
              "unique_value_ratio": uniq_ratio,
              "max_repeat_run": repeat_run,
              "effective_unique_rate_hz": sig.get("effective_unique_rate_hz")}
    elif offset_abs is not None and offset_abs > TH_CROSS_NODE_OFFSET_MS:
        dx = "clock_desync"
        ev = {"rule": "values intact, timestamps out of step with other nodes",
              "cross_node_offset_ms": offset,
              "threshold_ms": TH_CROSS_NODE_OFFSET_MS}
    elif disp["available"] and disp["exceeded"]:
        dx = "displacement"
        ev = {"rule": "gravity direction moved beyond the alignment-scaled "
                      "threshold",
              "angle_deg": disp["angle_deg"],
              "threshold_deg": disp["threshold_deg"],
              "estimated_alignment": disp["estimated_alignment"],
              "observability": disp["observability"]}
    else:
        dx = "healthy"
        ev = {"rule": "no rule fired",
              "nan_fraction": nan_frac,
              "repeat_fraction": repeat_frac,
              "unique_value_ratio": uniq_ratio,
              "cross_node_offset_ms": offset,
              "gravity_angle_deg": disp.get("angle_deg")}

    # A node the ladder calls healthy but whose sub-scores are poor is still
    # flagged: the state and the diagnosis are separate answers.
    return {
        "node": node,
        "diagnosis": dx,
        "evidence": ev,
        "state": state,
        "trust_weight": TRUST_WEIGHT[state],
        "min_sub_score": min_score,
        "geometric_mean_sub_score": _geometric_mean(sub_vals),
        "sub_scores": subs,
        "aggregation": "min (health is a conjunction, not an average)",
        "displacement": disp,
        "gyro_available": sig.get("gyro", {}).get("available", False),
        "channels_present": sig.get("channels_present", []),
    }


def diagnose_window(w, references: Optional[dict] = None) -> dict:
    """Full monitor output for one window. BlindWindow only."""
    bw = require_blind(w, "health.diagnose.diagnose_window")
    sigs = window_signals(bw)
    refs = references or {}

    nodes = {}
    for node, sig in sigs["nodes"].items():
        nodes[node] = diagnose_node(sig, node, refs.get(node))
        nodes[node]["signals"] = sig

    worst = min((n["min_sub_score"] for n in nodes.values()), default=1.0)
    flagged = [n for n, v in nodes.items() if v["state"] != "HEALTHY"]
    return {
        "start_sec": sigs["start_sec"],
        "end_sec": sigs["end_sec"],
        "nodes": nodes,
        "flagged_nodes": flagged,
        "worst_min_sub_score": worst,
        "network_state": state_for(worst),
    }
