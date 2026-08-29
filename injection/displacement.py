"""F5 — sensor displacement: the node has rotated on the limb.

Applies a 3D rotation R(theta) about a fixed axis to the node's accel AND gyro
vectors jointly. Both must rotate: they share the sensor's body frame, so
rotating only the accelerometer would describe a physically impossible device
whose two sensors disagree about which way is up.

    severity 1..4 -> theta = 15, 30, 45, 90 degrees

Rotation axis
-------------
Default is the limb's long axis, the direction a strap actually lets a sensor
twist around. Per placement, as an axis index into (x, y, z):

    wrist -> x   forearm long axis
    ankle -> x   shank long axis
    chest -> y   torso vertical axis

These follow the datasets' documented sensor orientations, but the exact
mounting rotation of each unit is not published per subject, so treat the axis
choice as a stated assumption rather than a measured fact. It is configurable.

LIMITATION — state this in any paper text, before a reviewer does
------------------------------------------------------------------
This changes the sensor's REFERENCE FRAME but not the underlying motion
dynamics. Real strap slippage alters both: a loose sensor also damps, lags and
adds impact transients that a pure rotation cannot reproduce.

This is therefore a defensible PROXY for displacement, not a reproduction of
it. Results should be read as "the model's sensitivity to a change of sensor
frame", which is a real and relevant failure mode, rather than as "the model's
sensitivity to a sensor coming loose".

Realised effect
---------------
The actual angle between the pre- and post-rotation MEAN GRAVITY vectors. Note
this is not always equal to theta: rotating about an axis that happens to be
parallel to the mean gravity vector leaves gravity unchanged, so the realised
angle is near zero even though the frame genuinely rotated. Reporting the
realised value is the point -- see the Phase 3 walkthrough.
"""

from __future__ import annotations

import math

import numpy as np

from core.datasource import NodeFrame
from injection.base import InjectorStrategy, is_nan_triple

THETA_DEG = {1: 15.0, 2: 30.0, 3: 45.0, 4: 90.0}

# Axis index into (x, y, z) per node placement -- the limb's long axis.
DEFAULT_AXIS = {"wrist": 0, "ankle": 0, "chest": 1}
AXIS_NAMES = {0: "x", 1: "y", 2: "z"}


def rotation_matrix(axis_idx: int, theta_deg: float) -> np.ndarray:
    """Right-handed rotation about a principal axis."""
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    if axis_idx == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis_idx == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _mean_vec(vals: list) -> np.ndarray | None:
    good = [v for v in vals if v is not None and not is_nan_triple(v)]
    return np.mean(np.array(good, dtype=float), axis=0) if good else None


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cos))


class DisplacementInjector(InjectorStrategy):
    name = "displacement"
    # Only accel is REQUIRED: a node without a gyro can still be displaced.
    required_channels = ("accel",)

    def __init__(self, axis_override: int | None = None):
        self.axis_override = axis_override

    def params(self, severity: int) -> dict:
        return {"theta_deg": THETA_DEG[severity],
                "axis": "limb long axis (per-node default)",
                "unit": "degrees of sensor-frame rotation"}

    def apply(self, frames: list[NodeFrame], severity: int,
              rng: np.random.Generator, ctx: dict) -> dict:
        theta = THETA_DEG[severity]
        node = ctx["target_node"]
        axis = (self.axis_override if self.axis_override is not None
                else DEFAULT_AXIS.get(node, 0))
        R = rotation_matrix(axis, theta)

        if not frames:
            return {"theta_deg": theta, "realised_angle_deg": 0.0}

        before = _mean_vec([f.accel_g for f in frames])
        # A node with no gyroscope is the ONE case where partial application is
        # correct rather than an error: there is genuinely no gyro to rotate.
        gyro_rotated = frames[0].gyro_dps is not None

        for f in frames:
            if not is_nan_triple(f.accel_g):
                f.accel_g = tuple(float(x) for x in R @ np.array(f.accel_g))
            if f.gyro_dps is not None and not is_nan_triple(f.gyro_dps):
                f.gyro_dps = tuple(float(x) for x in R @ np.array(f.gyro_dps))

        after = _mean_vec([f.accel_g for f in frames])
        realised = (_angle_between(before, after)
                    if before is not None and after is not None else 0.0)

        # How closely gravity lies along the rotation axis. This FULLY explains
        # the gap between theta and the realised angle:
        #     cos(realised) = cos^2(phi) + sin^2(phi) * cos(theta)
        # where phi is the angle between gravity and the axis. At phi = 90 deg
        # the realised angle equals theta; at phi = 0 it is zero, because
        # spinning a vector about itself changes nothing.
        #
        # Measured on PAMAP2 subject101, a requested 15 deg realises as 14.9 deg
        # on the ankle while lying (|g.axis| = 0.12) but only 2.3 deg on the
        # chest while standing (|g.axis| = 0.99). The rotation is applied at
        # exactly theta in both cases -- what differs is how much of it is
        # OBSERVABLE in the gravity direction, which is what actually reaches a
        # classifier. Phase 6 must therefore not treat displacement severity as
        # a fixed physical magnitude across postures; compare within an
        # activity, or condition on this alignment.
        alignment = None
        if before is not None:
            nb = np.linalg.norm(before)
            if nb > 1e-12:
                ax_vec = np.zeros(3)
                ax_vec[axis] = 1.0
                alignment = abs(float(np.dot(before / nb, ax_vec)))

        return {
            "theta_deg": theta,
            "realised_angle_deg": float(realised),
            "gravity_axis_alignment": alignment,
            "axis_index": axis,
            "axis_name": AXIS_NAMES[axis],
            "gyro_rotated": gyro_rotated,
            "n_frames": len(frames),
            "note": ("realised angle is the observable change in the mean "
                     "gravity direction; it equals theta only when gravity is "
                     "perpendicular to the rotation axis, and falls toward 0 as "
                     "gravity_axis_alignment approaches 1"),
        }
