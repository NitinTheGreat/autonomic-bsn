"""F1 — battery exhaustion: the node stops reporting partway through a window.

After a fraction of the window has elapsed, the target node's accel and gyro
become NaN tuples for the remainder.

    severity 1..4 -> frac_remaining = 0.25, 0.50, 0.75, 1.00

so severity 4 blanks the entire window and severity 1 blanks only the last
quarter.

NaN, not zero: zero is a real accelerometer reading (stationary) that a model
may legitimately interpret as such. NaN says "no measurement".
NaN, not None: None is reserved for structurally absent hardware. A flat
battery does not remove the gyroscope from the board.
"""

from __future__ import annotations

import numpy as np

from core.datasource import NodeFrame
from injection.base import InjectorStrategy, blank_frame, nan_fraction

FRAC_REMAINING = {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}


class DropoutInjector(InjectorStrategy):
    name = "dropout"
    required_channels = ("accel",)

    def params(self, severity: int) -> dict:
        return {"frac_remaining": FRAC_REMAINING[severity],
                "unit": "fraction of window blanked"}

    def apply(self, frames: list[NodeFrame], severity: int,
              rng: np.random.Generator, ctx: dict) -> dict:
        frac = FRAC_REMAINING[severity]
        n = len(frames)
        if n == 0:
            return {"frac_nan": 0.0, "n_blanked": 0, "onset_sec": None}

        # Onset by WINDOW DURATION, not sample index: a throttled or lossy
        # window may not have a uniform sample count.
        t0, t1 = ctx["window_start"], ctx["window_end"]
        onset = t0 + (1.0 - frac) * (t1 - t0)

        blanked = 0
        for f in frames:
            if f.t_sec >= onset - 1e-12:
                blank_frame(f)
                blanked += 1

        return {
            "frac_nan": nan_fraction(frames),
            "n_blanked": blanked,
            "n_frames": n,
            "onset_sec": float(onset),
            "onset_rel_sec": float(onset - t0),
        }
