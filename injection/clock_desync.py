"""F2 — clock desync: the node's timestamps drift out of alignment.

Shifts the target node's frame timestamps by a constant offset, so its samples
misalign against the other nodes' while the sensor values themselves are
untouched. Optional linear skew models a drifting oscillator.

    severity 1..4 -> delta_ms = 50, 200, 500, 2000

Skew is implemented but DISABLED by default (skew_ppm = 0.0). Over a 2.56 s
window even a large 200 ppm skew contributes only ~0.5 ms of additional drift,
which is negligible beside a 50-2000 ms offset. It is exposed because Phase 6
may run longer sessions where accumulated skew dominates; there the drift grows
as skew_ppm * 1e-6 * elapsed_seconds.

DERIVED vs MEASURED CLOCKS -- do not pool these silently
--------------------------------------------------------
MHEALTH ships no timestamp column: Phase 2 derives t_sec from the row index at
50 Hz and sets meta["timestamp_derived"] = True. PAMAP2's timestamps were
actually recorded.

Perturbing a perfectly regular synthetic clock is NOT the same experiment as
perturbing a real one, which already carries jitter, scheduling noise and
genuine gaps. Desync results from MHEALTH therefore are not comparable to
PAMAP2's.

This injector reads that flag, records `clock_source: "derived" | "measured"`
in the window metadata, and prints a one-time INFO when injecting into a
derived clock. It does NOT refuse -- the experiment is still meaningful on its
own terms. It simply makes the difference impossible to overlook in Phase 6.
"""

from __future__ import annotations

import numpy as np

from core.datasource import NodeFrame
from injection.base import InjectorStrategy

DELTA_MS = {1: 50.0, 2: 200.0, 3: 500.0, 4: 2000.0}

# Linear oscillator skew, parts per million. 0 disables it; see docstring.
DEFAULT_SKEW_PPM = 0.0


class ClockDesyncInjector(InjectorStrategy):
    name = "clock_desync"
    required_channels = ("accel",)     # timestamps exist for any node

    def __init__(self, skew_ppm: float = DEFAULT_SKEW_PPM):
        self.skew_ppm = float(skew_ppm)

    def params(self, severity: int) -> dict:
        return {"delta_ms": DELTA_MS[severity], "skew_ppm": self.skew_ppm,
                "unit": "milliseconds of timestamp offset"}

    def apply(self, frames: list[NodeFrame], severity: int,
              rng: np.random.Generator, ctx: dict) -> dict:
        delta_s = DELTA_MS[severity] / 1000.0
        if not frames:
            return {"delta_ms": 0.0, "clock_source": "unknown"}

        derived = bool(frames[0].meta.get("timestamp_derived", False))
        clock_source = "derived" if derived else "measured"

        if derived:
            key = "clock_desync_derived_notice"
            notices = ctx.get("notices")
            if notices is not None and key not in notices:
                notices.add(key)
                print("  INFO: injecting clock_desync into a DERIVED clock "
                      "(timestamps synthesised from row index). Real jitter is "
                      "absent, so these results are not comparable to a "
                      "measured-clock dataset such as PAMAP2.")

        t0 = frames[0].t_sec
        before = [f.t_sec for f in frames]
        for f in frames:
            elapsed = f.t_sec - t0
            f.t_sec = f.t_sec + delta_s + elapsed * self.skew_ppm * 1e-6
            f.meta = dict(f.meta)
            f.meta["clock_source"] = clock_source
            f.meta["clock_offset_applied_ms"] = DELTA_MS[severity]

        after = [f.t_sec for f in frames]
        realised = float(np.mean(np.array(after) - np.array(before)) * 1000.0)
        return {
            "delta_ms": realised,
            "requested_delta_ms": DELTA_MS[severity],
            "skew_ppm": self.skew_ppm,
            "clock_source": clock_source,
            "timestamp_derived": derived,
            "n_frames": len(frames),
        }
