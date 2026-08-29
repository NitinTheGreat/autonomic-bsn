"""F4 — rate degradation: a throttled node reporting stale values.

Downsamples the target node to `target_hz`, then zero-order-holds back onto the
original time grid: the last real sample is REPEATED until the next arrives.
That is what a throttled node actually looks like to a consumer — the value is
genuinely stale, not smoothly interpolated. Interpolating would invent
intermediate measurements that never existed.

    severity 1..4 -> native/2, native/4, native/8, native/16

    PAMAP2  100 Hz -> 50, 25, 12.5, 6.25 Hz
    MHEALTH  50 Hz -> 25, 12.5, 6.25, 3.125 Hz

THE STRUCTURAL DIFFERENCE FROM PACKET LOSS
------------------------------------------
Rate degradation produces NO gaps. Every timestamp still carries a value; the
value is merely repeated. Packet loss produces NaN gaps.

Phase 4's health monitor must distinguish "stale but present" from "absent", so
Phase 3 has to generate them distinguishably. Asserted in tests: this injector
introduces ZERO NaNs.

Realised effect is the actual unique-value rate in Hz, counted as distinct
hold-groups per second — which can differ slightly from the requested rate when
the window boundary falls mid-hold.
"""

from __future__ import annotations

import math

import numpy as np

from core.datasource import NodeFrame
from injection.base import InjectorStrategy

DIVISORS = {1: 2, 2: 4, 3: 8, 4: 16}


class RateDegradationInjector(InjectorStrategy):
    name = "rate_degradation"
    required_channels = ("accel",)

    def params(self, severity: int) -> dict:
        return {"divisor": DIVISORS[severity],
                "unit": "native_hz / divisor -> target_hz"}

    def apply(self, frames: list[NodeFrame], severity: int,
              rng: np.random.Generator, ctx: dict) -> dict:
        div = DIVISORS[severity]
        native = float(ctx["sampling_rate_hz"])
        target_hz = native / div
        n = len(frames)
        if n == 0:
            return {"target_hz": target_hz, "realised_hz": 0.0, "n_holds": 0}

        period = 1.0 / target_hz
        t0 = frames[0].t_sec

        # Walk the window holding the last emitted sample until a full update
        # period has elapsed.
        held_accel = frames[0].accel_g
        held_gyro = frames[0].gyro_dps
        last_update_t = t0
        n_holds = 1

        for f in frames:
            if f.t_sec - last_update_t >= period - 1e-12:
                held_accel = f.accel_g
                held_gyro = f.gyro_dps
                last_update_t = f.t_sec
                n_holds += 1
            else:
                f.accel_g = held_accel
                # Preserve structural absence exactly: never turn None into a
                # tuple, never a tuple into None.
                if f.gyro_dps is not None and held_gyro is not None:
                    f.gyro_dps = held_gyro

        duration = frames[-1].t_sec - t0
        realised_hz = (n_holds / duration) if duration > 0 else 0.0

        return {
            "target_hz": target_hz,
            "realised_hz": float(realised_hz),
            "native_hz": native,
            "divisor": div,
            "n_holds": n_holds,
            "n_frames": n,
            "n_nan": 0,          # invariant: this injector never blanks
        }
