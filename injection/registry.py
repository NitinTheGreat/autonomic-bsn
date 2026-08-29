"""Registry of failure types.

Phase 6's sweep iterates this dict, so adding a sixth failure type later
requires NO change to the sweep code -- register the class here and it appears
in every sweep, verification run and frontend dropdown automatically.
"""

from __future__ import annotations

from injection.base import SEVERITIES, FailureInjector, InjectorStrategy
from injection.clock_desync import ClockDesyncInjector
from injection.displacement import DisplacementInjector
from injection.dropout import DropoutInjector
from injection.packet_loss import PacketLossInjector
from injection.rate_degradation import RateDegradationInjector

STRATEGIES: dict[str, type[InjectorStrategy]] = {
    "dropout": DropoutInjector,                    # F1 battery exhaustion
    "clock_desync": ClockDesyncInjector,           # F2 timestamp drift
    "packet_loss": PacketLossInjector,             # F3 bursty BLE loss
    "rate_degradation": RateDegradationInjector,   # F4 throttling
    "displacement": DisplacementInjector,          # F5 sensor rotation
}

FAILURE_TYPES = list(STRATEGIES)

# Human-readable one-liners, reused by the CLI output and the frontend.
DESCRIPTIONS = {
    "dropout": "Battery exhaustion -- node stops reporting mid-window (NaN).",
    "clock_desync": "Timestamps drift out of alignment with the other nodes.",
    "packet_loss": "Bursty BLE loss, Gilbert-Elliott two-state Markov chain.",
    "rate_degradation": "Throttled node holding stale values (no gaps).",
    "displacement": "Sensor rotated on the limb -- reference frame changes.",
}


def make_injector(base, failure_type: str, severity: int, target_node: str,
                  seed: int) -> FailureInjector:
    """Factory. Raises InjectionError for an unknown type or bad severity."""
    return FailureInjector(base, failure_type, severity, target_node, seed)


__all__ = ["STRATEGIES", "FAILURE_TYPES", "DESCRIPTIONS", "SEVERITIES",
           "make_injector", "FailureInjector"]
