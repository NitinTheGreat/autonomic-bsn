"""The DataSource contract — the seam every later phase plugs into.

This module defines the ONLY interface between data production and everything
downstream. Phases 3-9 depend on its shape being stable, and Phase 9's
HardwareLiveSource (ESP32 nodes over BLE) must satisfy it unchanged. Do not
alter these dataclasses or the Protocol without revisiting every consumer.

Two design points carry real experimental weight
------------------------------------------------

1. ABSENT SENSOR vs ZERO READING.
   `gyro_dps` is Optional and is None when a node has no gyroscope at all --
   MHEALTH's chest node carries accelerometer + ECG only. It is NEVER a zero
   tuple. A zero gyro reading is a genuine physical measurement meaning "not
   rotating"; an absent gyro means "no such sensor exists here". Zero-filling
   would fabricate a "sensor reads zero" signal in exactly the degradation
   study this project runs, and would make a healthy node indistinguishable
   from a stuck one.

   Every NodeFrame additionally carries `channels_present` in its meta, so
   downstream code branches on declared availability rather than inferring it
   from None-ness at each use site.

   Note the third case: a node that HAS a gyroscope but dropped a sample
   carries a tuple containing NaN, not None. Sensor present, reading missing.

2. injected_failure IS WRITE-ONCE, READ-ONCE, AND NOT BY THE MONITOR.
   It is written ONLY by Phase 3's failure injector and read ONLY by Phase 4's
   separate scoring module, which compares detections against ground truth.

   *** The Phase 4 health monitor must NEVER read injected_failure. ***

   If the monitor can see the ground-truth label it is supposed to infer, its
   detection metrics become circular and meaningless — it would be scored on
   reading an answer key rather than on detecting degradation from signal.
   Throughout Phase 2 this field stays None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal, Optional, Protocol, runtime_checkable

# Canonical node ids. Datasets use varying names (PAMAP2 calls the wrist node
# "hand"); loaders normalise to these so downstream code is dataset-agnostic.
CANONICAL_NODE_IDS = ["wrist", "chest", "ankle"]

# Channel names allowed in meta["channels_present"].
CHANNEL_ACCEL = "accel"
CHANNEL_GYRO = "gyro"
CHANNEL_MAG = "mag"
CHANNEL_ECG = "ecg"


@dataclass
class NodeFrame:
    node_id: str                    # canonical: "wrist" | "chest" | "ankle"
    t_sec: float                    # seconds since session start, monotonic
    accel_g: tuple[float, float, float]
    gyro_dps: Optional[tuple[float, float, float]]   # None when absent
    source: Literal["dataset", "hardware"]
    injected_failure: Optional[str] = None
    meta: dict = field(default_factory=dict)


@dataclass
class Window:
    start_sec: float
    end_sec: float
    frames: dict[str, list[NodeFrame]]
    label: Optional[str]


@runtime_checkable
class DataSource(Protocol):
    node_ids: list[str]
    sampling_rate_hz: float

    def windows(self, window_sec: float = 2.56,
                overlap: float = 0.5) -> Iterator[Window]: ...


# --------------------------------------------------------------------------- #
# Unit conventions -- fixed here so datasets and hardware agree downstream.
#
# The ESP32 nodes planned for Phase 9 report acceleration in g and angular rate
# in deg/s natively. Normalising datasets to those units (rather than to SI)
# means the hardware source needs no conversion layer, and every consumer sees
# one unit system regardless of where the frame came from.
# --------------------------------------------------------------------------- #
STANDARD_GRAVITY = 9.80665          # m/s^2 per g (CODATA / ISO 80000-3)


def ms2_to_g(v: float) -> float:
    """m/s^2 -> g."""
    return v / STANDARD_GRAVITY


def rad_s_to_dps(v: float) -> float:
    """rad/s -> deg/s."""
    import math
    return v * 180.0 / math.pi
