"""FailureInjector — a DataSource DECORATOR implementing the failure taxonomy.

A FailureInjector wraps any DataSource and is itself a valid DataSource, so it
composes with DatasetReplaySource today and with Phase 9's HardwareLiveSource
unchanged — and injectors compose with each other.

    src = DatasetReplaySource("pamap2", label_set="PAMAP2_8")
    bad = FailureInjector(src, "dropout", severity=3, target_node="ankle",
                          seed=7)
    for w in bad.windows():          # same contract as src.windows()
        ...

*** Phase 4's health monitor must NEVER read NodeFrame.injected_failure. ***

That field exists solely so a SEPARATE scoring module can grade the monitor
afterwards. If the monitor can see the ground-truth label it is supposed to
infer, its detection metrics measure nothing but its ability to read an answer
key.

The three-state distinction from Phase 2 must survive injection
---------------------------------------------------------------
    has sensor, sample valid    -> (x, y, z)
    has sensor, sample dropped  -> tuple of NaN     <- ALL injectors use this
    has no sensor at all        -> None             <- structural, never injected

**No injector may ever write None.** A dead node still HAS its sensors; it
stopped reporting. Writing None would make an injected failure
indistinguishable from MHEALTH's structurally absent chest gyro, collapsing the
distinction Phase 2 was built to preserve. Asserted in tests.

NaN rather than zero, equally deliberately: zero is a real, physically
meaningful accelerometer reading (stationary), and a model may reasonably read
it as such. NaN says "no measurement", which is what actually happened.

Reproducibility
---------------
Every stochastic injector receives an explicit seed and uses its OWN
numpy Generator instance. Global numpy random state is never touched, because
that would make results depend on call order across Phase 6's sweep — two runs
of the same configuration could differ purely because of what ran before them.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from typing import Iterator, Optional

import numpy as np

from core.datasource import NodeFrame, Window

NAN3 = (float("nan"), float("nan"), float("nan"))

SEVERITIES = (1, 2, 3, 4)


class InjectionError(RuntimeError):
    """Raised when an injection cannot be applied as requested.

    Always loud. A silent no-op would mean the requested injection rate does
    not match the realised one, and Phase 6's severity axis would be quietly
    wrong for every point on the curve.
    """


def is_nan_triple(v: Optional[tuple]) -> bool:
    return v is not None and all(isinstance(c, float) and math.isnan(c)
                                 for c in v)


class InjectorStrategy(ABC):
    """One failure mode. Mutates a single node's frame list in place."""

    name: str = "abstract"
    # Channels this failure needs the target node to actually have.
    required_channels: tuple[str, ...] = ("accel",)

    @abstractmethod
    def params(self, severity: int) -> dict:
        """Requested parameters for a severity level."""

    @abstractmethod
    def apply(self, frames: list[NodeFrame], severity: int,
              rng: np.random.Generator, ctx: dict) -> dict:
        """Mutate `frames` in place. Return the REALISED effect metrics."""


class FailureInjector:
    """Wraps a DataSource, injecting one failure into one node."""

    def __init__(self, base, failure_type: str, severity: int,
                 target_node: str, seed: int):
        from injection.registry import STRATEGIES        # avoid import cycle

        if failure_type not in STRATEGIES:
            raise InjectionError(
                "Unknown failure_type %r. Registered: %s"
                % (failure_type, sorted(STRATEGIES)))
        if severity not in SEVERITIES:
            raise InjectionError(
                "severity must be one of %s, got %r" % (list(SEVERITIES),
                                                        severity))
        if target_node not in base.node_ids:
            raise InjectionError(
                "target_node %r not in base.node_ids %s"
                % (target_node, base.node_ids))

        self.base = base
        self.failure_type = failure_type
        self.severity = int(severity)
        self.target_node = target_node
        self.seed = int(seed)
        self.strategy: InjectorStrategy = STRATEGIES[failure_type]()

        # --- DataSource protocol surface, inherited from the wrapped source --
        self.node_ids: list[str] = list(base.node_ids)
        self.sampling_rate_hz: float = float(base.sampling_rate_hz)

        self.tag = "%s:sev%d" % (failure_type, self.severity)
        self._capability_checked = False
        self._notices: set[str] = set()

    # ------------------------------------------------------------------ API --
    def windows(self, window_sec: float = 2.56,
                overlap: float = 0.5) -> Iterator[Window]:
        # One Generator per injector instance, seeded explicitly. Never global
        # numpy state -- see the module docstring.
        rng = np.random.default_rng(self.seed)

        for w in self.base.windows(window_sec, overlap):
            target = w.frames.get(self.target_node)
            if not target:
                yield w
                continue

            if not self._capability_checked:
                self._check_capability(target[0])
                self._capability_checked = True

            # Deep-copy ONLY the target node. Every other node passes through
            # untouched, which is the control condition every later comparison
            # rests on.
            new_frames = dict(w.frames)
            injected = [copy.deepcopy(f) for f in target]

            ctx = {
                "window_start": w.start_sec,
                "window_end": w.end_sec,
                "sampling_rate_hz": self.sampling_rate_hz,
                "target_node": self.target_node,
                "notices": self._notices,
            }
            realised = self.strategy.apply(injected, self.severity, rng, ctx)

            for f in injected:
                f.injected_failure = self.tag
            new_frames[self.target_node] = injected

            meta = dict(w.meta) if w.meta else {}
            meta.update({
                "failure_type": self.failure_type,
                "severity": self.severity,
                "target_node": self.target_node,
                "seed": self.seed,
                "tag": self.tag,
                "requested": self.strategy.params(self.severity),
                "realised": realised,        # what ACTUALLY happened
            })
            # Injectors compose: keep a stack of every applied failure.
            stack = list(meta.get("injection_stack", []))
            stack.append(self.tag)
            meta["injection_stack"] = stack

            yield Window(start_sec=w.start_sec, end_sec=w.end_sec,
                         frames=new_frames, label=w.label, meta=meta)

    # ------------------------------------------------------------ internals --
    def _check_capability(self, sample: NodeFrame) -> None:
        """Refuse to target channels the node does not physically have."""
        present = set(sample.meta.get("channels_present", []))
        needed = set(self.strategy.required_channels)
        missing = needed - present
        if missing:
            raise InjectionError(
                "Cannot apply '%s' to node '%s': it has no %s.\n"
                "channels_present = %s.\n"
                "Refusing to silently no-op -- a no-op would mean the "
                "requested injection rate does not match the realised one, and "
                "Phase 6's severity axis would be quietly wrong.\n"
                "Pick a different target_node, or a failure type that only "
                "needs %s."
                % (self.failure_type, self.target_node,
                   " and ".join(sorted(missing)), sorted(present),
                   sorted(present)))

    def describe(self) -> str:
        return ("%s severity %d on %s (seed %d)"
                % (self.failure_type, self.severity, self.target_node,
                   self.seed))


# --------------------------------------------------------------------------- #
# helpers shared by injectors
# --------------------------------------------------------------------------- #
def blank_frame(f: NodeFrame) -> None:
    """Mark this instant as 'no measurement' on every channel the node HAS.

    NaN, never None: the sensor exists, it stopped reporting. None is reserved
    for structurally absent hardware and must never be written by an injector.
    """
    f.accel_g = NAN3
    if f.gyro_dps is not None:          # preserve structural absence exactly
        f.gyro_dps = NAN3


def nan_fraction(frames: list[NodeFrame]) -> float:
    if not frames:
        return 0.0
    return sum(1 for f in frames if is_nan_triple(f.accel_g)) / len(frames)
