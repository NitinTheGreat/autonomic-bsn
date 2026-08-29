"""BlindWindow — a read-only view that PHYSICALLY CANNOT expose ground truth.

The health monitor must infer degradation from observable signal statistics
alone. If it can read `injected_failure` or the injection metadata, its
detection metrics measure nothing but its ability to read an answer key, and
every number in the Results section becomes circular.

Phase 3's handoff asked for this to be enforced structurally rather than by
discipline. This module is that enforcement: `health.signals` and
`health.diagnose` accept ONLY a BlindWindow, and a BlindWindow has no path to
the stripped fields.

Why AttributeError instead of returning None
--------------------------------------------
Silently returning None would let buggy code "work" while reading nothing --
a monitor that accidentally branched on `frame.injected_failure` would simply
see None everywhere and appear correct, and the bug would survive review. A
raised AttributeError fails at the first read, loudly, in a stack trace that
names the offending line.

What is stripped
----------------
    NodeFrame.injected_failure
    Window.label                  the true activity -- see note below
    Window.meta: failure_type, severity, target_node, seed, requested,
                 realised, gravity_axis_alignment, tag, injection_stack

`label` is stripped beyond what the Phase 4 brief listed. In deployment the
activity is exactly what the system is trying to infer, so a monitor that
conditions on the true activity is using information it would never have. The
scoring module reads the ORIGINAL Window, so stratifying results by activity is
unaffected.

What is preserved
-----------------
    meta: channels_present, timestamp_derived, clock_source, dataset_name,
          subject_id, sampling_rate_hz, node placement
    every sensor value and timestamp

`channels_present` in particular MUST survive: a node cannot be penalised for a
sensor it never had, and MHEALTH's chest has no gyroscope.
"""

from __future__ import annotations

from typing import Iterator

from core.datasource import NodeFrame, Window

# Ground-truth keys that must never reach the monitor.
STRIPPED_META_KEYS = frozenset({
    "failure_type", "severity", "target_node", "seed", "requested",
    "realised", "gravity_axis_alignment", "tag", "injection_stack",
})

# Keys the monitor legitimately needs to reason about the signal.
PRESERVED_META_KEYS = frozenset({
    "channels_present", "timestamp_derived", "clock_source", "dataset_name",
    "subject_id", "sampling_rate_hz", "clock_offset_applied_ms", "ecg",
    "heart_rate", "timestamp_source",
})


class GroundTruthAccessError(AttributeError):
    """Raised when monitor code reaches for a stripped ground-truth field."""


class BlindFrame:
    """A NodeFrame with `injected_failure` removed rather than blanked."""

    __slots__ = ("node_id", "t_sec", "accel_g", "gyro_dps", "source", "meta")

    def __init__(self, f: NodeFrame):
        self.node_id = f.node_id
        self.t_sec = f.t_sec
        self.accel_g = f.accel_g
        self.gyro_dps = f.gyro_dps          # None stays None: absent sensor
        self.source = f.source
        self.meta = {k: v for k, v in (f.meta or {}).items()
                     if k not in STRIPPED_META_KEYS}

    def __getattr__(self, name):
        if name == "injected_failure":
            raise GroundTruthAccessError(
                "BlindFrame deliberately does not expose 'injected_failure'.\n"
                "That field is ground truth, written by Phase 3's injector and "
                "read ONLY by health.score_detection. A monitor that reads it "
                "is scored on its ability to read an answer key, which makes "
                "every detection metric circular.\n"
                "If you need it for scoring, use the original Window.")
        raise AttributeError(
            "%r object has no attribute %r" % (type(self).__name__, name))

    @property
    def channels_present(self) -> list[str]:
        return list(self.meta.get("channels_present", []))

    def has_gyro(self) -> bool:
        """Structural availability, not 'is this sample valid'."""
        return self.gyro_dps is not None


class BlindWindow:
    """A Window with every ground-truth field removed."""

    __slots__ = ("start_sec", "end_sec", "frames", "meta", "_node_ids")

    def __init__(self, w: Window):
        self.start_sec = w.start_sec
        self.end_sec = w.end_sec
        self.frames: dict[str, list[BlindFrame]] = {
            node: [BlindFrame(f) for f in fs] for node, fs in w.frames.items()}
        self.meta = {k: v for k, v in (w.meta or {}).items()
                     if k not in STRIPPED_META_KEYS}
        self._node_ids = sorted(self.frames)

    def __getattr__(self, name):
        if name in ("label", "injected_failure") or name in STRIPPED_META_KEYS:
            raise GroundTruthAccessError(
                "BlindWindow deliberately does not expose %r.\n"
                "It is ground truth about this window. The monitor must infer "
                "degradation from signal statistics alone; health.score_"
                "detection reads the original Window for scoring." % name)
        raise AttributeError(
            "%r object has no attribute %r" % (type(self).__name__, name))

    @property
    def node_ids(self) -> list[str]:
        return list(self._node_ids)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def __repr__(self) -> str:
        return ("BlindWindow(%.2f-%.2fs, nodes=%s)"
                % (self.start_sec, self.end_sec, self._node_ids))


def blind(w: Window) -> BlindWindow:
    """Wrap a Window for monitor consumption."""
    return BlindWindow(w)


def blind_all(windows: Iterator[Window]) -> Iterator[BlindWindow]:
    for w in windows:
        yield BlindWindow(w)


def require_blind(w, caller: str = "this function") -> BlindWindow:
    """Assert the caller was handed a BlindWindow, not a raw Window.

    Called at every entry point in health.signals and health.diagnose. Passing
    a raw Window would silently re-open the ground-truth channel, so it is a
    hard error rather than an implicit conversion.
    """
    if isinstance(w, BlindWindow):
        return w
    raise TypeError(
        "%s accepts only a BlindWindow, got %s.\n"
        "Wrap it first: health.window_view.blind(window). Accepting a raw "
        "Window here would re-open the ground-truth channel this module "
        "exists to close." % (caller, type(w).__name__))
