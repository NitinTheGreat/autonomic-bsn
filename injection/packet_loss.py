"""F3 — bursty packet loss, Gilbert-Elliott two-state Markov chain.

NOT i.i.d. Bernoulli. Real BLE loss is bursty, and i.i.d. loss is markedly
easier for a classifier to survive: isolated missing samples are trivially
interpolated over, while a 6-sample burst removes a whole gait feature. Using
Bernoulli here would understate the effect of degradation and make the entire
study optimistic.

Model
-----
Two states, Good (no loss) and Bad (loss). Loss probability is 1 in Bad and 0
in Good, so the stationary probability of Bad IS the mean loss rate L:

    r = 1 / B                       # Bad -> Good, giving mean burst length B
    p = L / ((1 - L) * B)           # Good -> Bad

Check: pi_Bad = p / (p + r) = L.

    severity 1..4 -> (L, B) = (0.05, 3), (0.15, 4), (0.30, 6), (0.50, 8)

Lost samples become NaN tuples -- the same convention as dropout, because a
lost packet and a dead node both mean "no data for this instant". They differ
in PATTERN, which is exactly what Phase 4's monitor must learn to separate.
"""

from __future__ import annotations

import numpy as np

from core.datasource import NodeFrame
from injection.base import InjectorStrategy, blank_frame

# severity -> (mean loss rate L, mean burst length B in samples)
LOSS_PARAMS = {1: (0.05, 3.0), 2: (0.15, 4.0), 3: (0.30, 6.0), 4: (0.50, 8.0)}


def transition_probs(L: float, B: float) -> tuple[float, float]:
    """(p_good_to_bad, r_bad_to_good) for target loss rate L, burst length B."""
    r = 1.0 / B
    p = L / ((1.0 - L) * B)
    return p, r


def run_chain(n: int, L: float, B: float,
              rng: np.random.Generator) -> np.ndarray:
    """Boolean array: True where the sample is lost."""
    p, r = transition_probs(L, B)
    lost = np.zeros(n, dtype=bool)
    # Start in the stationary distribution so short windows are not biased
    # toward Good by an arbitrary cold start.
    bad = bool(rng.random() < L)
    for i in range(n):
        lost[i] = bad
        if bad:
            if rng.random() < r:
                bad = False
        elif rng.random() < p:
            bad = True
    return lost


def burst_stats(lost: np.ndarray) -> dict:
    """Realised loss rate and mean burst length."""
    n = len(lost)
    if n == 0:
        return {"loss_rate": 0.0, "mean_burst": 0.0, "n_bursts": 0}
    bursts, run = [], 0
    for v in lost:
        if v:
            run += 1
        elif run:
            bursts.append(run)
            run = 0
    if run:
        bursts.append(run)
    return {
        "loss_rate": float(lost.mean()),
        "mean_burst": float(np.mean(bursts)) if bursts else 0.0,
        "n_bursts": len(bursts),
    }


class PacketLossInjector(InjectorStrategy):
    name = "packet_loss"
    required_channels = ("accel",)

    def params(self, severity: int) -> dict:
        L, B = LOSS_PARAMS[severity]
        p, r = transition_probs(L, B)
        return {"target_loss_rate": L, "target_mean_burst": B,
                "p_good_to_bad": p, "r_bad_to_good": r,
                "model": "gilbert_elliott"}

    def apply(self, frames: list[NodeFrame], severity: int,
              rng: np.random.Generator, ctx: dict) -> dict:
        L, B = LOSS_PARAMS[severity]
        lost = run_chain(len(frames), L, B, rng)
        for f, is_lost in zip(frames, lost):
            if is_lost:
                blank_frame(f)
        st = burst_stats(lost)
        return {"loss_rate": st["loss_rate"], "mean_burst": st["mean_burst"],
                "n_bursts": st["n_bursts"], "n_frames": len(frames),
                "target_loss_rate": L, "target_mean_burst": B}
