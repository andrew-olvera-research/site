"""Directed failure-weighted course quotas for fixed-window PPO.

The frontier rule (``adaptive_frontier_weights``) centres lane quotas on a
target competence and deliberately de-emphasises courses at 0%. This variant
is the Green et al. (2026, A.4) analogue for a finite pool: every course keeps
a floor, surplus lanes go to courses in proportion to their failure rate, and
a course whose competence stays flat near zero for many windows is treated as
a plateaued task and decays toward the floor instead of absorbing lanes.
"""
import numpy as np


def directed_failure_weights(base_weights, previous_competence, observed_competence,
                             stall_windows, *, ema, power, minimum_multiplier,
                             maximum_multiplier, stall_competence, stall_patience,
                             stall_factor, progress_epsilon=0.02):
    """Return (weights, competence, multipliers, stall_windows).

    ``stall_windows`` counts consecutive windows in which a course was observed
    below ``stall_competence`` without moving by more than ``progress_epsilon``.
    Once it exceeds ``stall_patience`` the course's score is scaled by
    ``stall_factor`` (a plateau, not a frontier). Any observed progress resets it.
    """
    base = np.asarray(base_weights, np.float64)
    previous = np.asarray(previous_competence, np.float64)
    observed = np.asarray(observed_competence, np.float64)
    stalls = np.asarray(stall_windows, np.int64).copy()
    if not (base.shape == previous.shape == observed.shape == stalls.shape):
        raise ValueError('directed sampling arrays must have identical shapes')
    if (not 0.0 < ema <= 1.0 or power <= 0.0 or minimum_multiplier <= 0.0
            or maximum_multiplier < minimum_multiplier
            or not 0.0 <= stall_competence <= 1.0 or stall_patience < 1
            or not 0.0 < stall_factor <= 1.0):
        raise ValueError('invalid directed sampling settings')
    seen = np.isfinite(observed)
    updated = np.where(seen, (1.0 - ema) * previous + ema * observed, previous)
    moved = np.abs(updated - previous) > progress_epsilon
    low = updated < stall_competence
    stalls = np.where(seen & low & ~moved, stalls + 1, np.where(seen & moved, 0, stalls))
    stalled = stalls > stall_patience
    score = np.power(np.clip(1.0 - updated, 0.0, 1.0), power)
    score = score / max(float(score.mean()), 1.0e-6)
    multiplier = np.clip(score, minimum_multiplier, maximum_multiplier)
    # Apply the plateau decay AFTER the cap: in a mostly solved pool every
    # failing course sits at the cap, so a decay before clipping does nothing.
    multiplier = np.where(
        stalled, np.maximum(multiplier * stall_factor, minimum_multiplier), multiplier)
    return base * multiplier, updated, multiplier, stalls
