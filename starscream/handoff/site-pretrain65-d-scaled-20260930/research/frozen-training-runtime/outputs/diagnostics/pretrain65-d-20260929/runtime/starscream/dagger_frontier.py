"""Offline prototype of bounded curriculum weights, not enabled in training.

Task horizon is part of a cell: a short collection window must not silently
change the policy's full-lap route goal. Admission requires separate teacher
qualification; these weights do not certify a start or a course as recoverable.
"""
import numpy as np


def replay_weights(cells, valid, frontier, frontier_mass=0.25):
    """Balance [family, track, task_horizon, regime, gate] within two pools.

    `frontier` identifies newly admitted tracks; everything retained is anchor.
    Repeating a long rollout cannot increase its cell's probability mass.
    Refuse an empty pool instead of silently losing the retention guarantee.
    """
    cells = np.asarray(cells)
    valid = np.asarray(valid, dtype=bool)
    frontier = np.asarray(frontier, dtype=bool)
    if cells.ndim != 2 or cells.shape[1] != 5 or valid.shape != (len(cells),) or frontier.shape != valid.shape:
        raise ValueError('expected N x 5 cells and N valid/frontier masks')
    if not 0 < frontier_mass < 1:
        raise ValueError('both pools require positive mass')
    weights = np.zeros(len(cells), dtype=np.float64)

    def assign(ids, depth, mass):
        if depth == cells.shape[1]:
            weights[ids] = mass / len(ids)
            return
        values = np.unique(cells[ids, depth])
        for value in values:
            assign(ids[cells[ids, depth] == value], depth + 1, mass / len(values))

    for pool, mass in [(False, 1 - frontier_mass), (True, frontier_mass)]:
        ids = np.flatnonzero(valid & (frontier == pool))
        if not len(ids):
            raise ValueError('missing valid anchor or frontier data; qualify/collect first')
        assign(ids, 0, mass)
    return weights


def budget_step(current, *, teacher_qualified, retained_ok, frontier_ready,
                stable_evaluations, step=0.05, lower=0.1, upper=0.4):
    """Change distribution, never roll back weights based on a noisy SR point.

    Callers determine readiness using matched-seed confidence bounds, including
    gate survival and full completions. Two independent evaluations required.
    This helper neither evaluates policies nor implements those statistical tests.
    """
    if not lower <= current <= upper or not 0 < step <= upper-lower:
        raise ValueError('invalid curriculum budget')
    if not teacher_qualified:
        return current, 'reject_candidate'
    if stable_evaluations < 2:
        return current, 'hold_for_evidence'
    if not retained_ok:
        return max(lower, current-step), 'repair_retention'
    if frontier_ready:
        return min(upper, current+step), 'expand'
    return current, 'fit_current_frontier'
