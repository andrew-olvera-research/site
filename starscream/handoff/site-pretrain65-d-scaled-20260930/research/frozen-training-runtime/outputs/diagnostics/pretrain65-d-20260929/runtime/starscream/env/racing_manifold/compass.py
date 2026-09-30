"""Trajectory support and feasible-graph compass prototypes.

These are task proposal scores, NOT estimates of causal learning transfer.
Windows retain order and never wrap across reset/terminal boundaries. A course
is a set of control problems; a single global average is deliberately avoided.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import kron, eye, csr_matrix, vstack
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra


def ground_clearance_profile(track, *, floor_z: float = 0.0) -> dict:
    """Ambient geometry lost by translation-invariant route coordinates.

    These are gate clearances, not swept-volume or expert-trajectory collision
    certification. ``floor_z`` is the physical floor, not a display bound.
    """
    rows = []
    if not np.isfinite(floor_z):
        raise ValueError("floor height must be finite")
    for i,g in enumerate(track.gates):
        if not g.render:
            continue
        extent = .5*(abs(float(g.lateral[2]))*g.size[0]+abs(float(g.up[2]))*g.size[1])
        rows.append(dict(gate=i, center_above_floor_m=float(g.position[2]-floor_z),
                         aperture_bottom_above_floor_m=float(g.position[2]-extent-floor_z)))
    if not rows:
        raise ValueError("ground profile requires physical gates")
    return dict(floor_z=float(floor_z), gates=rows,
                minimum_center_clearance_m=min(r["center_above_floor_m"] for r in rows),
                minimum_aperture_clearance_m=min(r["aperture_bottom_above_floor_m"] for r in rows))


@dataclass(frozen=True)
class ControlWindows:
    values: np.ndarray
    gates: np.ndarray

    def __post_init__(self):
        x, g = np.asarray(self.values, float), np.asarray(self.gates)
        if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError("windows must be a finite nonempty matrix")
        if g.shape != (len(x),) or not np.isfinite(g).all() or (g < 0).any():
            raise ValueError("windows require valid per-row gate indices")
        if not np.equal(g, g.astype(int)).all() or (np.diff(g) < 0).any():
            raise ValueError("windows must be in episode order with integer gates")
        object.__setattr__(self, "values", x)
        object.__setattr__(self, "gates", g.astype(int))


def control_windows(features, actions, dynamics, gates, *, hz: float,
                    bins_per_gate: int = 4, horizon_s: float = .2,
                    view: str = "policy") -> ControlWindows:
    """Legacy route-six physical scaling; 3 samples from actual contiguous flight.

    Route positions /20m; gate aperture /3m; velocity /20m/s; rates /10rad/s;
    motors /3000rad/s; CTBR /[40,10,10,10]. Dynamics already rate-normalized.
    Block RMS weights .25 task, .35 route, .15 action, .25 dynamics.
    No fit statistics from the target. No averaging incompatible trajectories.
    """
    f, a, d, g = map(np.asarray, (features, actions, dynamics, gates))
    if f.ndim != 2 or f.shape[1] != 103 or a.shape != (len(f), 4):
        raise ValueError("requires aligned legacy103/route-six and CTBR actions")
    if d.shape != (len(f), 12) or g.shape != (len(f),):
        raise ValueError("requires aligned invariant12 dynamics and gate phases")
    if hz <= 0 or bins_per_gate < 1 or horizon_s < 0:
        raise ValueError("invalid window timing or bins")
    if not all(np.isfinite(v).all() for v in (f, a, d, g)) or (np.diff(g) < 0).any():
        raise ValueError("nonfinite or disordered episode")
    offsets = np.unique(np.rint(np.array([0, .5, 1]) * horizon_s * hz).astype(int))
    task = f[:, :19] / np.array([20]*6 + [1]*6 + [10]*3 + [3000]*4)
    route = f[:, 19:97].reshape(-1, 6, 13).copy()
    route[..., :3] /= 20
    route[..., 9:11] /= 3
    route = route.reshape(-1, 78)
    if view == "policy":
        blocks = (task, route, a / np.array([40, 10, 10, 10]), d)
    elif view == "motion":
        # Remove controller-specific yaw/roll choices from task similarity.
        # Preserve physical acceleration in the active directed-gate frame.
        axes = f[:, 6:12].reshape(-1,2,3)
        rotation = np.stack([axes[:,0], axes[:,1], np.cross(axes[:,0],axes[:,1])],axis=2)
        gate_acceleration = np.einsum("nij,nj->ni",rotation,d[:,3:6])
        blocks = (task[:,:6], route, a[:,:1]/40, gate_acceleration)
    else:
        raise ValueError("view must be policy or motion")
    values = np.concatenate([b*np.sqrt(w/b.shape[1]) for b, w in
                             zip(blocks, (.25, .35, .15, .25))], axis=1)
    anchors = []
    for gate in np.unique(g):
        eligible = np.flatnonzero((g == gate) & (np.arange(len(g)) + offsets[-1] < len(g)))
        if not len(eligible):
            raise ValueError(f"gate {gate} has no complete windows")
        indices = np.rint(np.linspace(0, len(eligible)-1, bins_per_gate)).astype(int)
        anchors.extend(eligible[indices])
    anchors = np.asarray(anchors)
    x = values[anchors[:, None] + offsets].reshape(len(anchors), -1) / np.sqrt(len(offsets))
    return ControlWindows(x, g[anchors])


def ordered_distance(source: ControlWindows, target: ControlWindows) -> float:
    """Start/end-anchored DTW; no cyclic shift, Euclidean ground distance."""
    cost = cdist(source.values, target.values)
    dp = np.full((len(source.values)+1, len(target.values)+1), np.inf)
    lengths = np.zeros(dp.shape, int)
    dp[0, 0] = 0
    for i in range(1, dp.shape[0]):
        for j in range(1, dp.shape[1]):
            predecessors = ((i-1,j-1), (i-1,j), (i,j-1))
            p, q = min(predecessors, key=lambda ij: dp[ij])
            dp[i,j] = dp[p,q] + cost[i-1,j-1]
            lengths[i,j] = lengths[p,q] + 1
    return float(dp[-1,-1] / lengths[-1,-1])


def balanced_transport(source: ControlWindows, target: ControlWindows) -> float:
    """Exact small empirical W1 transport. Order is inside each window only."""
    cost = cdist(source.values, target.values)
    n, m = cost.shape
    # Equal mass per phase bin: duration/failure frequency cannot set the metric.
    constraints = vstack([kron(eye(n), csr_matrix(np.ones((1,m)))),
                          kron(csr_matrix(np.ones((1,n))), eye(m))]).tocsr()
    result = linprog(cost.ravel(), A_eq=constraints,
                     b_eq=np.r_[np.full(n,1/n), np.full(m,1/m)],
                     bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun)


def target_residual(target: ControlWindows, support: Sequence[ControlWindows]) -> np.ndarray:
    if not support:
        raise ValueError("support cannot be empty")
    return cdist(target.values, np.concatenate([s.values for s in support])).min(axis=1)


def support_gain(target: ControlWindows, support: Sequence[ControlWindows],
                 candidate: ControlWindows, *, critical_gates: Sequence[int] = ()) -> dict:
    """Asymmetric marginal target coverage gain; report every gate and worst tail.

    A proposal cannot count repeatedly covered easy states as new coverage.
    Do not call a reduction evidence that PPO will learn those states.
    """
    before = target_residual(target, support)
    after = np.minimum(before, target_residual(target, [candidate]))
    rows = []
    for gate in np.unique(target.gates):
        mask = target.gates == gate
        b, a = float(before[mask].mean()), float(after[mask].mean())
        rows.append(dict(gate=int(gate), before=b, after=a,
                         gain_fraction=float((b-a)/max(b,1e-12))))
    unknown = set(critical_gates) - set(target.gates)
    if unknown:
        raise ValueError(f"critical gates absent from target: {unknown}")
    critical = [r["gain_fraction"] for r in rows if r["gate"] in critical_gates]
    return dict(mean_before=float(before.mean()), mean_after=float(after.mean()),
                gain_fraction=float((before.mean()-after.mean())/max(before.mean(),1e-12)),
                worst20_before=float(np.sort(before)[-max(1,int(np.ceil(.2*len(before)))):].mean()),
                worst20_after=float(np.sort(after)[-max(1,int(np.ceil(.2*len(after)))):].mean()),
                critical_min_gain=None if not critical else float(min(critical)), per_gate=rows)


def feasible_path(distances, eligible, *, source: int, target: int, radius: float) -> dict:
    """Shortest path on MPCC-qualified local edges; disconnection is an outcome.

    This is task-space reachability only. Policy admission must independently
    enforce per-course competence/confidence and source retention online.
    """
    d, ok = np.asarray(distances, float), np.asarray(eligible, bool)
    if d.shape != (len(ok), len(ok)) or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("invalid graph distances")
    if not np.allclose(d, d.T) or radius <= 0:
        raise ValueError("graph requires symmetric distances and positive radius")
    if not (0 <= source < len(ok) and 0 <= target < len(ok)):
        raise ValueError("source/target index out of range")
    connected = (d <= radius) & ok[:,None] & ok[None,:]
    np.fill_diagonal(connected, False)
    # Sparse graph zero means absent; preserve duplicate task edges explicitly.
    weights = np.where(connected, np.maximum(d,1e-12), 0)
    distance, previous = dijkstra(csr_matrix(weights), directed=False,
                                  indices=source, return_predecessors=True)
    if not ok[source] or not ok[target] or not np.isfinite(distance[target]):
        return dict(reachable=False, path=[], cost=None,
                    reached_indices=np.flatnonzero(np.isfinite(distance) & ok).tolist())
    path = [target]
    while path[-1] != source:
        path.append(int(previous[path[-1]]))
    return dict(reachable=True, path=path[::-1], cost=float(distance[target]))


class LinearWindowEncoder:
    """Train-only PCA sanity baseline for a learned chart, NOT transfer learning.

    Reconstruction residual is an OOD diagnostic; projecting a novel target
    close by discarding its unusual directions must not count as progress.
    """
    def fit(self, windows: Sequence[ControlWindows], rank: int = 16):
        if not windows or rank < 1:
            raise ValueError("positive rank and nonempty training windows required")
        x = np.concatenate([w.values for w in windows])
        self.center = x.mean(0)
        _, _, vt = np.linalg.svd(x-self.center, full_matrices=False)
        self.basis = vt[:min(rank, len(x)-1)]
        return self

    def transform(self, windows: ControlWindows) -> ControlWindows:
        return ControlWindows((windows.values-self.center) @ self.basis.T, windows.gates)

    def residual(self, windows: ControlWindows) -> float:
        x = windows.values-self.center
        return float(np.linalg.norm(x-(x@self.basis.T)@self.basis, axis=1).mean())
