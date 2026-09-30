"""Empirical task embeddings from privileged DAgger trajectory distributions.

Geometry is only a proxy for the control problem.  This module embeds the
joint distribution that is actually supervised: current privileged state,
rolling route, previous command, MPCC action label, and next-state dynamics.
Global distribution summaries are paired with an ordered lap-phase signature
so two tracks cannot appear close merely because they contain the same turns
in a different order.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from numpy.typing import NDArray


F64 = NDArray[np.float64]

DEFAULT_BLOCK_WEIGHTS: Mapping[str, float] = {
    "task_state": 0.20,
    "route": 0.30,
    "previous_command": 0.05,
    "teacher_action": 0.20,
    "dynamics_target": 0.25,
}


@dataclass(frozen=True, slots=True)
class TrajectoryDistribution:
    """Control-relevant samples for one track under one collection regime."""

    name: str
    phase: F64
    blocks: Mapping[str, F64]
    source: str = ""

    def __post_init__(self) -> None:
        phase = np.asarray(self.phase, np.float64)
        if phase.ndim != 1 or not len(phase) or not np.all(np.isfinite(phase)):
            raise ValueError("trajectory phase must be a non-empty finite vector")
        normalized: dict[str, F64] = {}
        for name, raw in self.blocks.items():
            values = np.asarray(raw, np.float64)
            if values.ndim != 2 or values.shape[0] != len(phase):
                raise ValueError(f"trajectory block {name!r} has incompatible rows")
            if values.shape[1] < 1 or not np.all(np.isfinite(values)):
                raise ValueError(f"trajectory block {name!r} must be finite and non-empty")
            normalized[str(name)] = values
        if not normalized:
            raise ValueError("trajectory distribution needs at least one block")
        object.__setattr__(self, "phase", np.mod(phase, 1.0))
        object.__setattr__(self, "blocks", normalized)


@dataclass(frozen=True, slots=True)
class TrajectoryBehaviorProfile:
    """Fixed-dimensional embedding of one trajectory distribution."""

    name: str
    embedding: F64
    block_embeddings: Mapping[str, F64]
    rows: int
    source: str = ""

    def __post_init__(self) -> None:
        embedding = np.asarray(self.embedding, np.float64)
        blocks = {
            str(name): np.asarray(value, np.float64)
            for name, value in self.block_embeddings.items()
        }
        if embedding.ndim != 1 or not np.all(np.isfinite(embedding)):
            raise ValueError("behavior embedding must be a finite vector")
        if any(value.ndim != 1 or not np.all(np.isfinite(value)) for value in blocks.values()):
            raise ValueError("behavior block embeddings must be finite vectors")
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(self, "block_embeddings", blocks)


@dataclass(frozen=True, slots=True)
class BehavioralPosition:
    """Projection of one task onto an empirically defined source-target ray."""

    progress: float
    off_axis_distance: float
    target_distance: float
    source_distance: float

    def to_mapping(self) -> dict[str, float]:
        return {
            "progress": self.progress,
            "off_axis_distance": self.off_axis_distance,
            "target_distance": self.target_distance,
            "source_distance": self.source_distance,
        }


@dataclass(frozen=True, slots=True)
class PhaseAlignment:
    """Monotone alignment between two ordered control-problem signatures."""

    normalized_cost: float
    source_phases: F64
    target_phases: F64
    pairs: tuple[tuple[int, int], ...]
    pair_distances: F64

    def to_mapping(self) -> dict[str, Any]:
        return {
            "normalized_cost": self.normalized_cost,
            "source_phases": self.source_phases.tolist(),
            "target_phases": self.target_phases.tolist(),
            "pairs": [list(pair) for pair in self.pairs],
            "pair_distances": self.pair_distances.tolist(),
            "maximum_pair_distance": float(np.max(self.pair_distances)),
            "mean_pair_distance": float(np.mean(self.pair_distances)),
        }


class BehavioralEmbeddingModel:
    """Robust, block-balanced embedding of state/action trajectory occupancy.

    This is intentionally a statistical model rather than another learned
    neural encoder.  It is deterministic, cheap enough to run in every course
    qualification job, and exposes which semantic block caused a candidate to
    move.  A later policy-gradient/Fisher embedding can be added after this
    admission contract has accumulated enough transfer observations.
    """

    def __init__(
        self,
        *,
        block_weights: Mapping[str, float] | None = None,
        quantiles: Sequence[float] = (0.10, 0.50, 0.90),
        phase_samples: int = 16,
        clip: float = 8.0,
        minimum_scale: float = 1.0e-3,
    ) -> None:
        weights = dict(block_weights or DEFAULT_BLOCK_WEIGHTS)
        if not weights or any(float(value) <= 0.0 for value in weights.values()):
            raise ValueError("behavior block weights must be positive")
        total = float(sum(weights.values()))
        self.block_weights = {name: float(value) / total for name, value in weights.items()}
        self.quantiles = tuple(float(value) for value in quantiles)
        if not self.quantiles or any(not 0.0 < value < 1.0 for value in self.quantiles):
            raise ValueError("behavior quantiles must lie strictly inside (0, 1)")
        if int(phase_samples) < 4:
            raise ValueError("phase_samples must be at least four")
        self.phase_samples = int(phase_samples)
        self.clip = float(clip)
        self.minimum_scale = float(minimum_scale)
        self._centers: dict[str, F64] = {}
        self._scales: dict[str, F64] = {}

    @property
    def fitted(self) -> bool:
        return bool(self._centers)

    def fit(self, distributions: Sequence[TrajectoryDistribution]) -> "BehavioralEmbeddingModel":
        if not distributions:
            raise ValueError("cannot fit a behavior embedding without distributions")
        required = tuple(self.block_weights)
        for block in required:
            missing = [row.name for row in distributions if block not in row.blocks]
            if missing:
                raise ValueError(f"behavior block {block!r} is missing from {missing}")
            widths = {row.blocks[block].shape[1] for row in distributions}
            if len(widths) != 1:
                raise ValueError(f"behavior block {block!r} has inconsistent widths")
            pooled = np.concatenate([row.blocks[block] for row in distributions], axis=0)
            center = np.median(pooled, axis=0)
            q25, q75 = np.quantile(pooled, (0.25, 0.75), axis=0)
            self._centers[block] = center
            self._scales[block] = np.maximum(q75 - q25, self.minimum_scale)
        return self

    def _standardize(self, block: str, values: F64) -> F64:
        if not self.fitted:
            raise RuntimeError("fit the behavior embedding model before transform")
        standardized = (values - self._centers[block]) / self._scales[block]
        return np.clip(standardized, -self.clip, self.clip)

    def _global_summary(self, values: F64) -> F64:
        width = values.shape[1]
        pieces = [values.mean(axis=0), values.std(axis=0)]
        pieces.extend(np.quantile(values, self.quantiles, axis=0))
        return np.concatenate(pieces) / np.sqrt(width * len(pieces))

    def _phase_summary(self, values: F64, phase: F64) -> F64:
        unique = np.unique(phase)
        local = []
        for value in unique:
            selected = values[np.isclose(phase, value, rtol=0.0, atol=1.0e-6)]
            local.append(np.concatenate([selected.mean(axis=0), selected.std(axis=0)]))
        local_values = np.asarray(local, np.float64)
        closed_phase = np.concatenate([unique, [1.0]])
        closed_values = np.concatenate([local_values, local_values[:1]], axis=0)
        query = np.arange(self.phase_samples, dtype=np.float64) / self.phase_samples
        signature = np.column_stack([
            np.interp(query, closed_phase, closed_values[:, index])
            for index in range(closed_values.shape[1])
        ])
        return signature.reshape(-1) / np.sqrt(signature.size)

    def transform(self, distribution: TrajectoryDistribution) -> TrajectoryBehaviorProfile:
        block_embeddings: dict[str, F64] = {}
        weighted: list[F64] = []
        for block, weight in self.block_weights.items():
            values = self._standardize(block, distribution.blocks[block])
            global_summary = self._global_summary(values) / np.sqrt(2.0)
            phase_summary = self._phase_summary(values, distribution.phase) / np.sqrt(2.0)
            embedding = np.concatenate([global_summary, phase_summary])
            block_embeddings[block] = embedding
            weighted.append(np.sqrt(weight) * embedding)
        return TrajectoryBehaviorProfile(
            name=distribution.name,
            embedding=np.concatenate(weighted),
            block_embeddings=block_embeddings,
            rows=len(distribution.phase),
            source=distribution.source,
        )

    def fit_transform(
        self, distributions: Sequence[TrajectoryDistribution],
    ) -> tuple[TrajectoryBehaviorProfile, ...]:
        self.fit(distributions)
        return tuple(self.transform(row) for row in distributions)

    def phase_signature(
        self,
        distribution: TrajectoryDistribution,
        *,
        blocks: Sequence[str] = ("route", "teacher_action", "dynamics_target"),
    ) -> tuple[F64, F64]:
        """Return a block-balanced signature at every physical gate phase.

        This intentionally keeps the native gate count. The fixed-size profile
        above is useful for retrieval; this ordered signature is for detecting
        missing/duplicated maneuver phases and for constructing graph edges.
        """

        phases = np.unique(distribution.phase)
        pieces: list[F64] = []
        for block in blocks:
            if block not in self.block_weights:
                raise ValueError(f"unknown behavior block {block!r}")
            values = self._standardize(block, distribution.blocks[block])
            phase_values = np.stack([
                values[np.isclose(distribution.phase, phase, rtol=0.0, atol=1.0e-6)].mean(axis=0)
                for phase in phases
            ])
            pieces.append(
                np.sqrt(self.block_weights[block])
                * phase_values
                / np.sqrt(phase_values.shape[1])
            )
        return phases, np.concatenate(pieces, axis=1)


def align_phase_signatures(
    source_phases: F64,
    source_signature: F64,
    target_phases: F64,
    target_signature: F64,
) -> PhaseAlignment:
    """Dynamic-time-warp ordered gate phases under a Euclidean block metric."""

    source_phases = np.asarray(source_phases, np.float64)
    target_phases = np.asarray(target_phases, np.float64)
    source = np.asarray(source_signature, np.float64)
    target = np.asarray(target_signature, np.float64)
    if (
        source.ndim != 2 or target.ndim != 2
        or source.shape[1] != target.shape[1]
        or source.shape[0] != len(source_phases)
        or target.shape[0] != len(target_phases)
    ):
        raise ValueError("phase signatures have incompatible shapes")
    pair_cost = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2)
    accumulated = np.full(pair_cost.shape, np.inf, np.float64)
    parent = np.full((*pair_cost.shape, 2), -1, np.int64)
    accumulated[0, 0] = pair_cost[0, 0]
    for left in range(source.shape[0]):
        for right in range(target.shape[0]):
            if left == 0 and right == 0:
                continue
            candidates: list[tuple[float, int, int]] = []
            if left:
                candidates.append((accumulated[left - 1, right], left - 1, right))
            if right:
                candidates.append((accumulated[left, right - 1], left, right - 1))
            if left and right:
                candidates.append((accumulated[left - 1, right - 1], left - 1, right - 1))
            value, parent_left, parent_right = min(candidates, key=lambda row: row[0])
            accumulated[left, right] = value + pair_cost[left, right]
            parent[left, right] = (parent_left, parent_right)
    pairs: list[tuple[int, int]] = []
    left, right = source.shape[0] - 1, target.shape[0] - 1
    while left >= 0 and right >= 0:
        pairs.append((left, right))
        previous = parent[left, right]
        if previous[0] < 0:
            break
        left, right = int(previous[0]), int(previous[1])
    pairs.reverse()
    distances = np.asarray([pair_cost[left, right] for left, right in pairs], np.float64)
    return PhaseAlignment(
        normalized_cost=float(accumulated[-1, -1] / len(pairs)),
        source_phases=source_phases,
        target_phases=target_phases,
        pairs=tuple(pairs),
        pair_distances=distances,
    )


def behavioral_position(
    source: F64,
    target: F64,
    query: F64,
) -> BehavioralPosition:
    """Project a query embedding onto a source-target behavioral direction."""

    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    query = np.asarray(query, np.float64)
    if source.shape != target.shape or source.shape != query.shape:
        raise ValueError("behavioral position embeddings must have matching shapes")
    direction = target - source
    denominator = float(direction @ direction)
    if denominator <= 1.0e-12:
        raise ValueError("source and target behavior embeddings are indistinguishable")
    delta = query - source
    progress = float(delta @ direction / denominator)
    residual = delta - progress * direction
    return BehavioralPosition(
        progress=progress,
        off_axis_distance=float(np.linalg.norm(residual)),
        target_distance=float(np.linalg.norm(query - target)),
        source_distance=float(np.linalg.norm(delta)),
    )


def replay_trajectory_distributions(
    path: str | Path,
    *,
    group: str = "online",
    maximum_rows_per_track: int = 0,
    seed: int = 0,
    name_prefix: str = "",
) -> tuple[TrajectoryDistribution, ...]:
    """Load per-track legacy103 trajectory distributions from a DAgger shard."""

    replay_path = Path(path)
    rows: list[TrajectoryDistribution] = []
    with h5py.File(replay_path, "r") as archive:
        contract = json.loads(str(archive.attrs["contract_json"]))
        if str(contract.get("observation_contract")) != "starscream_route_v1":
            raise ValueError("behavioral replay audit currently requires legacy103 observations")
        input_dim = int(contract["input_dim"])
        route_gate_count = int(contract["route_gate_count"])
        route_end = 19 + route_gate_count * 13
        if route_end + 6 != input_dim:
            raise ValueError("legacy103 replay has an inconsistent route-width contract")
        payload = archive[group]
        track_ids = np.asarray(payload["tracks"][:], np.int64)
        rng = np.random.default_rng(seed)
        for track_index, raw_path in enumerate(contract["tracks"]):
            indices = np.flatnonzero(track_ids == track_index)
            if "dynamics_valid" in payload:
                indices = indices[np.asarray(payload["dynamics_valid"][indices], bool)]
            if maximum_rows_per_track > 0 and len(indices) > maximum_rows_per_track:
                indices = np.sort(rng.choice(
                    indices, size=int(maximum_rows_per_track), replace=False,
                ))
            if not len(indices):
                continue
            history = np.asarray(payload["histories"][indices, -1], np.float64)
            name = Path(str(raw_path)).stem
            rows.append(TrajectoryDistribution(
                name=f"{name_prefix}{name}",
                phase=np.asarray(payload["progress"][indices], np.float64),
                blocks={
                    "task_state": history[:, :19],
                    "route": history[:, 19:route_end],
                    "previous_command": history[:, route_end:input_dim],
                    "teacher_action": np.asarray(payload["actions"][indices], np.float64),
                    "dynamics_target": np.asarray(payload["dynamics"][indices], np.float64),
                },
                source=str(replay_path),
            ))
    return tuple(rows)


def corrected_target_embedding(
    source_anchor: TrajectoryBehaviorProfile,
    reference_source: TrajectoryBehaviorProfile,
    reference_target: TrajectoryBehaviorProfile,
) -> F64:
    """Transfer a within-run target delta onto an anchor from another run.

    DAgger replay is policy- and randomization-conditioned.  Subtracting the
    same-source profile from the reference run removes most collector/checkpoint
    nuisance before comparing generated candidates from another run.
    """

    if not (
        source_anchor.embedding.shape == reference_source.embedding.shape
        == reference_target.embedding.shape
    ):
        raise ValueError("corrected target profiles must share an embedding contract")
    return source_anchor.embedding + (
        reference_target.embedding - reference_source.embedding
    )
