"""Cheap, policy-contract-faithful descriptors of rolling gate grammar.

The privileged actor does not observe an unordered point cloud or merely the
next inter-gate displacement.  It observes six ordered 13-D flight-plan
records in the active gate's directed frame.  Candidate generation therefore
needs a static proxy in that same coordinate system before paying for MPCC
qualification and frozen-policy probes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..tracks import Track
from .occupancy import PhaseAlignment, align_phase_signatures


F64 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RouteGrammarProfile:
    """Ordered and fixed-width representations of one route contract."""

    name: str
    phases: F64
    records: F64
    embedding: F64
    gate_count: int
    future_gates: int


@dataclass(frozen=True, slots=True)
class RouteDirectedPosition:
    """Candidate coordinates on and away from a source-to-target route ray."""

    progress: float
    off_axis_ratio: float
    source_distance: float
    target_distance: float
    target_gain_fraction: float

    def to_mapping(self) -> dict[str, float]:
        return {
            "progress": self.progress,
            "off_axis_ratio": self.off_axis_ratio,
            "source_distance": self.source_distance,
            "target_distance": self.target_distance,
            "target_gain_fraction": self.target_gain_fraction,
        }


def _normalize_records(
    records: F64,
    *,
    position_scale_m: float,
    aperture_scale_m: float,
) -> F64:
    if position_scale_m <= 0.0 or aperture_scale_m <= 0.0:
        raise ValueError("route grammar scales must be positive")
    shaped = np.asarray(records, np.float64).copy()
    if shaped.ndim != 3 or shaped.shape[-1] != 13:
        raise ValueError("route records must have shape [phase, gate, 13]")
    shaped[..., 0:3] /= float(position_scale_m)
    shaped[..., 9:11] /= float(aperture_scale_m)
    return shaped


def _periodic_interpolate(values: F64, samples: int) -> F64:
    if samples < 4:
        raise ValueError("route grammar needs at least four phase samples")
    count = values.shape[0]
    phases = np.arange(count, dtype=np.float64) / count
    closed_phase = np.concatenate([phases, [1.0]])
    closed_values = np.concatenate([values, values[:1]], axis=0)
    query = np.arange(samples, dtype=np.float64) / samples
    return np.column_stack([
        np.interp(query, closed_phase, closed_values[:, index])
        for index in range(closed_values.shape[1])
    ])


def route_grammar_profile(
    track: Track,
    *,
    future_gates: int = 6,
    phase_samples: int = 24,
    position_scale_m: float = 20.0,
    aperture_scale_m: float = 3.0,
) -> RouteGrammarProfile:
    """Encode the exact static route fields supplied to the legacy103 actor.

    Every physical gate is used once as the active phase.  ``Track.flight_plan``
    supplies the same terminal repetition, directed frame, route order, and
    opposite-side bit as the environment observation contract.
    """

    future_gates = int(future_gates)
    if future_gates < 1:
        raise ValueError("future_gates must be positive")
    rows = np.stack([
        track.flight_plan(index, future_gates)["records"]
        for index in range(len(track.gates))
    ]).astype(np.float64)
    normalized = _normalize_records(
        rows,
        position_scale_m=position_scale_m,
        aperture_scale_m=aperture_scale_m,
    )
    flattened = normalized.reshape(len(track.gates), -1)
    ordered = _periodic_interpolate(flattened, int(phase_samples))
    # Retain both occupancy and order.  Scaling by width prevents a longer
    # route horizon from changing distance merely by adding coordinates.
    width = flattened.shape[1]
    global_summary = np.concatenate([
        flattened.mean(axis=0),
        flattened.std(axis=0),
        np.quantile(flattened, (0.10, 0.50, 0.90), axis=0).reshape(-1),
    ]) / np.sqrt(5.0 * width)
    ordered_summary = ordered.reshape(-1) / np.sqrt(ordered.size)
    embedding = np.concatenate([
        global_summary / np.sqrt(2.0),
        ordered_summary / np.sqrt(2.0),
    ])
    return RouteGrammarProfile(
        name=track.name,
        phases=np.arange(len(track.gates), dtype=np.float64) / len(track.gates),
        records=flattened / np.sqrt(width),
        embedding=embedding,
        gate_count=len(track.gates),
        future_gates=future_gates,
    )


def route_grammar_alignment(
    source: RouteGrammarProfile,
    target: RouteGrammarProfile,
) -> PhaseAlignment:
    """Monotonically align contract-faithful route phases with DTW."""

    if source.records.shape[1] != target.records.shape[1]:
        raise ValueError("route grammar profiles use different contracts")
    return align_phase_signatures(
        source.phases, source.records,
        target.phases, target.records,
    )


def route_directed_position(
    source: RouteGrammarProfile,
    target: RouteGrammarProfile,
    candidate: RouteGrammarProfile,
) -> RouteDirectedPosition:
    """Project a candidate onto the frozen route-contract transfer direction."""

    if not (
        source.embedding.shape == target.embedding.shape == candidate.embedding.shape
    ):
        raise ValueError("route grammar embeddings use different contracts")
    direction = target.embedding - source.embedding
    squared_distance = float(direction @ direction)
    if squared_distance <= 1.0e-16:
        raise ValueError("source and target route embeddings are indistinguishable")
    offset = candidate.embedding - source.embedding
    progress = float(offset @ direction / squared_distance)
    residual = offset - progress * direction
    source_distance = float(np.linalg.norm(offset))
    target_distance = float(np.linalg.norm(candidate.embedding - target.embedding))
    baseline_target_distance = float(np.sqrt(squared_distance))
    return RouteDirectedPosition(
        progress=progress,
        off_axis_ratio=float(np.linalg.norm(residual) / baseline_target_distance),
        source_distance=source_distance,
        target_distance=target_distance,
        target_gain_fraction=float(1.0 - target_distance / baseline_target_distance),
    )
