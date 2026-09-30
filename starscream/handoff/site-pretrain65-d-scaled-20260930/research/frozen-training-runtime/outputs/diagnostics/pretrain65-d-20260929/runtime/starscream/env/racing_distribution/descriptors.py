"""Pose-invariant geometric and teacher-trajectory descriptors.

Static gate coordinates are insufficient for distribution design.  These
descriptors retain ordered transition geometry and, when MPCC qualification is
available, add pace, saturation, recovery, and clearance information from the
teacher's near-time-optimal trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..tracks import Track
from .grammar import cyclic_ngrams


F64 = NDArray[np.float64]
TRANSITION_COLUMNS = (
    "segment_length_m", "signed_yaw_turn_degrees", "turn_3d_degrees",
    "elevation_change_m", "absolute_slope", "gate_normal_mismatch_degrees",
    "gate_up_change_degrees", "curvature_proxy_m_inv", "torsion_proxy",
    "lap_phase",
)


@dataclass(frozen=True, slots=True)
class CourseDescriptor:
    """Compact quality-diversity descriptor plus ordered local features."""

    values: Mapping[str, float]
    transition_features: F64
    primitive_labels: tuple[str, ...]
    primitive_bigrams: tuple[tuple[str, ...], ...]
    primitive_trigrams: tuple[tuple[str, ...], ...]
    dynamic_available: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "dynamic_available": bool(self.dynamic_available),
            "primitive_bigrams": [list(item) for item in self.primitive_bigrams],
            "primitive_labels": list(self.primitive_labels),
            "primitive_trigrams": [list(item) for item in self.primitive_trigrams],
            "transition_columns": list(TRANSITION_COLUMNS),
            "transition_features": self.transition_features.tolist(),
            "values": {str(key): float(value) for key, value in self.values.items()},
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CourseDescriptor":
        columns = tuple(str(item) for item in payload.get("transition_columns", ()))
        if columns and columns != TRANSITION_COLUMNS:
            raise ValueError("course descriptor transition-column contract drift")
        return cls(
            values={str(k): float(v) for k, v in dict(payload["values"]).items()},
            transition_features=np.asarray(payload["transition_features"], np.float64),
            primitive_labels=tuple(str(item) for item in payload.get("primitive_labels", ())),
            primitive_bigrams=tuple(
                tuple(str(value) for value in item)
                for item in payload.get("primitive_bigrams", ())
            ),
            primitive_trigrams=tuple(
                tuple(str(value) for value in item)
                for item in payload.get("primitive_trigrams", ())
            ),
            dynamic_available=bool(payload.get("dynamic_available", False)),
        )


def _unit(rows: F64) -> F64:
    return rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1.0e-12)


def transition_descriptors(track: Track) -> F64:
    """Return ordered local descriptors invariant to XY translation and yaw."""

    points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
    normals = np.stack([gate.normal for gate in track.gates]).astype(np.float64)
    ups = np.stack([gate.up for gate in track.gates]).astype(np.float64)
    segments = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(segments, axis=1)
    units = _unit(segments)
    incoming = np.roll(units, 1, axis=0)
    dot = np.clip(np.sum(incoming * units, axis=1), -1.0, 1.0)
    turns_3d = np.degrees(np.arccos(dot))
    incoming_xy = _unit(incoming[:, :2])
    outgoing_xy = _unit(units[:, :2])
    cross_z = incoming_xy[:, 0] * outgoing_xy[:, 1] - incoming_xy[:, 1] * outgoing_xy[:, 0]
    dot_xy = np.clip(np.sum(incoming_xy * outgoing_xy, axis=1), -1.0, 1.0)
    signed_yaw = np.degrees(np.arctan2(cross_z, dot_xy))
    slope = np.abs(segments[:, 2]) / np.maximum(lengths, 1.0e-12)
    mismatch = np.degrees(np.arccos(np.clip(
        np.abs(np.sum(normals * incoming, axis=1)), 0.0, 1.0,
    )))
    up_change = np.degrees(np.arccos(np.clip(
        np.sum(ups * np.roll(ups, 1, axis=0), axis=1), -1.0, 1.0,
    )))
    local_scale = 0.5 * (lengths + np.roll(lengths, 1))
    curvature = 2.0 * np.sin(np.radians(turns_3d) * 0.5) / np.maximum(local_scale, 1.0e-9)
    previous = np.roll(incoming, 1, axis=0)
    torsion = np.einsum("ij,ij->i", np.cross(previous, incoming), units)
    phase = np.arange(len(points), dtype=np.float64) / max(len(points), 1)
    return np.column_stack([
        lengths, signed_yaw, turns_3d, segments[:, 2], slope, mismatch,
        up_change, curvature, torsion, phase,
    ])


def _qualification_values(
    qualification: Mapping[str, Any] | None,
    *, p99_curvature: float, maximum_lateral_acceleration_mps2: float,
) -> tuple[dict[str, float], bool]:
    selected = dict((qualification or {}).get("selected") or {})
    speed = selected.get("speed_mps")
    if speed is None:
        return {
            "qualified_speed_mps": 0.0,
            "teacher_elapsed_time_s": 0.0,
            "teacher_average_speed_mps": 0.0,
            "teacher_solver_failure_fraction": 0.0,
            "teacher_recovery_fraction": 0.0,
            "teacher_minimum_body_clearance_m": 0.0,
            "teacher_collective_saturation_fraction": 0.0,
            "teacher_body_rate_saturation_fraction": 0.0,
            "lateral_demand_ratio": 0.0,
        }, False
    speed_value = float(speed)
    scenarios = list(selected.get("scenarios") or ())

    def aggregate(name: str, default: float = 0.0, reduction: str = "mean") -> float:
        values = [float(item[name]) for item in scenarios if item.get(name) is not None]
        if not values:
            raw = selected.get(name)
            return default if raw is None else float(raw)
        if reduction == "min":
            return float(min(values))
        if reduction == "max":
            return float(max(values))
        return float(np.mean(values))

    elapsed = float(selected.get("mean_elapsed_time", aggregate("elapsed_time")))
    return {
        "qualified_speed_mps": speed_value,
        "teacher_elapsed_time_s": elapsed,
        "teacher_average_speed_mps": float(
            (qualification or {}).get("racing_line_length_m", 0.0) / max(elapsed, 1.0e-9)
        ),
        "teacher_solver_failure_fraction": aggregate("solver_failure_fraction"),
        "teacher_recovery_fraction": aggregate("recovery_fraction"),
        "teacher_minimum_body_clearance_m": aggregate(
            "minimum_body_clearance", reduction="min"
        ),
        "teacher_collective_saturation_fraction": aggregate(
            "collective_saturation_fraction"
        ),
        "teacher_body_rate_saturation_fraction": aggregate(
            "body_rate_saturation_fraction"
        ),
        "lateral_demand_ratio": float(
            speed_value * speed_value * p99_curvature
            / max(maximum_lateral_acceleration_mps2, 1.0e-9)
        ),
    }, True


def course_descriptor(
    track: Track, *, racing_line: Any | None = None,
    qualification: Mapping[str, Any] | None = None,
    maximum_lateral_acceleration_mps2: float = 30.0,
) -> CourseDescriptor:
    """Describe geometry and optional MPCC behavior with one stable contract."""

    local = transition_descriptors(track)
    points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
    if racing_line is None:
        length = float(np.sum(local[:, 0]))
        curvature_samples = local[:, 7]
    else:
        length = float(racing_line.length)
        query = np.linspace(0.0, racing_line.length, 2400, endpoint=False)
        curvature_samples = np.asarray(racing_line.evaluate(query)["curvature"], np.float64)
    p99_curvature = float(np.quantile(curvature_samples, 0.99))
    metadata = dict(track.metadata or {})
    # Grammar courses retain the semantic macro program.  Expanded gate labels
    # are useful for rendering but would turn one split-S into four unrelated
    # tokens and invalidate topology-coverage measurements.
    labels = tuple(str(item) for item in metadata.get(
        "macro_labels", metadata.get("primitive_labels", [gate.name for gate in track.gates])
    ))
    values: dict[str, float] = {
        "length_m": length,
        "gate_count": float(len(track.gates)),
        "gate_density_per_100m": float(100.0 * len(track.gates) / max(length, 1.0e-9)),
        "minimum_gate_spacing_m": float(np.min(local[:, 0])),
        "maximum_gate_spacing_m": float(np.max(local[:, 0])),
        "vertical_excursion_m": float(np.ptp(points[:, 2])),
        "mean_absolute_elevation_change_m": float(np.mean(np.abs(local[:, 3]))),
        "p95_turn_degrees": float(np.quantile(local[:, 2], 0.95)),
        "maximum_turn_degrees": float(np.max(local[:, 2])),
        "p99_curvature_m_inv": p99_curvature,
        "maximum_curvature_m_inv": float(np.max(curvature_samples)),
        "mean_gate_normal_mismatch_degrees": float(np.mean(local[:, 5])),
        "maximum_gate_up_change_degrees": float(np.max(local[:, 6])),
        "hard_transition_fraction": float(np.mean(
            (local[:, 2] >= 55.0) | (local[:, 4] >= 0.28)
        )),
    }
    dynamics, available = _qualification_values(
        qualification, p99_curvature=p99_curvature,
        maximum_lateral_acceleration_mps2=maximum_lateral_acceleration_mps2,
    )
    values.update(dynamics)
    return CourseDescriptor(
        values=values,
        transition_features=local,
        primitive_labels=labels,
        primitive_bigrams=cyclic_ngrams(labels, 2),
        primitive_trigrams=cyclic_ngrams(labels, 3),
        dynamic_available=available,
    )


def descriptor_distance(
    left: CourseDescriptor, right: CourseDescriptor,
    *, fields: Sequence[str] | None = None,
) -> float:
    """Scaled Euclidean distance for novelty and nearest-support audits."""

    selected = tuple(fields or (
        "length_m", "gate_density_per_100m", "vertical_excursion_m",
        "p95_turn_degrees", "p99_curvature_m_inv", "hard_transition_fraction",
        "qualified_speed_mps", "lateral_demand_ratio",
    ))
    scales = {
        "length_m": 100.0, "gate_density_per_100m": 20.0,
        "vertical_excursion_m": 5.0, "p95_turn_degrees": 90.0,
        "p99_curvature_m_inv": 0.7, "hard_transition_fraction": 0.5,
        "qualified_speed_mps": 16.0, "lateral_demand_ratio": 1.0,
    }
    delta = np.asarray([
        (float(left.values.get(name, 0.0)) - float(right.values.get(name, 0.0)))
        / scales.get(name, 1.0)
        for name in selected
    ], np.float64)
    return float(np.linalg.norm(delta))
