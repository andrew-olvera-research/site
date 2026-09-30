"""Invariant descriptors for a spatial racing-course manifold.

The descriptor is intentionally interpretable.  Scalar features describe the
physical task while a fixed-length ordered signature retains where demanding
transitions occur around a lap.  Global translation and yaw are nuisance
variables and therefore disappear from both blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from ..racing_distribution.descriptors import (
    CourseDescriptor,
    course_descriptor,
    transition_descriptors,
)
from ..tracks import Track


F64 = NDArray[np.float64]

FEATURE_NAMES = (
    "length_m",
    "gate_count",
    "gate_density_per_100m",
    "minimum_gate_spacing_m",
    "maximum_gate_spacing_m",
    "spacing_coefficient_of_variation",
    "vertical_excursion_m",
    "vertical_travel_fraction",
    "p95_absolute_slope",
    "p95_turn_degrees",
    "maximum_turn_degrees",
    "turn_alternation_fraction",
    "turn_direction_imbalance",
    "p99_curvature_m_inv",
    "maximum_curvature_m_inv",
    "curve_p95_curvature_m_inv",
    "curve_p95_absolute_torsion_m_inv",
    "p95_absolute_torsion_proxy",
    "hard_transition_fraction",
    "close_transition_fraction",
    "close_hard_transition_fraction",
    "mean_gate_normal_mismatch_degrees",
    "maximum_gate_up_change_degrees",
    "minimum_nonadjacent_spacing_m",
)

# Physical fallback scales stop a tiny empirical spread in a small atlas from
# becoming an enormous standardized distance.  Atlas fitting may increase a
# scale, but never reduce it below one quarter of these engineering units.
DEFAULT_FEATURE_SCALES = np.asarray(
    [
        80.0, 10.0, 12.0, 4.0, 12.0, 0.45, 4.0, 0.30, 0.35, 65.0,
        100.0, 0.45, 0.55, 0.45, 0.90, 0.45, 0.35, 0.40, 0.40, 0.40,
        0.30, 35.0, 80.0, 4.0,
    ],
    np.float64,
)

# First nine columns of transition_descriptors, in the same order as the
# racing-distribution contract.  Dividing before interpolation makes the
# ordered block dimensionless and block balancing in the atlas predictable.
TRANSITION_SCALES = np.asarray(
    [10.0, 90.0, 90.0, 2.5, 0.5, 40.0, 90.0, 0.6, 0.4], np.float64
)


@dataclass(frozen=True, slots=True)
class TrackGeometryProfile:
    """Scalar and ordered geometry for one course."""

    name: str
    fingerprint: str
    feature_names: tuple[str, ...]
    feature_vector: F64
    ordered_signature: F64
    transition_features: F64
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        vector = np.asarray(self.feature_vector, np.float64)
        signature = np.asarray(self.ordered_signature, np.float64)
        transitions = np.asarray(self.transition_features, np.float64)
        if vector.shape != (len(self.feature_names),):
            raise ValueError("feature vector does not match feature-name contract")
        if signature.ndim != 2 or signature.shape[1] != len(TRANSITION_SCALES):
            raise ValueError("ordered signature must have shape [phase, 9]")
        if transitions.ndim != 2 or transitions.shape[1] < 9:
            raise ValueError("transition features must have at least nine columns")
        if not (
            np.all(np.isfinite(vector))
            and np.all(np.isfinite(signature))
            and np.all(np.isfinite(transitions))
        ):
            raise ValueError("geometry profile values must be finite")
        object.__setattr__(self, "feature_vector", vector)
        object.__setattr__(self, "ordered_signature", signature)
        object.__setattr__(self, "transition_features", transitions)

    def to_mapping(self, *, include_arrays: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "fingerprint": self.fingerprint,
            "features": {
                name: float(value)
                for name, value in zip(self.feature_names, self.feature_vector)
            },
        }
        if include_arrays:
            payload["ordered_signature"] = self.ordered_signature.tolist()
            payload["transition_features"] = self.transition_features.tolist()
        return payload


def ordered_transition_signature(
    track: Track, *, phase_samples: int = 64,
) -> F64:
    """Interpolate ordered local geometry onto a fixed lap-phase grid."""

    if phase_samples < 8:
        raise ValueError("phase_samples must be at least eight")
    local = transition_descriptors(track)[:, :9].astype(np.float64)
    count = len(local)
    source_phase = np.arange(count + 1, dtype=np.float64) / count
    target_phase = np.arange(phase_samples, dtype=np.float64) / phase_samples
    closed = np.concatenate([local, local[:1]], axis=0)
    signature = np.column_stack([
        np.interp(target_phase, source_phase, closed[:, column])
        for column in range(local.shape[1])
    ])
    return signature / TRANSITION_SCALES


def _curve_differential_metrics(track: Track, samples: int = 512) -> tuple[float, float]:
    """Return arc-weighted curvature and torsion quantiles of a cubic curve."""

    points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
    if track.loop:
        points = np.concatenate([points, points[:1]], axis=0)
    chord = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(chord < 1.0e-6):
        return 0.0, 0.0
    parameter = np.concatenate([[0.0], np.cumsum(chord)])
    spline = CubicSpline(
        parameter,
        points,
        axis=0,
        bc_type="periodic" if track.loop else "not-a-knot",
    )
    query = np.linspace(parameter[0], parameter[-1], max(samples, 32 * len(points)))
    first = np.asarray(spline(query, 1), np.float64)
    second = np.asarray(spline(query, 2), np.float64)
    third = np.asarray(spline(query, 3), np.float64)
    cross = np.cross(first, second)
    first_norm = np.linalg.norm(first, axis=1)
    cross_norm_sq = np.sum(cross * cross, axis=1)
    curvature = np.linalg.norm(cross, axis=1) / np.maximum(first_norm**3, 1.0e-10)
    torsion = np.sum(cross * third, axis=1) / np.maximum(cross_norm_sq, 1.0e-10)
    weights = first_norm / max(float(first_norm.sum()), 1.0e-12)
    return _weighted_quantile(curvature, weights, 0.95), _weighted_quantile(
        np.abs(torsion), weights, 0.95
    )


def _weighted_quantile(values: F64, weights: F64, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= max(float(cumulative[-1]), 1.0e-12)
    return float(np.interp(quantile, cumulative, values[order]))


def _minimum_nonadjacent_spacing(track: Track) -> float:
    points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
    count = len(points)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    valid = np.ones((count, count), dtype=np.bool_)
    np.fill_diagonal(valid, False)
    for index in range(count):
        if index + 1 < count:
            valid[index, index + 1] = valid[index + 1, index] = False
        if track.loop:
            valid[index, (index + 1) % count] = False
            valid[(index + 1) % count, index] = False
    selected = distances[valid]
    return float(selected.min()) if len(selected) else float("inf")


def _feature_values(track: Track, base: CourseDescriptor) -> dict[str, float]:
    local = base.transition_features.astype(np.float64)
    lengths = local[:, 0]
    signed_turn = local[:, 1]
    turn_sign = np.sign(signed_turn)
    active = np.abs(signed_turn) >= 8.0
    active_sign = turn_sign[active]
    alternation = (
        float(np.mean(active_sign != np.roll(active_sign, 1)))
        if len(active_sign) > 1 else 0.0
    )
    turn_imbalance = float(
        abs(np.sum(signed_turn)) / max(float(np.sum(np.abs(signed_turn))), 1.0e-9)
    )
    curve_curvature, curve_torsion = _curve_differential_metrics(track)
    values = dict(base.values)
    values.update({
        "spacing_coefficient_of_variation": float(
            np.std(lengths) / max(float(np.mean(lengths)), 1.0e-9)
        ),
        "vertical_travel_fraction": float(
            np.sum(np.abs(local[:, 3])) / max(float(np.sum(lengths)), 1.0e-9)
        ),
        "p95_absolute_slope": float(np.quantile(local[:, 4], 0.95)),
        "turn_alternation_fraction": alternation,
        "turn_direction_imbalance": turn_imbalance,
        "curve_p95_curvature_m_inv": curve_curvature,
        "curve_p95_absolute_torsion_m_inv": curve_torsion,
        "p95_absolute_torsion_proxy": float(np.quantile(np.abs(local[:, 8]), 0.95)),
        "close_transition_fraction": float(np.mean(lengths <= 5.0)),
        "close_hard_transition_fraction": float(np.mean(
            (lengths <= 5.0) & ((local[:, 2] >= 55.0) | (local[:, 4] >= 0.28))
        )),
        "minimum_nonadjacent_spacing_m": _minimum_nonadjacent_spacing(track),
    })
    return values


def analyze_track_geometry(
    track: Track,
    *,
    phase_samples: int = 64,
    racing_line: Any | None = None,
    qualification: Mapping[str, Any] | None = None,
) -> TrackGeometryProfile:
    """Build the invariant scalar-plus-sequence geometry representation."""

    base = course_descriptor(
        track, racing_line=racing_line, qualification=qualification
    )
    values = _feature_values(track, base)
    vector = np.asarray([values[name] for name in FEATURE_NAMES], np.float64)
    return TrackGeometryProfile(
        name=track.name,
        fingerprint=track.fingerprint,
        feature_names=FEATURE_NAMES,
        feature_vector=vector,
        ordered_signature=ordered_transition_signature(
            track, phase_samples=phase_samples
        ),
        transition_features=base.transition_features,
        values=values,
    )
