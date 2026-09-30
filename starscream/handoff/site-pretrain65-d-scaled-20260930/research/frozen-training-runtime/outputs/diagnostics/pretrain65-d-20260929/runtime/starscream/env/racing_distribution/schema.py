"""Typed contracts for zero-shot racing-task generation.

The generator treats a course as a composition of maneuver primitives and a
continuous metric realization.  Real benchmark courses are intentionally not
part of this schema: a generator may be configured from physical limits and
training-only statistics, never from held-out geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class GeneratorBackend(str, Enum):
    """Track realization used by a controlled generator ablation."""

    INFORMED_SPLINE = "informed_spline"
    MANEUVER_GRAMMAR = "maneuver_grammar"


class ManeuverKind(str, Enum):
    """Reusable local flight problems, independent of world pose."""

    STRAIGHT = "straight"
    ACCELERATION = "acceleration"
    BRAKING = "braking"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    HAIRPIN_LEFT = "hairpin_left"
    HAIRPIN_RIGHT = "hairpin_right"
    SLALOM_LEFT = "slalom_left"
    SLALOM_RIGHT = "slalom_right"
    CLIMB = "climb"
    DIVE = "dive"
    SPLIT_S_LEFT = "split_s_left"
    SPLIT_S_RIGHT = "split_s_right"
    CORKSCREW_LEFT = "corkscrew_left"
    CORKSCREW_RIGHT = "corkscrew_right"
    STACKED_REVERSAL = "stacked_reversal"


HARD_MANEUVERS = frozenset({
    ManeuverKind.HAIRPIN_LEFT,
    ManeuverKind.HAIRPIN_RIGHT,
    ManeuverKind.SPLIT_S_LEFT,
    ManeuverKind.SPLIT_S_RIGHT,
    ManeuverKind.CORKSCREW_LEFT,
    ManeuverKind.CORKSCREW_RIGHT,
    ManeuverKind.STACKED_REVERSAL,
})

VERTICAL_MANEUVERS = frozenset({
    ManeuverKind.CLIMB,
    ManeuverKind.DIVE,
    ManeuverKind.SPLIT_S_LEFT,
    ManeuverKind.SPLIT_S_RIGHT,
    ManeuverKind.CORKSCREW_LEFT,
    ManeuverKind.CORKSCREW_RIGHT,
    ManeuverKind.STACKED_REVERSAL,
})


@dataclass(frozen=True, slots=True)
class ManeuverSpec:
    """One macro in a compositional course program."""

    kind: ManeuverKind
    length_scale: float = 1.0
    vertical_scale: float = 1.0
    severity: float = 0.5

    def __post_init__(self) -> None:
        if not 0.55 <= float(self.length_scale) <= 1.65:
            raise ValueError("maneuver length_scale must be in [0.55, 1.65]")
        if not 0.45 <= float(self.vertical_scale) <= 1.65:
            raise ValueError("maneuver vertical_scale must be in [0.45, 1.65]")
        if not 0.0 <= float(self.severity) <= 1.0:
            raise ValueError("maneuver severity must be in [0, 1]")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "length_scale": float(self.length_scale),
            "vertical_scale": float(self.vertical_scale),
            "severity": float(self.severity),
        }


@dataclass(frozen=True, slots=True)
class CourseProgram:
    """Pose-independent ordered maneuver composition."""

    maneuvers: tuple[ManeuverSpec, ...]
    seed: int
    split: str
    backend: GeneratorBackend = GeneratorBackend.MANEUVER_GRAMMAR
    program_id: str = ""

    def __post_init__(self) -> None:
        if len(self.maneuvers) < 4:
            raise ValueError("course programs require at least four maneuvers")
        if self.split not in {"train", "validation", "composition_holdout"}:
            raise ValueError(f"unsupported course-program split {self.split!r}")

    @property
    def kinds(self) -> tuple[ManeuverKind, ...]:
        return tuple(item.kind for item in self.maneuvers)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "maneuvers": [item.to_mapping() for item in self.maneuvers],
            "program_id": self.program_id,
            "seed": int(self.seed),
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class RacingDistributionConfig:
    """Physical envelope shared by both generator backends."""

    minimum_macros: int = 6
    maximum_macros: int = 10
    minimum_gates: int = 12
    maximum_gates: int = 28
    minimum_length_m: float = 55.0
    maximum_length_m: float = 155.0
    minimum_gate_spacing_m: float = 1.8
    maximum_gate_spacing_m: float = 22.0
    minimum_nonadjacent_gate_spacing_m: float = 1.25
    minimum_route_clearance_m: float = 0.75
    minimum_altitude_m: float = 1.0
    maximum_altitude_m: float = 9.5
    gate_width_range_m: tuple[float, float] = (1.75, 2.50)
    gate_height_range_m: tuple[float, float] = (1.65, 2.35)
    bounds_xy_m: tuple[float, float] = (22.0, 35.0)
    bounds_margin_m: float = 6.0
    spline_control_points: tuple[int, int] = (7, 12)
    spline_radial_range_m: tuple[float, float] = (9.0, 22.0)
    spline_vertical_amplitude_m: tuple[float, float] = (0.3, 3.5)
    spline_control_sampling_mode: str = "radial_sorted"
    spline_control_order_mode: str = "sampled"
    spline_minimum_control_point_spacing_m: float = 2.5
    spline_fit_mode: str = "periodic_cubic"
    spline_arc_spacing_mode: str = "bounded_lognormal"
    spline_gate_orientation_mode: str = "tangent_3d"
    spline_smoothing: float = 0.0
    gate_normal_jitter_degrees: float = 8.0
    gate_roll_range_degrees: tuple[float, float] = (-35.0, 35.0)
    generation_attempts: int = 320
    racing_line_samples: int = 700
    racing_line_offset_iterations: int = 8
    maximum_p99_curvature_m_inv: float = 1.20
    maximum_peak_curvature_m_inv: float = 2.20
    maximum_closure_correction_fraction: float = 0.42
    maximum_segment_turn_degrees: float = 178.0
    required_ngram_order: int = 3
    minimum_train_ngram_coverage: float = 0.72
    minimum_validation_ngram_support: float = 0.90
    minimum_vertical_track_fraction: float = 0.45
    minimum_hard_track_fraction: float = 0.45
    minimum_qd_occupancy_fraction: float = 0.35
    maximum_dynamic_qualification_yield_gap: float = 0.40
    qd_edges: Mapping[str, tuple[float, ...]] = field(default_factory=lambda: {
        "length_m": (70.0, 90.0, 115.0, 140.0),
        "vertical_excursion_m": (1.0, 2.5, 4.5, 6.5),
        "p95_turn_degrees": (25.0, 45.0, 70.0, 105.0),
        "gate_density_per_100m": (10.0, 14.0, 19.0, 25.0),
        "p99_curvature_m_inv": (0.15, 0.30, 0.55, 0.85),
        "qualified_speed_mps": (10.0, 13.0, 16.0, 19.0),
        "lateral_demand_ratio": (0.20, 0.45, 0.75, 1.05),
    })

    def __post_init__(self) -> None:
        if not 4 <= self.minimum_macros <= self.maximum_macros:
            raise ValueError("invalid macro-count range")
        if not 6 <= self.minimum_gates <= self.maximum_gates:
            raise ValueError("invalid gate-count range")
        if not 0.0 < self.minimum_length_m < self.maximum_length_m:
            raise ValueError("invalid course-length range")
        if not 0.0 < self.minimum_gate_spacing_m < self.maximum_gate_spacing_m:
            raise ValueError("invalid gate-spacing range")
        if self.required_ngram_order not in {2, 3}:
            raise ValueError("required_ngram_order must be two or three")
        if self.spline_control_sampling_mode not in {
            "radial_sorted", "bounded_uniform",
        }:
            raise ValueError("unknown spline_control_sampling_mode")
        if self.spline_control_order_mode not in {"sampled", "azimuth_sorted"}:
            raise ValueError("unknown spline_control_order_mode")
        if (
            not np.isfinite(self.spline_minimum_control_point_spacing_m)
            or self.spline_minimum_control_point_spacing_m <= 0.0
        ):
            raise ValueError(
                "spline minimum control-point spacing must be finite and positive"
            )
        if self.spline_fit_mode not in {"periodic_cubic", "periodic_splprep"}:
            raise ValueError("unknown spline_fit_mode")
        if self.spline_arc_spacing_mode not in {"bounded_lognormal", "equal_arc"}:
            raise ValueError("unknown spline_arc_spacing_mode")
        if self.spline_gate_orientation_mode not in {"tangent_3d", "yaw_tangent"}:
            raise ValueError("unknown spline_gate_orientation_mode")
        if not np.isfinite(self.spline_smoothing) or self.spline_smoothing < 0.0:
            raise ValueError("spline_smoothing must be finite and non-negative")
        for name, edges in self.qd_edges.items():
            if not edges or any(right <= left for left, right in zip(edges, edges[1:])):
                raise ValueError(f"QD edges for {name!r} must be strictly increasing")

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["qd_edges"] = {
            str(name): [float(item) for item in edges]
            for name, edges in self.qd_edges.items()
        }
        return payload
