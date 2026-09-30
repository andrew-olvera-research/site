"""Directed geometry coordinates for policy-coupled manifold continuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .atlas import ManifoldAtlas
from .descriptors import TrackGeometryProfile
from .generator import CONTROLLABLE_FEATURE_NAMES


F64 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DirectedPosition:
    """Position of one geometry relative to a frozen source-to-target ray."""

    progress: float
    off_axis_distance: float
    target_distance: float
    descriptor_target_distance: float
    ordered_target_distance: float

    def to_mapping(self) -> dict[str, float]:
        return {
            "progress": self.progress,
            "off_axis_distance": self.off_axis_distance,
            "target_distance": self.target_distance,
            "descriptor_target_distance": self.descriptor_target_distance,
            "ordered_target_distance": self.ordered_target_distance,
        }


@dataclass(frozen=True, slots=True)
class FeatureShift:
    name: str
    source: float
    target: float
    absolute_delta: float
    standardized_delta: float
    controllable_by_local_spline: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "target": self.target,
            "absolute_delta": self.absolute_delta,
            "standardized_delta": self.standardized_delta,
            "controllable_by_local_spline": self.controllable_by_local_spline,
        }


class DirectedTransitionModel:
    """Frozen geometric model for moving from one course toward another.

    ``progress`` projects onto the complete block-balanced atlas ray: zero is
    the source and one is the target. ``off_axis_distance`` exposes generator
    drift that a plain reduction in target distance can hide. Policy success,
    retention, and MPCC validity intentionally remain external constraints;
    geometry alone must never admit a curriculum candidate.
    """

    def __init__(
        self,
        atlas: ManifoldAtlas,
        source: TrackGeometryProfile,
        target: TrackGeometryProfile,
    ) -> None:
        if source.feature_names != target.feature_names:
            raise ValueError("source and target descriptor contracts differ")
        self.atlas = atlas
        self.source = source
        self.target = target
        self._source_vector = atlas.representation_vector(source)
        self._target_vector = atlas.representation_vector(target)
        self._ray = self._target_vector - self._source_vector
        self._ray_norm_sq = float(self._ray @ self._ray)
        if self._ray_norm_sq <= 1.0e-12:
            raise ValueError("source and target are geometrically indistinguishable")

    @property
    def topology_change_required(self) -> bool:
        source_count = int(round(self.source.values["gate_count"]))
        target_count = int(round(self.target.values["gate_count"]))
        return source_count != target_count

    def feature_shifts(self) -> tuple[FeatureShift, ...]:
        rows = []
        for index, name in enumerate(self.source.feature_names):
            source = float(self.source.feature_vector[index])
            target = float(self.target.feature_vector[index])
            delta = target - source
            rows.append(FeatureShift(
                name=name,
                source=source,
                target=target,
                absolute_delta=delta,
                standardized_delta=float(delta / self.atlas.feature_scale[index]),
                controllable_by_local_spline=name in CONTROLLABLE_FEATURE_NAMES,
            ))
        return tuple(sorted(rows, key=lambda row: abs(row.standardized_delta), reverse=True))

    def position(self, profile: TrackGeometryProfile) -> DirectedPosition:
        vector = self.atlas.representation_vector(profile)
        displacement = vector - self._source_vector
        progress = float(displacement @ self._ray / self._ray_norm_sq)
        projection = self._source_vector + progress * self._ray
        target, descriptor, ordered = self.atlas.distance_between(profile, self.target)
        return DirectedPosition(
            progress=progress,
            off_axis_distance=float(np.linalg.norm(vector - projection)),
            target_distance=target,
            descriptor_target_distance=descriptor,
            ordered_target_distance=ordered,
        )

    def to_mapping(self) -> dict[str, Any]:
        source_count = int(round(self.source.values["gate_count"]))
        target_count = int(round(self.target.values["gate_count"]))
        total, descriptor, ordered = self.atlas.distance_between(self.source, self.target)
        return {
            "source": self.source.name,
            "target": self.target.name,
            "source_gate_count": source_count,
            "target_gate_count": target_count,
            "topology_change_required": self.topology_change_required,
            "distance": {
                "total": total,
                "descriptor": descriptor,
                "ordered": ordered,
            },
            "feature_shifts": [row.to_mapping() for row in self.feature_shifts()],
            "continuation_contract": {
                "continuous_stage": "locally deform source while preserving source gate count",
                "topology_stage": (
                    "explicit gate merge/deletion is required"
                    if self.topology_change_required else "none"
                ),
                "candidate_constraints": [
                    "local geometry valid",
                    "MPCC-qualified at requested pace",
                    "policy success remains inside competence band",
                    "protected-source capability remains above retention floor",
                ],
            },
        }
