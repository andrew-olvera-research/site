"""Robust low-dimensional maps of a racing-course distribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .descriptors import (
    DEFAULT_FEATURE_SCALES,
    TRANSITION_SCALES,
    TrackGeometryProfile,
)


F64 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SupportDiagnostic:
    name: str
    nearest_name: str
    distance: float
    descriptor_distance: float
    ordered_distance: float
    inside_training_radius: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nearest_name": self.nearest_name,
            "distance": self.distance,
            "descriptor_distance": self.descriptor_distance,
            "ordered_distance": self.ordered_distance,
            "inside_training_radius": self.inside_training_radius,
        }


class ManifoldAtlas:
    """A block-balanced robust embedding with explicit support distances."""

    def __init__(
        self,
        profiles: Sequence[TrackGeometryProfile],
        *,
        components: int = 3,
        descriptor_weight: float = 0.58,
    ) -> None:
        if len(profiles) < 2:
            raise ValueError("a manifold atlas needs at least two tracks")
        if not 0.0 < descriptor_weight < 1.0:
            raise ValueError("descriptor_weight must be in (0, 1)")
        if len({profile.name for profile in profiles}) != len(profiles):
            raise ValueError("profile names must be unique")
        shape = profiles[0].ordered_signature.shape
        names = profiles[0].feature_names
        if any(
            profile.ordered_signature.shape != shape
            or profile.feature_names != names
            for profile in profiles
        ):
            raise ValueError("all profiles must share descriptor contracts")
        self.profiles = tuple(profiles)
        self.names = tuple(profile.name for profile in profiles)
        self.feature_names = names
        self.descriptor_weight = float(descriptor_weight)
        descriptor = np.stack([profile.feature_vector for profile in profiles])
        self.feature_center = np.median(descriptor, axis=0)
        mad = 1.4826 * np.median(
            np.abs(descriptor - self.feature_center[None, :]), axis=0
        )
        self.feature_scale = np.maximum(mad, 0.25 * DEFAULT_FEATURE_SCALES)
        signature = np.stack([profile.ordered_signature for profile in profiles])
        self.signature_center = np.median(signature, axis=0)
        signature_mad = 1.4826 * np.median(
            np.abs(signature - self.signature_center[None, :, :]), axis=0
        )
        self.signature_scale = np.maximum(signature_mad, 0.25)
        representation = np.stack([self._representation(item) for item in profiles])
        self.representation_center = representation.mean(axis=0)
        centered = representation - self.representation_center[None, :]
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
        rank = max(1, min(components, len(profiles) - 1, right.shape[0]))
        self.components = right[:rank]
        self.coordinates = centered @ self.components.T
        variance = singular**2
        self.explained_variance_ratio = (
            variance[:rank] / max(float(variance.sum()), 1.0e-12)
        )
        self.pairwise_distances = np.linalg.norm(
            representation[:, None, :] - representation[None, :, :], axis=-1
        )
        nonzero = self.pairwise_distances[
            ~np.eye(len(profiles), dtype=np.bool_)
        ]
        self.training_radius = float(np.quantile(nonzero, 0.90))

    def _blocks(self, profile: TrackGeometryProfile) -> tuple[F64, F64]:
        descriptor = (
            profile.feature_vector - self.feature_center
        ) / self.feature_scale
        ordered = (
            profile.ordered_signature - self.signature_center
        ) / self.signature_scale
        ordered = ordered.reshape(-1) / np.sqrt(ordered.size)
        return descriptor, ordered

    def _representation(self, profile: TrackGeometryProfile) -> F64:
        descriptor, ordered = self._blocks(profile)
        return np.concatenate([
            np.sqrt(self.descriptor_weight) * descriptor,
            np.sqrt(1.0 - self.descriptor_weight) * ordered,
        ])

    def representation_vector(self, profile: TrackGeometryProfile) -> F64:
        """Return the frozen block-balanced vector used by atlas distances."""

        return self._representation(profile).copy()

    def transform(self, profile: TrackGeometryProfile) -> F64:
        representation = self._representation(profile)
        return (representation - self.representation_center) @ self.components.T

    def standardized_descriptor(self, profile: TrackGeometryProfile) -> F64:
        return (profile.feature_vector - self.feature_center) / self.feature_scale

    def distances(self, profile: TrackGeometryProfile) -> tuple[F64, F64, F64]:
        query_descriptor, query_ordered = self._blocks(profile)
        descriptor_rows: list[float] = []
        ordered_rows: list[float] = []
        for reference in self.profiles:
            descriptor, ordered = self._blocks(reference)
            descriptor_rows.append(float(np.linalg.norm(query_descriptor - descriptor)))
            ordered_rows.append(float(np.linalg.norm(query_ordered - ordered)))
        descriptor_distance = np.asarray(descriptor_rows, np.float64)
        ordered_distance = np.asarray(ordered_rows, np.float64)
        total = np.sqrt(
            self.descriptor_weight * descriptor_distance**2
            + (1.0 - self.descriptor_weight) * ordered_distance**2
        )
        return total, descriptor_distance, ordered_distance

    def distance_between(
        self,
        left: TrackGeometryProfile,
        right: TrackGeometryProfile,
    ) -> tuple[float, float, float]:
        """Distance two arbitrary profiles under this atlas' frozen metric."""

        left_descriptor, left_ordered = self._blocks(left)
        right_descriptor, right_ordered = self._blocks(right)
        descriptor = float(np.linalg.norm(left_descriptor - right_descriptor))
        ordered = float(np.linalg.norm(left_ordered - right_ordered))
        total = float(np.sqrt(
            self.descriptor_weight * descriptor**2
            + (1.0 - self.descriptor_weight) * ordered**2
        ))
        return total, descriptor, ordered

    def support(self, profile: TrackGeometryProfile) -> SupportDiagnostic:
        return self.support_against(profile, self.names)

    def support_against(
        self,
        profile: TrackGeometryProfile,
        reference_names: Sequence[str],
    ) -> SupportDiagnostic:
        """Measure a query against an explicit represented course set."""

        if not reference_names:
            raise ValueError("reference_names cannot be empty")
        total, descriptor, ordered = self.distances(profile)
        indices = [self.names.index(name) for name in reference_names]
        index = min(indices, key=lambda item: total[item])
        reference_pairwise = self.pairwise_distances[np.ix_(indices, indices)]
        nonzero = reference_pairwise[~np.eye(len(indices), dtype=np.bool_)]
        radius = (
            float(np.quantile(nonzero, 0.90))
            if len(nonzero) else self.training_radius
        )
        return SupportDiagnostic(
            name=profile.name,
            nearest_name=self.names[index],
            distance=float(total[index]),
            descriptor_distance=float(descriptor[index]),
            ordered_distance=float(ordered[index]),
            inside_training_radius=bool(total[index] <= radius),
        )

    def support_against_profiles(
        self,
        profile: TrackGeometryProfile,
        references: Sequence[TrackGeometryProfile],
    ) -> SupportDiagnostic:
        """Measure a query against generated profiles under a frozen atlas."""

        if not references:
            raise ValueError("references cannot be empty")
        rows = [self.distance_between(profile, reference) for reference in references]
        index = int(np.argmin([row[0] for row in rows]))
        pairwise = np.asarray([
            [self.distance_between(left, right)[0] for right in references]
            for left in references
        ], np.float64)
        nonzero = pairwise[~np.eye(len(references), dtype=np.bool_)]
        radius = float(np.quantile(nonzero, 0.90)) if len(nonzero) else self.training_radius
        total, descriptor, ordered = rows[index]
        return SupportDiagnostic(
            name=profile.name,
            nearest_name=references[index].name,
            distance=total,
            descriptor_distance=descriptor,
            ordered_distance=ordered,
            inside_training_radius=bool(total <= radius),
        )

    @staticmethod
    def _transition_windows(profile: TrackGeometryProfile, horizon: int) -> F64:
        local = profile.transition_features[:, :9] / TRANSITION_SCALES
        rows = [
            np.concatenate([
                local[(start + offset) % len(local)]
                for offset in range(horizon)
            ])
            for start in range(len(local))
        ]
        return np.stack(rows) / np.sqrt(horizon * local.shape[1])

    def transition_chain_support(
        self,
        profile: TrackGeometryProfile,
        reference_names: Sequence[str],
        *,
        horizon: int = 3,
    ) -> dict[str, Any]:
        """Nearest support for every ordered local transition chain."""

        if horizon < 1:
            raise ValueError("transition horizon must be positive")
        if not reference_names:
            raise ValueError("reference_names cannot be empty")
        references = [self.profiles[self.names.index(name)] for name in reference_names]
        return self.transition_chain_support_profiles(
            profile, references, horizon=horizon
        )

    def transition_chain_support_profiles(
        self,
        profile: TrackGeometryProfile,
        references: Sequence[TrackGeometryProfile],
        *,
        horizon: int = 3,
    ) -> dict[str, Any]:
        """Nearest local-chain support against arbitrary generated profiles."""

        if horizon < 1:
            raise ValueError("transition horizon must be positive")
        if not references:
            raise ValueError("references cannot be empty")
        query = self._transition_windows(profile, horizon)
        reference_rows: list[F64] = []
        reference_labels: list[tuple[str, int]] = []
        for reference in references:
            windows = self._transition_windows(reference, horizon)
            reference_rows.append(windows)
            reference_labels.extend(
                (reference.name, index + 1) for index in range(len(windows))
            )
        support = np.concatenate(reference_rows, axis=0)
        pairwise = np.linalg.norm(
            query[:, None, :] - support[None, :, :], axis=-1
        )
        nearest_index = np.argmin(pairwise, axis=1)
        nearest = pairwise[np.arange(len(query)), nearest_index]
        worst = np.argsort(nearest)[::-1][: min(5, len(nearest))]
        return {
            "name": profile.name,
            "horizon": horizon,
            "reference_names": [reference.name for reference in references],
            "p50_distance": float(np.quantile(nearest, 0.50)),
            "p90_distance": float(np.quantile(nearest, 0.90)),
            "maximum_distance": float(nearest.max()),
            "mean_distance": float(nearest.mean()),
            "per_start_gate_distance": nearest.tolist(),
            "worst_start_phases": [
                {
                    "gate": int(index + 1),
                    "distance": float(nearest[index]),
                    "nearest_track": reference_labels[int(nearest_index[index])][0],
                    "nearest_gate": reference_labels[int(nearest_index[index])][1],
                }
                for index in worst
            ],
        }

    def k_center_indices(self, count: int) -> tuple[int, ...]:
        """Select a compact rehearsal core with greedy farthest-point sampling."""

        if not 1 <= count <= len(self.profiles):
            raise ValueError("count must lie inside the atlas size")
        medoid = int(np.argmin(self.pairwise_distances.mean(axis=1)))
        selected = [medoid]
        while len(selected) < count:
            nearest = np.min(self.pairwise_distances[:, selected], axis=1)
            nearest[selected] = -np.inf
            selected.append(int(np.argmax(nearest)))
        return tuple(selected)

    def outward_direction(self, name: str) -> F64:
        """Return a unit descriptor-space ray from atlas centre through a track."""

        profile = self.profiles[self.names.index(name)]
        direction = self.standardized_descriptor(profile)
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12:
            raise ValueError(f"profile {name!r} lies at the robust atlas centre")
        return direction / norm

    def frontier_names(self, count: int = 3) -> tuple[str, ...]:
        radius = np.linalg.norm(self.coordinates, axis=1)
        order = np.argsort(radius)[::-1][: min(count, len(radius))]
        return tuple(self.names[index] for index in order)

    def to_mapping(self, *, include_profiles: bool = True) -> dict[str, Any]:
        component_drivers = []
        for component_index, component in enumerate(self.components):
            descriptor_loading = component[: len(self.feature_names)]
            order = np.argsort(np.abs(descriptor_loading))[::-1][:6]
            component_drivers.append({
                "component": component_index + 1,
                "features": [
                    {
                        "name": self.feature_names[index],
                        "loading": float(descriptor_loading[index]),
                    }
                    for index in order
                ],
            })
        payload: dict[str, Any] = {
            "schema": "starscream-racing-manifold-atlas-v1",
            "feature_names": list(self.feature_names),
            "descriptor_weight": self.descriptor_weight,
            "feature_center": self.feature_center.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "training_radius": self.training_radius,
            "component_drivers": component_drivers,
            "frontier_names": list(self.frontier_names()),
            "k_center_names": [
                self.names[index]
                for index in self.k_center_indices(min(3, len(self.names)))
            ],
            "tracks": [
                {
                    "name": name,
                    "coordinates": self.coordinates[index].tolist(),
                    "mean_neighbor_distance": float(
                        np.mean(np.delete(self.pairwise_distances[index], index))
                    ),
                }
                for index, name in enumerate(self.names)
            ],
            "pairwise_distances": {
                name: {
                    other: float(self.pairwise_distances[index, other_index])
                    for other_index, other in enumerate(self.names)
                }
                for index, name in enumerate(self.names)
            },
        }
        if include_profiles:
            payload["profiles"] = [
                profile.to_mapping(include_arrays=False) for profile in self.profiles
            ]
        return payload
