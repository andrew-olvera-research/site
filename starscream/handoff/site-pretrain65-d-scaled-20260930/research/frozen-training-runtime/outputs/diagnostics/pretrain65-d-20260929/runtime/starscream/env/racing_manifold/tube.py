"""Constrained course tubes for directional policy-manifold experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..tracks import Track
from .atlas import ManifoldAtlas
from .descriptors import TrackGeometryProfile, analyze_track_geometry
from .generator import CONTROLLABLE_FEATURE_NAMES, SplineManifoldGenerator
from .primitives import PrimitiveKind, graft_primitive
from .transition import DirectedPosition, DirectedTransitionModel


@dataclass(frozen=True, slots=True)
class CourseTubeConfig:
    """Geometry-only admission contract for one local source manifold tube."""

    candidate_count: int = 160
    train_continuous_count: int = 8
    validation_continuous_count: int = 4
    minimum_source_distance: float = 0.04
    maximum_source_distance: float = 1.10
    minimum_progress: float = 0.002
    maximum_progress: float = 0.12
    maximum_off_axis_distance: float = 1.05
    maximum_chain3_mean: float = 0.34
    maximum_chain6_mean: float = 0.38
    minimum_pairwise_distance: float = 0.025
    standardized_step_minimum: float = 0.05
    standardized_step_maximum: float = 0.32
    direction_fraction_minimum: float = 0.35
    direction_fraction_maximum: float = 0.85
    noise_scale: float = 0.40
    allow_topology_mismatch_direction_only: bool = False

    def __post_init__(self) -> None:
        if self.candidate_count < self.train_continuous_count + self.validation_continuous_count:
            raise ValueError("course tube candidate pool is too small")
        if not 0.0 <= self.direction_fraction_minimum <= self.direction_fraction_maximum <= 1.0:
            raise ValueError("direction fractions must lie in [0, 1]")
        if not 0.0 < self.standardized_step_minimum <= self.standardized_step_maximum:
            raise ValueError("invalid standardized step range")


@dataclass(frozen=True, slots=True)
class CourseTubeCandidate:
    track: Track
    profile: TrackGeometryProfile
    split: str
    stratum: str
    seed: int
    source_distance: float
    source_descriptor_distance: float
    source_ordered_distance: float
    position: DirectedPosition
    chain3_mean: float
    chain6_mean: float
    direction_alignment: float
    requested_step: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.track.name,
            "fingerprint": self.track.fingerprint,
            "split": self.split,
            "stratum": self.stratum,
            "seed": self.seed,
            "source_distance": self.source_distance,
            "source_descriptor_distance": self.source_descriptor_distance,
            "source_ordered_distance": self.source_ordered_distance,
            "position": self.position.to_mapping(),
            "chain3_mean": self.chain3_mean,
            "chain6_mean": self.chain6_mean,
            "direction_alignment": self.direction_alignment,
            "requested_step": self.requested_step,
        }


class DirectionalCourseTube:
    """Generate a local, auditable tube from source toward a remote target.

    The target defines a direction, not an endpoint.  This distinction is
    important when source and target have different gate-count topology: the
    continuous generator is only asked for small source-supported steps.
    MPCC qualification and policy competence remain external admission gates.
    """

    def __init__(
        self,
        source: Track,
        target: Track,
        atlas: ManifoldAtlas,
        *,
        config: CourseTubeConfig | None = None,
    ) -> None:
        self.source = source
        self.target = target
        self.atlas = atlas
        self.config = config or CourseTubeConfig()
        self.source_profile = analyze_track_geometry(source)
        self.target_profile = analyze_track_geometry(target)
        if (
            len(source.gates) != len(target.gates)
            and not self.config.allow_topology_mismatch_direction_only
        ):
            raise ValueError(
                "directional course tubes preserve source gate count; an "
                "explicit topology-normalized bridge is required before using "
                "a target with a different gate count"
            )
        self.transition = DirectedTransitionModel(
            atlas, self.source_profile, self.target_profile,
        )
        self.generator = SplineManifoldGenerator(
            source,
            smooth_passes=2,
            minimum_gate_spacing_m=1.5,
            minimum_nonadjacent_spacing_m=0.9,
            maximum_gate_displacement_m=0.65,
        )
        target_direction = (
            self.target_profile.feature_vector - self.source_profile.feature_vector
        ) / atlas.feature_scale
        controllable = np.asarray([
            name in CONTROLLABLE_FEATURE_NAMES
            for name in self.source_profile.feature_names
        ])
        target_direction[~controllable] = 0.0
        norm = float(np.linalg.norm(target_direction))
        if norm <= 1.0e-12:
            raise ValueError("source-to-target direction has no controllable coordinates")
        self._direction = target_direction / norm
        self._controllable = controllable

    def _candidate(
        self, *, seed: int, ordinal: int,
    ) -> CourseTubeCandidate | None:
        cfg = self.config
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=len(self._direction))
        noise[~self._controllable] = 0.0
        # Keep stochastic exploration orthogonal to the desired ray so the
        # direction coefficient has a stable interpretation.
        noise -= self._direction * float(noise @ self._direction)
        noise_norm = float(np.linalg.norm(noise))
        if noise_norm > 1.0e-12:
            noise /= noise_norm
        fraction = float(rng.uniform(
            cfg.direction_fraction_minimum, cfg.direction_fraction_maximum,
        ))
        direction = fraction * self._direction + cfg.noise_scale * (1.0 - fraction) * noise
        direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
        step = float(rng.uniform(
            cfg.standardized_step_minimum, cfg.standardized_step_maximum,
        ))
        proposal = self.generator.propose(
            self.atlas,
            standardized_direction=direction,
            standardized_step=step,
            ridge=0.12,
            name=f"swift_v60_tube_candidate_{ordinal:03d}_{seed}",
        )
        if not proposal.valid:
            return None
        profile = proposal.profile
        source_distance, descriptor, ordered = self.atlas.distance_between(
            self.source_profile, profile,
        )
        position = self.transition.position(profile)
        chain3 = self.atlas.transition_chain_support_profiles(
            profile, (self.source_profile,), horizon=3,
        )
        chain6 = self.atlas.transition_chain_support_profiles(
            profile, (self.source_profile,), horizon=6,
        )
        achieved = proposal.achieved_standardized_delta[self._controllable]
        requested = self._direction[self._controllable]
        alignment = float(
            achieved @ requested
            / max(float(np.linalg.norm(achieved) * np.linalg.norm(requested)), 1.0e-12)
        )
        if not (
            cfg.minimum_source_distance <= source_distance <= cfg.maximum_source_distance
            and cfg.minimum_progress <= position.progress <= cfg.maximum_progress
            and position.off_axis_distance <= cfg.maximum_off_axis_distance
            and float(chain3["mean_distance"]) <= cfg.maximum_chain3_mean
            and float(chain6["mean_distance"]) <= cfg.maximum_chain6_mean
            and alignment > 0.0
        ):
            return None
        metadata = dict(proposal.track.metadata or {})
        metadata["directional_course_tube"] = {
            "schema": "starscream-directional-course-tube-v1",
            "source": self.source.name,
            "target": self.target.name,
            "seed": int(seed),
            "source_distance": source_distance,
            "progress": position.progress,
            "off_axis_distance": position.off_axis_distance,
            "chain3_mean": float(chain3["mean_distance"]),
            "chain6_mean": float(chain6["mean_distance"]),
            "direction_alignment": alignment,
            "requested_step": step,
        }
        track = Track(
            name=proposal.track.name,
            gates=proposal.track.gates,
            bounds=proposal.track.bounds,
            loop=proposal.track.loop,
            metadata=metadata,
        )
        return CourseTubeCandidate(
            track=track,
            profile=profile,
            split="candidate",
            stratum="continuous",
            seed=int(seed),
            source_distance=source_distance,
            source_descriptor_distance=descriptor,
            source_ordered_distance=ordered,
            position=position,
            chain3_mean=float(chain3["mean_distance"]),
            chain6_mean=float(chain6["mean_distance"]),
            direction_alignment=alignment,
            requested_step=step,
        )

    def _select_diverse(
        self,
        candidates: Sequence[CourseTubeCandidate],
        *,
        count: int,
        excluded: Sequence[CourseTubeCandidate] = (),
    ) -> list[CourseTubeCandidate]:
        if count < 1:
            return []
        pool = list(candidates)
        selected = list(excluded)
        output: list[CourseTubeCandidate] = []
        # Start near the middle of the admitted directional tube, then use a
        # maximin design under the same frozen manifold metric.
        while pool and len(output) < count:
            if not selected:
                index = min(
                    range(len(pool)),
                    key=lambda item: abs(pool[item].position.progress - 0.045),
                )
            else:
                distances = [
                    min(
                        self.atlas.distance_between(row.profile, other.profile)[0]
                        for other in selected
                    )
                    for row in pool
                ]
                index = int(np.argmax(distances))
                if distances[index] < self.config.minimum_pairwise_distance:
                    break
            chosen = pool.pop(index)
            output.append(chosen)
            selected.append(chosen)
        if len(output) != count:
            raise RuntimeError(
                f"directional tube produced only {len(output)}/{count} diverse courses"
            )
        return output

    def generate(
        self, *, seed: int, excluded_names: Sequence[str] = (),
    ) -> tuple[CourseTubeCandidate, ...]:
        cfg = self.config
        excluded_name_set = {str(name) for name in excluded_names}
        candidates = [
            row for index in range(cfg.candidate_count)
            if (row := self._candidate(
                seed=seed + 104729 * index, ordinal=index,
            )) is not None and row.track.name not in excluded_name_set
        ]
        candidates.sort(key=lambda row: (
            row.position.progress, row.source_distance, row.track.name,
        ))
        required = cfg.train_continuous_count + cfg.validation_continuous_count
        if len(candidates) < required:
            raise RuntimeError(
                f"only {len(candidates)}/{required} candidates passed the tube contract"
            )
        train = self._select_diverse(candidates, count=cfg.train_continuous_count)
        remaining = [row for row in candidates if row not in train]
        validation = self._select_diverse(
            remaining, count=cfg.validation_continuous_count, excluded=train,
        )

        def assign(row: CourseTubeCandidate, split: str) -> CourseTubeCandidate:
            return CourseTubeCandidate(
                track=row.track, profile=row.profile, split=split,
                stratum=row.stratum, seed=row.seed,
                source_distance=row.source_distance,
                source_descriptor_distance=row.source_descriptor_distance,
                source_ordered_distance=row.source_ordered_distance,
                position=row.position, chain3_mean=row.chain3_mean,
                chain6_mean=row.chain6_mean,
                direction_alignment=row.direction_alignment,
                requested_step=row.requested_step,
            )

        return tuple(
            [assign(row, "train") for row in train]
            + [assign(row, "validation") for row in validation]
        )

    def graft_local_primitive(
        self,
        base: CourseTubeCandidate,
        kind: PrimitiveKind,
        *,
        anchor: int,
        severity: float,
        split: str,
        name: str,
    ) -> CourseTubeCandidate:
        track = graft_primitive(
            base.track, kind, anchor=anchor, severity=severity, name=name,
        )
        profile = analyze_track_geometry(track)
        source_distance, descriptor, ordered = self.atlas.distance_between(
            self.source_profile, profile,
        )
        position = self.transition.position(profile)
        chain3 = self.atlas.transition_chain_support_profiles(
            profile, (self.source_profile, base.profile), horizon=3,
        )
        chain6 = self.atlas.transition_chain_support_profiles(
            profile, (self.source_profile, base.profile), horizon=6,
        )
        # Primitive edits are allowed a wider local envelope, but remain
        # bounded and must preserve recognizable source-chain support.
        if (
            source_distance > 1.65
            or position.progress > 0.16
            or position.progress < -0.10
            or position.off_axis_distance > 1.65
            or float(chain3["mean_distance"]) > 0.48
            or float(chain6["mean_distance"]) > 0.52
        ):
            raise ValueError(f"{kind} graft escaped the local Swift tube")
        metadata = dict(track.metadata or {})
        metadata["directional_course_tube"] = {
            "schema": "starscream-directional-course-tube-v1",
            "source": self.source.name,
            "target": self.target.name,
            "primitive": kind,
            "source_distance": source_distance,
            "progress": position.progress,
            "off_axis_distance": position.off_axis_distance,
            "chain3_mean": float(chain3["mean_distance"]),
            "chain6_mean": float(chain6["mean_distance"]),
        }
        track = Track(
            name=track.name, gates=track.gates, bounds=track.bounds,
            loop=track.loop, metadata=metadata,
        )
        return CourseTubeCandidate(
            track=track, profile=profile, split=split, stratum=f"primitive_{kind}",
            seed=base.seed, source_distance=source_distance,
            source_descriptor_distance=descriptor,
            source_ordered_distance=ordered, position=position,
            chain3_mean=float(chain3["mean_distance"]),
            chain6_mean=float(chain6["mean_distance"]),
            direction_alignment=base.direction_alignment,
            requested_step=base.requested_step,
        )
