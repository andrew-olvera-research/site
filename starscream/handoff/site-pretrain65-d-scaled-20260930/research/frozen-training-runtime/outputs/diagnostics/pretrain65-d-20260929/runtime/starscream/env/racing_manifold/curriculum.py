"""Auditable curriculum expansion in traversal and primitive coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..tracks import Track
from .atlas import ManifoldAtlas
from .descriptors import TrackGeometryProfile, analyze_track_geometry
from .generator import ManifoldExtensionProposal, SplineManifoldGenerator
from .primitives import PrimitiveKind, analyze_primitives, graft_primitive, primitive_coverage


@dataclass(frozen=True, slots=True)
class CoverageSnapshot:
    represented_names: tuple[str, ...]
    global_mean: float
    global_p90: float
    global_maximum: float
    transition: dict[int, dict[str, float]]
    primitive: dict[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "represented_names": list(self.represented_names),
            "global": {
                "mean": self.global_mean,
                "p90": self.global_p90,
                "maximum": self.global_maximum,
            },
            "transition": {str(key): value for key, value in self.transition.items()},
            "primitive": self.primitive,
        }


@dataclass(frozen=True, slots=True)
class ExpansionStep:
    iteration: int
    axis: str
    source: str
    target: str
    candidate: Track
    accepted: bool
    reasons: tuple[str, ...]
    proposal: ManifoldExtensionProposal | None
    before: CoverageSnapshot
    after: CoverageSnapshot

    def to_mapping(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "axis": self.axis,
            "source": self.source,
            "target": self.target,
            "candidate": self.candidate.name,
            "candidate_fingerprint": self.candidate.fingerprint,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "proposal": None if self.proposal is None else self.proposal.to_mapping(),
            "before": self.before.to_mapping(),
            "after": self.after.to_mapping(),
        }


class ManifoldCurriculumHarness:
    """Expand a represented set toward fixed targets without moving the ruler."""

    def __init__(self, targets: Sequence[Track]) -> None:
        if len(targets) < 2:
            raise ValueError("curriculum analysis needs at least two target tracks")
        self.targets = tuple(targets)
        self.target_profiles = tuple(analyze_track_geometry(track) for track in targets)
        self.atlas = ManifoldAtlas(self.target_profiles)

    def center_index(self) -> int:
        return self.atlas.k_center_indices(1)[0]

    def coverage(self, represented: Sequence[Track]) -> CoverageSnapshot:
        if not represented:
            raise ValueError("represented set cannot be empty")
        profiles = tuple(analyze_track_geometry(track) for track in represented)
        global_distance = np.asarray([
            self.atlas.support_against_profiles(target, profiles).distance
            for target in self.target_profiles
        ], np.float64)
        transition: dict[int, dict[str, float]] = {}
        for horizon in (1, 3, 6):
            values = np.asarray([
                self.atlas.transition_chain_support_profiles(
                    target, profiles, horizon=horizon
                )["mean_distance"]
                for target in self.target_profiles
            ], np.float64)
            transition[horizon] = {
                "mean": float(values.mean()),
                "p90": float(np.quantile(values, 0.90)),
                "maximum": float(values.max()),
            }
        return CoverageSnapshot(
            represented_names=tuple(track.name for track in represented),
            global_mean=float(global_distance.mean()),
            global_p90=float(np.quantile(global_distance, 0.90)),
            global_maximum=float(global_distance.max()),
            transition=transition,
            primitive=primitive_coverage([analyze_primitives(track) for track in represented]),
        )

    def _farthest_target(self, profiles: Sequence[TrackGeometryProfile]) -> int:
        distance = [
            self.atlas.support_against_profiles(target, profiles).distance
            for target in self.target_profiles
        ]
        return int(np.argmax(distance))

    def _nearest_source(
        self, target: TrackGeometryProfile, profiles: Sequence[TrackGeometryProfile]
    ) -> int:
        return int(np.argmin([
            self.atlas.distance_between(target, profile)[0] for profile in profiles
        ]))

    def expand(
        self,
        *,
        iterations: int = 6,
        standardized_step: float = 0.18,
        primitives: Sequence[PrimitiveKind] = ("split_s", "inverted_gate", "corkscrew"),
        primitive_every: int = 2,
        primitive_severity: float = 0.55,
    ) -> tuple[tuple[Track, ...], tuple[ExpansionStep, ...]]:
        """Run a deterministic farthest-target expansion prototype.

        Geometry candidates must improve global or three-transition coverage.
        Primitive candidates are admitted on their separate coverage axis; they
        never masquerade as global traversal progress.
        """

        if iterations < 1:
            raise ValueError("iterations must be positive")
        represented: list[Track] = [self.targets[self.center_index()]]
        records: list[ExpansionStep] = []
        primitive_index = 0
        # A target-specific continuation path matters when a local inverse step
        # improves chain coverage before it becomes globally nearest. Without
        # this state the harness can regenerate the same first step forever.
        frontier_source: dict[str, int] = {}
        for iteration in range(1, iterations + 1):
            before = self.coverage(represented)
            use_primitive = bool(primitives and primitive_every > 0 and iteration % primitive_every == 0)
            if use_primitive:
                source = represented[(iteration // primitive_every - 1) % len(represented)]
                kind = primitives[primitive_index % len(primitives)]
                primitive_index += 1
                anchor = (2 * iteration) % len(source.gates)
                candidate = graft_primitive(
                    source, kind, anchor=anchor, severity=primitive_severity,
                    name=f"curriculum_{iteration:02d}_{kind}",
                )
                trial = [*represented, candidate]
                after = self.coverage(trial)
                before_primitive = before.primitive
                after_primitive = after.primitive
                improved = (
                    after_primitive["label_count"] > before_primitive["label_count"]
                    or after_primitive["maximum_roll_degrees"] > before_primitive["maximum_roll_degrees"] + 1.0
                    or after_primitive["maximum_vertical_reversals"] > before_primitive["maximum_vertical_reversals"]
                )
                reasons = () if improved else ("no-primitive-coverage-gain",)
                if improved:
                    represented.append(candidate)
                else:
                    after = before
                records.append(ExpansionStep(
                    iteration=iteration, axis="primitive", source=source.name,
                    target=kind, candidate=candidate, accepted=improved,
                    reasons=reasons, proposal=None, before=before, after=after,
                ))
                continue
            profiles = [analyze_track_geometry(track) for track in represented]
            target_index = self._farthest_target(profiles)
            target = self.target_profiles[target_index]
            source_index = frontier_source.get(
                target.name, self._nearest_source(target, profiles)
            )
            source = represented[source_index]
            proposal = SplineManifoldGenerator(source).propose(
                self.atlas, target=target, standardized_step=standardized_step,
                name=f"curriculum_{iteration:02d}_toward_{target.name}",
            )
            candidate = proposal.track
            trial = [*represented, candidate]
            after = self.coverage(trial)
            global_gain = before.global_mean - after.global_mean
            chain_gain = before.transition[3]["mean"] - after.transition[3]["mean"]
            duplicate = any(
                candidate.fingerprint == existing.fingerprint
                for existing in represented
            )
            accepted = bool(
                proposal.valid and not duplicate
                and (global_gain > 1.0e-5 or chain_gain > 1.0e-5)
            )
            reasons = proposal.reasons
            if duplicate:
                reasons = reasons + ("duplicate-candidate",)
            if proposal.valid and not accepted:
                if not duplicate:
                    reasons = reasons + ("no-target-coverage-gain",)
            if accepted:
                represented.append(candidate)
                frontier_source[target.name] = len(represented) - 1
            else:
                after = before
            records.append(ExpansionStep(
                iteration=iteration, axis="geometry", source=source.name,
                target=target.name, candidate=candidate, accepted=accepted,
                reasons=reasons, proposal=proposal, before=before, after=after,
            ))
        return tuple(represented), tuple(records)
