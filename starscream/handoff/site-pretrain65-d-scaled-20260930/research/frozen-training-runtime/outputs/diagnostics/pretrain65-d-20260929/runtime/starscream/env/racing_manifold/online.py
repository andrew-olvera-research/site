"""Stratified course generation and policy-gated online curricula.

Racing-course space is not one Euclidean vector.  Gate positions, frames, and
apertures are continuous inside a fixed route topology, while gate count,
rendered/route-only checkpoints, traversal direction, and maneuver primitives
form discrete strata.  This module keeps those coordinates explicit and gives
an online RL driver a small, testable state machine for expanding competence
without changing the task pool inside an on-policy update.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence

import numpy as np

from ..tracks import Track
from .generator import RouteExtensionProposal, RouteSplineManifoldGenerator
from .morph import PhaseAlignedTrackMorpher
from .primitives import PrimitiveKind, analyze_primitives, graft_primitive
from .route_grammar import RouteDirectedPosition, RouteGrammarProfile, route_grammar_profile


CurriculumAction = Literal["expand", "hold", "rehearse", "retreat", "complete"]


@dataclass(frozen=True, slots=True)
class ManifoldStratum:
    """Discrete and primitive coordinates that route embeddings can obscure."""

    route_checkpoint_count: int
    physical_gate_count: int
    virtual_checkpoint_count: int
    opposite_entry_count: int
    vertical_reversal_count: int
    high_roll_gate_fraction: float
    maximum_roll_degrees: float
    primitive_labels: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "route_checkpoint_count": self.route_checkpoint_count,
            "physical_gate_count": self.physical_gate_count,
            "virtual_checkpoint_count": self.virtual_checkpoint_count,
            "opposite_entry_count": self.opposite_entry_count,
            "vertical_reversal_count": self.vertical_reversal_count,
            "high_roll_gate_fraction": self.high_roll_gate_fraction,
            "maximum_roll_degrees": self.maximum_roll_degrees,
            "primitive_labels": list(self.primitive_labels),
        }


def manifold_stratum(track: Track) -> ManifoldStratum:
    primitive = analyze_primitives(track)
    physical = sum(bool(gate.render and gate.kind == "gate") for gate in track.gates)
    return ManifoldStratum(
        route_checkpoint_count=len(track.gates),
        physical_gate_count=physical,
        virtual_checkpoint_count=len(track.gates) - physical,
        opposite_entry_count=sum(gate.enter_from_opposite_side for gate in track.gates),
        vertical_reversal_count=primitive.vertical_reversal_count,
        high_roll_gate_fraction=primitive.high_roll_gate_fraction,
        maximum_roll_degrees=primitive.maximum_roll_degrees,
        primitive_labels=primitive.labels,
    )


def _embedding_parts(profile: RouteGrammarProfile) -> tuple[np.ndarray, np.ndarray]:
    """Recover global and ordered blocks from ``route_grammar_profile``."""

    width = int(profile.records.shape[1])
    global_width = 5 * width
    if len(profile.embedding) <= global_width:
        raise ValueError("route grammar embedding has no ordered phase block")
    ordered_width = len(profile.embedding) - global_width
    if ordered_width % width:
        raise ValueError("route grammar embedding has an invalid phase width")
    return (
        np.asarray(profile.embedding[:global_width], np.float64),
        np.asarray(profile.embedding[global_width:], np.float64).reshape(-1, width),
    )


def _embedding_with_gate_shift(
    profile: RouteGrammarProfile, gate_shift: int,
) -> np.ndarray:
    """Rebuild the ordered embedding after an exact cyclic gate renumbering."""

    global_embedding, existing_ordered = _embedding_parts(profile)
    width = int(profile.records.shape[1])
    phase_samples = int(existing_ordered.shape[0])
    # ``records`` carries one factor 1/sqrt(width); undo it before applying
    # the same periodic interpolation and embedding normalization used by
    # ``route_grammar_profile``.
    values = np.roll(
        np.asarray(profile.records, np.float64) * np.sqrt(width),
        int(gate_shift), axis=0,
    )
    count = len(values)
    phase = np.arange(count, dtype=np.float64) / count
    closed_phase = np.concatenate([phase, [1.0]])
    closed_values = np.concatenate([values, values[:1]], axis=0)
    query = np.arange(phase_samples, dtype=np.float64) / phase_samples
    ordered = np.column_stack([
        np.interp(query, closed_phase, closed_values[:, index])
        for index in range(width)
    ])
    ordered_embedding = (
        ordered.reshape(-1) / np.sqrt(ordered.size) / np.sqrt(2.0)
    )
    return np.concatenate([global_embedding, ordered_embedding])


def _phase_align_embedding(
    reference: RouteGrammarProfile,
    candidate: RouteGrammarProfile,
) -> tuple[np.ndarray, int]:
    """Cyclically align a loop route without changing its physical task.

    Gate zero is an indexing convention for loop tracks.  The original route
    ray incorrectly treated a cyclic renumbering as a large geometry change,
    which becomes especially damaging when crossing a gate-count stratum.
    """

    _, reference_ordered = _embedding_parts(reference)
    _, candidate_ordered = _embedding_parts(candidate)
    if reference_ordered.shape != candidate_ordered.shape:
        raise ValueError("route profiles use incompatible phase contracts")
    best: tuple[float, int, np.ndarray] | None = None
    reference_embedding = np.asarray(reference.embedding, np.float64)
    for shift in range(candidate.gate_count):
        embedding = _embedding_with_gate_shift(candidate, shift)
        distance = float(np.linalg.norm(embedding - reference_embedding))
        row = (distance, shift, embedding)
        if best is None or row[0] < best[0]:
            best = row
    assert best is not None
    return best[2], int(best[1])


@dataclass(frozen=True, slots=True)
class StratifiedPosition:
    """Continuous route ray plus explicit non-Euclidean course coordinates."""

    route: RouteDirectedPosition
    phase_shift: int
    source_stratum: ManifoldStratum
    target_stratum: ManifoldStratum
    candidate_stratum: ManifoldStratum
    axis_progress: dict[str, float | None]
    axis_target_residual: dict[str, float]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "route": self.route.to_mapping(),
            "phase_shift": self.phase_shift,
            "source_stratum": self.source_stratum.to_mapping(),
            "target_stratum": self.target_stratum.to_mapping(),
            "candidate_stratum": self.candidate_stratum.to_mapping(),
            "axis_progress": dict(self.axis_progress),
            "axis_target_residual": dict(self.axis_target_residual),
        }


def _axis_progress(source: float, target: float, candidate: float) -> float | None:
    delta = float(target - source)
    if abs(delta) <= 1.0e-12:
        return None
    return float((candidate - source) / delta)


def stratified_position(
    source_track: Track,
    target_track: Track,
    candidate_track: Track,
    *,
    future_gates: int = 6,
    phase_samples: int = 24,
) -> StratifiedPosition:
    """Place a candidate on a cyclic-invariant, mixed course coordinate chart."""

    source = route_grammar_profile(
        source_track, future_gates=future_gates, phase_samples=phase_samples,
    )
    raw_target = route_grammar_profile(
        target_track, future_gates=future_gates, phase_samples=phase_samples,
    )
    raw_candidate = route_grammar_profile(
        candidate_track, future_gates=future_gates, phase_samples=phase_samples,
    )
    target_embedding, _ = _phase_align_embedding(source, raw_target)
    source_embedding = np.asarray(source.embedding, np.float64)
    direction = target_embedding - source_embedding
    squared_distance = float(direction @ direction)
    if squared_distance <= 1.0e-16:
        raise ValueError("source and target route contracts are indistinguishable")

    # A candidate can be cyclically numbered in several equivalent ways. Pick
    # the numbering closest to the frozen source-target ray, not merely source.
    best: tuple[float, float, int, np.ndarray, float] | None = None
    for shift in range(raw_candidate.gate_count):
        embedding = _embedding_with_gate_shift(raw_candidate, shift)
        offset = embedding - source_embedding
        progress = float(offset @ direction / squared_distance)
        residual = offset - progress * direction
        target_distance = float(np.linalg.norm(embedding - target_embedding))
        row = (
            float(np.linalg.norm(residual)), target_distance, shift, embedding,
            progress,
        )
        if best is None or row[:2] < best[:2]:
            best = row
    assert best is not None
    off_axis, target_distance, phase_shift, candidate_embedding, progress = best
    baseline = float(np.sqrt(squared_distance))
    offset = candidate_embedding - source_embedding
    route = RouteDirectedPosition(
        progress=progress,
        off_axis_ratio=float(off_axis / baseline),
        source_distance=float(np.linalg.norm(offset)),
        target_distance=target_distance,
        target_gain_fraction=float(1.0 - target_distance / baseline),
    )
    source_stratum = manifold_stratum(source_track)
    target_stratum = manifold_stratum(target_track)
    candidate_stratum = manifold_stratum(candidate_track)
    axes = {
        "route_checkpoint_count": (
            source_stratum.route_checkpoint_count,
            target_stratum.route_checkpoint_count,
            candidate_stratum.route_checkpoint_count,
        ),
        "physical_gate_count": (
            source_stratum.physical_gate_count,
            target_stratum.physical_gate_count,
            candidate_stratum.physical_gate_count,
        ),
        "opposite_entry_count": (
            source_stratum.opposite_entry_count,
            target_stratum.opposite_entry_count,
            candidate_stratum.opposite_entry_count,
        ),
        "vertical_reversal_count": (
            source_stratum.vertical_reversal_count,
            target_stratum.vertical_reversal_count,
            candidate_stratum.vertical_reversal_count,
        ),
        "high_roll_gate_fraction": (
            source_stratum.high_roll_gate_fraction,
            target_stratum.high_roll_gate_fraction,
            candidate_stratum.high_roll_gate_fraction,
        ),
        "maximum_roll_degrees": (
            source_stratum.maximum_roll_degrees,
            target_stratum.maximum_roll_degrees,
            candidate_stratum.maximum_roll_degrees,
        ),
    }
    return StratifiedPosition(
        route=route,
        phase_shift=phase_shift,
        source_stratum=source_stratum,
        target_stratum=target_stratum,
        candidate_stratum=candidate_stratum,
        axis_progress={
            name: _axis_progress(*values) for name, values in axes.items()
        },
        axis_target_residual={
            name: abs(float(values[2] - values[1])) for name, values in axes.items()
        },
    )


@dataclass(frozen=True, slots=True)
class StratifiedCourseProposal:
    track: Track
    operation: str
    position: StratifiedPosition
    local_route_proposal: RouteExtensionProposal | None
    valid: bool
    reasons: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "track": self.track.name,
            "fingerprint": self.track.fingerprint,
            "operation": self.operation,
            "position": self.position.to_mapping(),
            "local_route_proposal": (
                None if self.local_route_proposal is None
                else self.local_route_proposal.to_mapping()
            ),
            "valid": self.valid,
            "reasons": list(self.reasons),
        }


class StratifiedRouteGenerator:
    """Propose local moves and explicit course-stratum boundary crossings."""

    def __init__(
        self,
        source: Track,
        target: Track,
        *,
        future_gates: int = 6,
        phase_samples: int = 24,
    ) -> None:
        self.source = source
        self.target = target
        self.future_gates = int(future_gates)
        self.phase_samples = int(phase_samples)
        self.topology_bridge = PhaseAlignedTrackMorpher(source, target)

    def _position(self, candidate: Track) -> StratifiedPosition:
        return stratified_position(
            self.source, self.target, candidate,
            future_gates=self.future_gates, phase_samples=self.phase_samples,
        )

    def continuous(
        self,
        base: Track,
        *,
        target_fraction_step: float,
        name: str,
        maximum_gate_displacement_m: float = 1.5,
    ) -> StratifiedCourseProposal:
        local = RouteSplineManifoldGenerator(
            base, self.target,
            future_gates=self.future_gates,
            phase_samples=self.phase_samples,
            maximum_gate_displacement_m=maximum_gate_displacement_m,
        ).propose(target_fraction_step=target_fraction_step, name=name)
        position = self._position(local.track)
        reasons = list(local.reasons)
        if position.route.target_gain_fraction <= 0.0:
            reasons.append("no-global-route-target-gain")
        valid = bool(local.valid and not reasons)
        return StratifiedCourseProposal(
            track=local.track,
            operation="continuous_route_inverse",
            position=position,
            local_route_proposal=local,
            valid=valid,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def topology_proxy(
        self,
        *,
        alpha: float,
        name: str,
    ) -> StratifiedCourseProposal:
        track = self.topology_bridge.materialize(
            alpha, name=name, demote_inserted_checkpoints=True,
        )
        position = self._position(track)
        points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
        pairwise = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        pairwise += np.eye(len(points)) * 1.0e6
        reasons: list[str] = []
        if float(pairwise.min()) < 0.75:
            reasons.append("checkpoint-collision")
        if sum(gate.render for gate in track.gates) != len(self.target.gates):
            reasons.append("physical-gate-count-not-target-aligned")
        return StratifiedCourseProposal(
            track=track,
            operation="demote_checkpoints_and_phase_morph",
            position=position,
            local_route_proposal=None,
            valid=not reasons,
            reasons=tuple(reasons),
        )

    def topology_proxy_step(
        self,
        *,
        current_alpha: float,
        maximum_route_distance: float,
        name: str,
        iterations: int = 32,
    ) -> StratifiedCourseProposal:
        """Take a bounded local step along the topology-normalized bridge.

        Morph ``alpha`` is not a distance coordinate: on Swift-to-CDRA, an
        equal alpha increment can be more than an order of magnitude harder
        late in the bridge than early in it. The online scheduler therefore
        asks for a maximum route-contract distance and this method inverts the
        alpha-to-distance map by bisection. This is only a proposal; MPCC and
        policy evidence still gate publication.
        """

        current_alpha = float(current_alpha)
        maximum_route_distance = float(maximum_route_distance)
        if not 0.0 <= current_alpha < 1.0:
            raise ValueError("current_alpha must lie in [0, 1)")
        if maximum_route_distance <= 0.0:
            raise ValueError("maximum_route_distance must be positive")
        if iterations < 1:
            raise ValueError("iterations must be positive")

        current = self.topology_bridge.materialize(
            current_alpha,
            name=f"{name}_current",
            demote_inserted_checkpoints=True,
        )

        def distance(alpha: float) -> float:
            candidate = self.topology_bridge.materialize(
                alpha,
                name=f"{name}_probe",
                demote_inserted_checkpoints=True,
            )
            return float(
                stratified_position(
                    current,
                    self.target,
                    candidate,
                    future_gates=self.future_gates,
                    phase_samples=self.phase_samples,
                ).route.source_distance
            )

        if distance(1.0) <= maximum_route_distance:
            selected_alpha = 1.0
        else:
            lower = current_alpha
            upper = 1.0
            for _ in range(iterations):
                midpoint = 0.5 * (lower + upper)
                if distance(midpoint) <= maximum_route_distance:
                    lower = midpoint
                else:
                    upper = midpoint
            selected_alpha = lower
        if selected_alpha <= current_alpha + 1.0e-8:
            raise ValueError("route-distance inversion could not make progress")
        proposal = self.topology_proxy(alpha=selected_alpha, name=name)
        metadata = dict(proposal.track.metadata or {})
        transition = dict(metadata.get("stratified_transition", {}))
        transition.update({
            "operation": "bounded_topology_proxy_step",
            "current_alpha": current_alpha,
            "selected_alpha": selected_alpha,
            "maximum_route_distance": maximum_route_distance,
        })
        metadata["stratified_transition"] = transition
        track = replace(proposal.track, metadata=metadata)
        return replace(
            proposal,
            track=track,
            operation="bounded_topology_proxy_step",
            position=self._position(track),
        )

    def target_topology(self, *, name: str) -> StratifiedCourseProposal:
        """Cross the final stratum by removing target-aligned route checkpoints.

        The exact target is returned rather than a cyclically renumbered copy.
        Global position and yaw are irrelevant to body/gate-relative policy
        inputs, while preserving the released geometry avoids benchmark drift.
        """

        metadata = dict(self.target.metadata or {})
        metadata["stratified_transition"] = {
            "operation": "remove_virtual_route_checkpoints",
            "source": self.source.name,
            "target": self.target.name,
        }
        track = replace(self.target, name=name, metadata=metadata)
        return StratifiedCourseProposal(
            track=track,
            operation="remove_virtual_route_checkpoints",
            position=self._position(track),
            local_route_proposal=None,
            valid=True,
            reasons=(),
        )

    def remove_virtual_checkpoint(
        self,
        proxy: Track,
        *,
        checkpoint_name: str,
        name: str,
    ) -> StratifiedCourseProposal:
        """Remove exactly one explicit route-only checkpoint.

        Gate-count transitions are deliberately one-at-a-time so policy and
        MPCC evidence can identify the particular discontinuity that fails.
        """

        matches = [
            index for index, gate in enumerate(proxy.gates)
            if gate.name == checkpoint_name and not gate.render
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one virtual checkpoint named {checkpoint_name!r}"
            )
        index = matches[0]
        gates = proxy.gates[:index] + proxy.gates[index + 1:]
        metadata = dict(proxy.metadata or {})
        history = list(metadata.get("topology_operations", ()))
        history.append({
            "operation": "remove_virtual_route_checkpoint",
            "checkpoint_name": checkpoint_name,
            "checkpoint_index": index,
        })
        metadata["topology_operations"] = history
        track = replace(proxy, name=name, gates=gates, metadata=metadata)
        return StratifiedCourseProposal(
            track=track,
            operation="remove_virtual_route_checkpoint",
            position=self._position(track),
            local_route_proposal=None,
            valid=True,
            reasons=(),
        )

    def primitive(
        self,
        base: Track,
        kind: PrimitiveKind,
        *,
        anchor: int,
        severity: float,
        name: str,
    ) -> StratifiedCourseProposal:
        track = graft_primitive(
            base, kind, anchor=anchor, severity=severity, name=name,
        )
        position = self._position(track)
        return StratifiedCourseProposal(
            track=track,
            operation=f"primitive:{kind}",
            position=position,
            local_route_proposal=None,
            valid=True,
            reasons=(),
        )

    def directionality(
        self,
        base: Track,
        *,
        gate_index: int,
        enter_from_opposite_side: bool,
        name: str,
    ) -> StratifiedCourseProposal:
        """Propose one explicit gate-entry direction change.

        Direction is a discrete route-grammar coordinate, not a gate-frame
        perturbation. Keeping it as a named operator prevents a continuous
        optimizer from hiding a traversal reversal inside quaternion distance.
        """

        gate_index = int(gate_index) % len(base.gates)
        gates = list(base.gates)
        changed = (
            gates[gate_index].enter_from_opposite_side
            != bool(enter_from_opposite_side)
        )
        gates[gate_index] = replace(
            gates[gate_index],
            enter_from_opposite_side=bool(enter_from_opposite_side),
        )
        metadata = dict(base.metadata or {})
        metadata["directionality_operation"] = {
            "gate_index": gate_index,
            "enter_from_opposite_side": bool(enter_from_opposite_side),
        }
        track = replace(base, name=name, gates=tuple(gates), metadata=metadata)
        reasons = () if changed else ("directionality-unchanged",)
        return StratifiedCourseProposal(
            track=track,
            operation="directionality:opposite_entry",
            position=self._position(track),
            local_route_proposal=None,
            valid=changed,
            reasons=reasons,
        )


@dataclass(frozen=True, slots=True)
class OnlinePolicyProbe:
    episodes: int
    success_rate: float
    mean_gate_fraction: float
    robust_score: float

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("policy probe requires at least one episode")
        for value in (self.success_rate, self.mean_gate_fraction, self.robust_score):
            if not 0.0 <= value <= 1.0:
                raise ValueError("policy probe metrics must lie in [0, 1]")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "mean_gate_fraction": self.mean_gate_fraction,
            "robust_score": self.robust_score,
        }


@dataclass(frozen=True, slots=True)
class OnlineCandidateEvidence:
    proposal: StratifiedCourseProposal
    mpcc_eligible: bool
    policy: OnlinePolicyProbe


@dataclass(frozen=True, slots=True)
class OnlineCurriculumConfig:
    frontier_success_floor: float = 0.35
    frontier_success_ceiling: float = 0.70
    mastery_success: float = 0.75
    minimum_gate_fraction: float = 0.65
    source_retention_floor: float = 0.80
    anchor_weight: float = 0.30
    frontier_weight: float = 0.50
    interior_weight: float = 0.20

    def __post_init__(self) -> None:
        values = (
            self.frontier_success_floor, self.frontier_success_ceiling,
            self.mastery_success, self.minimum_gate_fraction,
            self.source_retention_floor, self.anchor_weight,
            self.frontier_weight, self.interior_weight,
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("curriculum thresholds and weights must lie in [0, 1]")
        if not self.frontier_success_floor <= self.frontier_success_ceiling:
            raise ValueError("frontier success band is inverted")
        if self.mastery_success < self.frontier_success_ceiling:
            raise ValueError("mastery must not be below the frontier ceiling")
        if not np.isclose(
            self.anchor_weight + self.frontier_weight + self.interior_weight, 1.0,
        ):
            raise ValueError("curriculum sample weights must sum to one")


@dataclass(frozen=True, slots=True)
class OnlineCurriculumDecision:
    action: CurriculumAction
    selected_name: str | None
    reason: str
    sample_weights: dict[str, float]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "selected_name": self.selected_name,
            "reason": self.reason,
            "sample_weights": dict(self.sample_weights),
        }


class OnlineManifoldCurriculum:
    """Choose an immutable next pool from MPCC and policy evidence."""

    def __init__(self, config: OnlineCurriculumConfig | None = None) -> None:
        self.config = config or OnlineCurriculumConfig()

    def _weights(self, source_retention: float, *, rehearse: bool) -> dict[str, float]:
        cfg = self.config
        if rehearse:
            anchor = min(0.80, cfg.anchor_weight + 0.35)
            frontier = max(0.15, cfg.frontier_weight - 0.25)
            interior = 1.0 - anchor - frontier
        else:
            pressure = max(cfg.source_retention_floor + 0.08 - source_retention, 0.0)
            anchor = min(0.55, cfg.anchor_weight + pressure)
            frontier = cfg.frontier_weight
            interior = 1.0 - anchor - frontier
        return {
            "anchor_rehearsal": float(anchor),
            "interior_replay": float(max(interior, 0.0)),
            "frontier": float(frontier),
        }

    def decide(
        self,
        candidates: Sequence[OnlineCandidateEvidence],
        *,
        source_retention: float,
        current_frontier_success: float,
        target_reached: bool = False,
    ) -> OnlineCurriculumDecision:
        cfg = self.config
        if target_reached:
            return OnlineCurriculumDecision(
                "complete", None, "target stratum mastered",
                self._weights(source_retention, rehearse=False),
            )
        if source_retention < cfg.source_retention_floor:
            return OnlineCurriculumDecision(
                "rehearse", None, "protected source retention below floor",
                self._weights(source_retention, rehearse=True),
            )
        # Do not jump over a frontier the policy has already lost merely
        # because a farther candidate happened to pass a noisy probe.  The old
        # ordering allowed exactly that: candidate admission ran before this
        # guard, so the curriculum could expand while its active edge was below
        # the learning floor.
        if current_frontier_success < cfg.frontier_success_floor:
            return OnlineCurriculumDecision(
                "retreat", None, "current frontier fell below the learning floor",
                self._weights(source_retention, rehearse=True),
            )
        eligible = [
            row for row in candidates
            if row.proposal.valid and row.mpcc_eligible
            and row.policy.mean_gate_fraction >= cfg.minimum_gate_fraction
            and row.policy.success_rate >= cfg.frontier_success_floor
        ]
        if eligible:
            # Use the farthest still-live candidate, while preferring one in
            # the learning-progress band over an already mastered easy task.
            frontier = [
                row for row in eligible
                if row.policy.success_rate <= cfg.frontier_success_ceiling
            ] or eligible
            selected = max(
                frontier, key=lambda row: row.proposal.position.route.progress,
            )
            return OnlineCurriculumDecision(
                "expand", selected.proposal.track.name,
                "farthest MPCC-feasible candidate remains inside policy trust region",
                self._weights(source_retention, rehearse=False),
            )
        if current_frontier_success >= cfg.mastery_success:
            return OnlineCurriculumDecision(
                "hold", None, "no proposed neighbor is inside the competence band",
                self._weights(source_retention, rehearse=False),
            )
        return OnlineCurriculumDecision(
            "hold", None, "continue optimizing the current frontier",
            self._weights(source_retention, rehearse=False),
        )
