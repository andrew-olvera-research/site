"""Evidence-gated admission for generated racing tasks.

Static geometry is a proposal mechanism, not a training-data qualification.
Candidates enter a pretraining corpus only after they cover a missing
route-conditioned behavior and remain expert-solvable. Candidates enter an RL
curriculum only after the current policy also places them on a live competence
frontier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BehavioralAdmissionConfig:
    minimum_geometry_target_gain_fraction: float = 0.02
    minimum_behavior_target_gain_fraction: float = 0.02
    minimum_route_progress: float = 0.05
    maximum_behavior_off_axis_ratio: float = 1.00
    # Fallback-heavy MPCC rollouts are not clean expert evidence. Historical
    # v6 candidates qualified below 1%; a 10% ceiling admitted trajectories
    # whose labels changed discontinuously around solver failures.
    maximum_solver_failure_fraction: float = 0.02
    minimum_source_policy_success: float = 0.60
    policy_relative_success_band: tuple[float, float] = (0.40, 0.90)

    def __post_init__(self) -> None:
        low, high = self.policy_relative_success_band
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("policy relative success band must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class CandidateAdmissionEvidence:
    name: str
    topology_compatible: bool
    geometry_target_gain_fraction: float
    behavior_target_gain_fraction: float
    route_progress: float
    behavior_off_axis_ratio: float
    expert_success: bool
    solver_failure_fraction: float
    source_policy_success: float | None = None
    candidate_policy_success: float | None = None
    policy_episodes: int = 0
    absolute_learning_progress: float | None = None


@dataclass(frozen=True, slots=True)
class CandidateAdmissionDecision:
    name: str
    pretraining_eligible: bool
    curriculum_eligible: bool
    score: float
    reasons: tuple[str, ...]
    policy_target_band: tuple[float, float] | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pretraining_eligible": self.pretraining_eligible,
            "curriculum_eligible": self.curriculum_eligible,
            "score": self.score,
            "reasons": list(self.reasons),
            "policy_target_band": (
                None if self.policy_target_band is None
                else list(self.policy_target_band)
            ),
        }


def evaluate_candidate_admission(
    evidence: CandidateAdmissionEvidence,
    config: BehavioralAdmissionConfig | None = None,
) -> CandidateAdmissionDecision:
    """Apply the geometry -> behavior -> expert -> policy admission ladder."""

    cfg = config or BehavioralAdmissionConfig()
    reasons: list[str] = []
    if not evidence.topology_compatible:
        reasons.append("topology-not-normalized")
    if evidence.geometry_target_gain_fraction < cfg.minimum_geometry_target_gain_fraction:
        reasons.append("insufficient-geometry-target-gain")
    if evidence.behavior_target_gain_fraction < cfg.minimum_behavior_target_gain_fraction:
        reasons.append("insufficient-behavior-target-gain")
    if evidence.route_progress < cfg.minimum_route_progress:
        reasons.append("insufficient-route-support-progress")
    if evidence.behavior_off_axis_ratio > cfg.maximum_behavior_off_axis_ratio:
        reasons.append("behavior-shift-is-off-axis")
    if not evidence.expert_success:
        reasons.append("mpcc-did-not-complete")
    if evidence.solver_failure_fraction > cfg.maximum_solver_failure_fraction:
        reasons.append("mpcc-solver-failure-rate")
    pretraining_reasons = tuple(reasons)
    pretraining_eligible = not pretraining_reasons

    policy_band: tuple[float, float] | None = None
    if evidence.source_policy_success is None or evidence.candidate_policy_success is None:
        reasons.append("policy-frontier-not-measured")
    else:
        if evidence.source_policy_success < cfg.minimum_source_policy_success:
            reasons.append("source-policy-not-mastered")
        low, high = cfg.policy_relative_success_band
        policy_band = (
            low * evidence.source_policy_success,
            high * evidence.source_policy_success,
        )
        if not policy_band[0] <= evidence.candidate_policy_success <= policy_band[1]:
            reasons.append("candidate-outside-live-policy-frontier")
        if evidence.policy_episodes < 24:
            reasons.append("policy-frontier-under-sampled")

    # Score is only a ranking hint after hard admission. Learning progress is
    # deliberately absolute: both newly learned and newly forgotten tasks are
    # informative for replay, while the hard competence band prevents collapse.
    learning_progress = abs(float(evidence.absolute_learning_progress or 0.0))
    score = (
        0.30 * evidence.geometry_target_gain_fraction
        + 0.40 * evidence.behavior_target_gain_fraction
        + 0.30 * evidence.route_progress
        + 0.20 * learning_progress
        - 0.25 * max(evidence.behavior_off_axis_ratio - 0.50, 0.0)
        - 0.50 * evidence.solver_failure_fraction
    )
    return CandidateAdmissionDecision(
        name=evidence.name,
        pretraining_eligible=pretraining_eligible,
        curriculum_eligible=not reasons,
        score=float(score),
        reasons=tuple(reasons),
        policy_target_band=policy_band,
    )
