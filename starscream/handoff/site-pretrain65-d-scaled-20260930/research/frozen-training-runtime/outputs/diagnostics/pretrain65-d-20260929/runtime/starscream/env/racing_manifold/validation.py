"""Policy-dependent validation for generated racing-course frontiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr

from .frontier import wilson_interval
from .expert_qualification import MPCCAdmissionDecision
from .route_grammar import RouteDirectedPosition


@dataclass(frozen=True, slots=True)
class PolicyCohortObservation:
    """Closed-loop outcome for one track under one frozen evaluation cohort."""

    name: str
    episodes: int
    success_rate: float
    mean_gates: float
    target_gates: int
    gate_survival: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("policy cohort must contain at least one episode")
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success rate must lie in [0, 1]")
        if self.target_gates < 1 or len(self.gate_survival) != self.target_gates:
            raise ValueError("gate survival must match target_gates")
        if any(not 0.0 <= value <= 1.0 for value in self.gate_survival):
            raise ValueError("gate survival values must lie in [0, 1]")

    @property
    def gate_fraction(self) -> float:
        return float(np.clip(self.mean_gates / self.target_gates, 0.0, 1.0))

    @property
    def completion_aware_score(self) -> float:
        return 0.75 * self.success_rate + 0.25 * self.gate_fraction


@dataclass(frozen=True, slots=True)
class RobustPolicyOutcome:
    """Conservative aggregate across disjoint reset/dynamics cohorts."""

    cohort_count: int
    episodes: int
    pooled_success_rate: float
    pooled_success_interval: tuple[float, float]
    minimum_cohort_success_rate: float
    success_rate_range: float
    minimum_gate_fraction: float
    robust_transfer_score: float
    gate_survival_floor: tuple[float, ...]
    conditional_failure_hazard: tuple[float, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "cohort_count": self.cohort_count,
            "episodes": self.episodes,
            "pooled_success_rate": self.pooled_success_rate,
            "pooled_success_interval": list(self.pooled_success_interval),
            "minimum_cohort_success_rate": self.minimum_cohort_success_rate,
            "success_rate_range": self.success_rate_range,
            "minimum_gate_fraction": self.minimum_gate_fraction,
            "robust_transfer_score": self.robust_transfer_score,
            "gate_survival_floor": list(self.gate_survival_floor),
            "conditional_failure_hazard": list(self.conditional_failure_hazard),
        }


def aggregate_policy_cohorts(
    cohorts: Sequence[PolicyCohortObservation],
) -> RobustPolicyOutcome:
    """Aggregate outcomes without allowing a favorable cohort to hide collapse."""

    if not cohorts:
        raise ValueError("at least one policy cohort is required")
    gate_counts = {row.target_gates for row in cohorts}
    if len(gate_counts) != 1:
        raise ValueError("policy cohorts must use the same target gate count")
    episodes = int(sum(row.episodes for row in cohorts))
    successes = int(sum(round(row.success_rate * row.episodes) for row in cohorts))
    pooled = successes / episodes
    rates = np.asarray([row.success_rate for row in cohorts], np.float64)
    survival = np.min(
        np.asarray([row.gate_survival for row in cohorts], np.float64), axis=0,
    )
    incoming = np.concatenate(([1.0], survival[:-1]))
    hazard = np.divide(
        np.maximum(incoming - survival, 0.0),
        np.maximum(incoming, 1.0e-12),
    )
    minimum_success = float(np.min(rates))
    minimum_gate_fraction = float(min(row.gate_fraction for row in cohorts))
    lower, upper = wilson_interval(successes, episodes)
    # Completion dominates, but partial chaining and statistical confidence
    # keep the score informative before full laps become frequent.
    robust_score = (
        0.65 * minimum_success
        + 0.20 * minimum_gate_fraction
        + 0.15 * lower
    )
    return RobustPolicyOutcome(
        cohort_count=len(cohorts),
        episodes=episodes,
        pooled_success_rate=float(pooled),
        pooled_success_interval=(lower, upper),
        minimum_cohort_success_rate=minimum_success,
        success_rate_range=float(np.max(rates) - np.min(rates)),
        minimum_gate_fraction=minimum_gate_fraction,
        robust_transfer_score=float(robust_score),
        gate_survival_floor=tuple(float(value) for value in survival),
        conditional_failure_hazard=tuple(float(value) for value in hazard),
    )


@dataclass(frozen=True, slots=True)
class CandidateTransferValidation:
    """One generated candidate under geometry, MPCC, and policy evidence."""

    name: str
    route: RouteDirectedPosition
    mpcc: MPCCAdmissionDecision
    policy: RobustPolicyOutcome | None
    source_policy_score: float | None

    @property
    def relative_policy_competence(self) -> float | None:
        if self.policy is None or self.source_policy_score is None:
            return None
        return float(
            self.policy.robust_transfer_score / max(self.source_policy_score, 1.0e-12)
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "route": self.route.to_mapping(),
            "mpcc": self.mpcc.to_mapping(),
            "policy": None if self.policy is None else self.policy.to_mapping(),
            "source_policy_score": self.source_policy_score,
            "relative_policy_competence": self.relative_policy_competence,
        }


@dataclass(frozen=True, slots=True)
class GeneratorTransferScorecard:
    """Validation metrics that cannot reward impossible or collapsed tasks."""

    candidate_count: int
    rl_admission_yield: float
    dagger_admission_yield: float
    maximum_feasible_route_progress: float
    maximum_robustly_labelable_route_progress: float
    maximum_alive_route_progress: float
    maximum_frontier_route_progress: float
    maximum_curriculum_ready_route_progress: float
    alive_candidate_count: int
    frontier_candidate_count: int
    curriculum_ready_candidate_count: int
    policy_alive_yield: float
    policy_frontier_yield: float
    curriculum_ready_yield: float
    first_collapsed_route_progress_above_alive: float | None
    policy_frontier_resolution: float | None
    route_progress_competence_spearman: float | None
    competence_monotonicity_violations: int
    minimum_source_retention: float | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "contract": "starscream-generator-transfer-scorecard-v1",
            "primary_metric": {
                "name": "maximum_alive_route_progress",
                "value": self.maximum_alive_route_progress,
                "meaning": (
                    "largest source-to-target route progress that is MPCC feasible "
                    "and remains inside the current policy competence trust region"
                ),
            },
            "curriculum_selection_metric": {
                "name": "maximum_curriculum_ready_route_progress",
                "value": self.maximum_curriculum_ready_route_progress,
                "meaning": (
                    "largest alive source-to-target step that also clears the "
                    "absolute completion and gate-survival floors for clean PPO"
                ),
            },
            "candidate_count": self.candidate_count,
            "rl_admission_yield": self.rl_admission_yield,
            "dagger_admission_yield": self.dagger_admission_yield,
            "maximum_feasible_route_progress": self.maximum_feasible_route_progress,
            "maximum_robustly_labelable_route_progress": (
                self.maximum_robustly_labelable_route_progress
            ),
            "maximum_alive_route_progress": self.maximum_alive_route_progress,
            "maximum_frontier_route_progress": self.maximum_frontier_route_progress,
            "maximum_curriculum_ready_route_progress": (
                self.maximum_curriculum_ready_route_progress
            ),
            "alive_candidate_count": self.alive_candidate_count,
            "frontier_candidate_count": self.frontier_candidate_count,
            "curriculum_ready_candidate_count": (
                self.curriculum_ready_candidate_count
            ),
            "policy_alive_yield": self.policy_alive_yield,
            "policy_frontier_yield": self.policy_frontier_yield,
            "curriculum_ready_yield": self.curriculum_ready_yield,
            "first_collapsed_route_progress_above_alive": (
                self.first_collapsed_route_progress_above_alive
            ),
            "policy_frontier_resolution": self.policy_frontier_resolution,
            "route_progress_competence_spearman": (
                self.route_progress_competence_spearman
            ),
            "competence_monotonicity_violations": (
                self.competence_monotonicity_violations
            ),
            "minimum_source_retention": self.minimum_source_retention,
        }


def score_generator_transfer(
    candidates: Sequence[CandidateTransferValidation],
    *,
    policy_relative_band: tuple[float, float] = (0.40, 0.90),
    curriculum_minimum_success_rate: float = 0.40,
    curriculum_minimum_gate_fraction: float = 0.70,
    source_retention: Sequence[float] = (),
) -> GeneratorTransferScorecard:
    """Score directional coverage after hard feasibility/competence gates.

    Speed is intentionally absent.  It is reported for admitted tracks but may
    not compensate for a failed lap, an MPCC-unsolved task, or forgetting the
    already mastered source support.
    """

    if not candidates:
        raise ValueError("generator transfer scoring needs candidates")
    low, high = (float(item) for item in policy_relative_band)
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("policy relative band must lie in [0, 1]")
    if not 0.0 <= curriculum_minimum_success_rate <= 1.0:
        raise ValueError("curriculum success floor must lie in [0, 1]")
    if not 0.0 <= curriculum_minimum_gate_fraction <= 1.0:
        raise ValueError("curriculum gate floor must lie in [0, 1]")
    feasible = [row for row in candidates if row.mpcc.rl_eligible]
    labelable = [row for row in candidates if row.mpcc.dagger_eligible]
    measured = [
        row for row in feasible if row.relative_policy_competence is not None
    ]
    alive = [
        row for row in measured
        if float(row.relative_policy_competence) >= low
    ]
    frontier = [
        row for row in measured
        if low <= float(row.relative_policy_competence) <= high
    ]
    curriculum_ready = [
        row for row in alive
        if row.policy is not None
        and row.policy.minimum_cohort_success_rate
        >= curriculum_minimum_success_rate
        and row.policy.minimum_gate_fraction
        >= curriculum_minimum_gate_fraction
    ]
    positive = lambda rows: max(
        (max(float(row.route.progress), 0.0) for row in rows), default=0.0,
    )
    retention = tuple(float(item) for item in source_retention)
    if any(not np.isfinite(item) or not 0.0 <= item <= 1.0 for item in retention):
        raise ValueError("source retention values must lie in [0, 1]")
    maximum_alive = positive(alive)
    collapsed_above = sorted(
        max(float(row.route.progress), 0.0)
        for row in measured
        if float(row.relative_policy_competence) < low
        and float(row.route.progress) > maximum_alive
    )
    collapse_edge = collapsed_above[0] if collapsed_above else None
    ordered = sorted(measured, key=lambda row: float(row.route.progress))
    substantial_inversions = sum(
        1
        for left_index, left in enumerate(ordered)
        for right in ordered[left_index + 1:]
        if float(right.relative_policy_competence)
        > float(left.relative_policy_competence) + 0.10
    )
    if len(ordered) >= 3:
        rank = spearmanr(
            [float(row.route.progress) for row in ordered],
            [float(row.relative_policy_competence) for row in ordered],
        ).statistic
        route_competence_rank = (
            None if not np.isfinite(rank) else float(rank)
        )
    else:
        route_competence_rank = None
    return GeneratorTransferScorecard(
        candidate_count=len(candidates),
        rl_admission_yield=len(feasible) / len(candidates),
        dagger_admission_yield=len(labelable) / len(candidates),
        maximum_feasible_route_progress=positive(feasible),
        maximum_robustly_labelable_route_progress=positive(labelable),
        maximum_alive_route_progress=maximum_alive,
        maximum_frontier_route_progress=positive(frontier),
        maximum_curriculum_ready_route_progress=positive(curriculum_ready),
        alive_candidate_count=len(alive),
        frontier_candidate_count=len(frontier),
        curriculum_ready_candidate_count=len(curriculum_ready),
        policy_alive_yield=(len(alive) / len(measured) if measured else 0.0),
        policy_frontier_yield=(
            len(frontier) / len(measured) if measured else 0.0
        ),
        curriculum_ready_yield=(
            len(curriculum_ready) / len(measured) if measured else 0.0
        ),
        first_collapsed_route_progress_above_alive=collapse_edge,
        policy_frontier_resolution=(
            None if collapse_edge is None else collapse_edge - maximum_alive
        ),
        route_progress_competence_spearman=route_competence_rank,
        competence_monotonicity_violations=substantial_inversions,
        minimum_source_retention=(min(retention) if retention else None),
    )
