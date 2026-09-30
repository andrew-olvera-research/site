"""Hard MPCC admission contracts for generated racing tasks.

Geometry and route embeddings are useful proposal coordinates, but neither
proves that a candidate is a valid control problem.  This module interprets
the multi-start audits produced by ``audit_dagger_teacher_labels.py`` and
keeps three claims separate:

``physically_feasible``
    MPCC completes the nominal course without a collision.  This is the
    minimum contract for exposing a generated task to policy-gradient RL.

``teacher_labelable``
    Every physical gate phase was reached through an expert prefix and the
    nominal controller completed reliably.  This is required before the task
    may contribute DAgger labels.

``robustly_labelable``
    The same route remains sufficiently reliable under the configured plant
    randomization and DART perturbations.  This is the default pretraining
    admission contract.

Cold gate-normal resets are deliberately not part of this contract: they may
construct states that are impossible to reach on the course.  The canonical
audit uses dynamically realized expert prefixes instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class MPCCAdmissionThresholds:
    """Thresholds calibrated against the exact Swift 16.5 m/s teacher."""

    minimum_nominal_success_rate: float = 0.95
    minimum_nominal_start_success_rate: float = 0.95
    minimum_stress_success_rate: float = 0.70
    maximum_nominal_solver_failure_fraction: float = 0.025
    maximum_stress_solver_failure_fraction: float = 0.15
    minimum_nominal_repeats_per_start: int = 1
    minimum_stress_repeats_per_start: int = 2

    def __post_init__(self) -> None:
        rates = (
            self.minimum_nominal_success_rate,
            self.minimum_nominal_start_success_rate,
            self.minimum_stress_success_rate,
            self.maximum_nominal_solver_failure_fraction,
            self.maximum_stress_solver_failure_fraction,
        )
        if any(not np.isfinite(item) or not 0.0 <= item <= 1.0 for item in rates):
            raise ValueError("MPCC admission rates must lie in [0, 1]")
        if min(
            self.minimum_nominal_repeats_per_start,
            self.minimum_stress_repeats_per_start,
        ) < 1:
            raise ValueError("MPCC admission needs at least one repeat per start")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "minimum_nominal_success_rate": self.minimum_nominal_success_rate,
            "minimum_nominal_start_success_rate": (
                self.minimum_nominal_start_success_rate
            ),
            "minimum_stress_success_rate": self.minimum_stress_success_rate,
            "maximum_nominal_solver_failure_fraction": (
                self.maximum_nominal_solver_failure_fraction
            ),
            "maximum_stress_solver_failure_fraction": (
                self.maximum_stress_solver_failure_fraction
            ),
            "minimum_nominal_repeats_per_start": (
                self.minimum_nominal_repeats_per_start
            ),
            "minimum_stress_repeats_per_start": (
                self.minimum_stress_repeats_per_start
            ),
        }


@dataclass(frozen=True, slots=True)
class MPCCRegimeEvidence:
    name: str
    episodes: int
    success_rate: float
    crash_rate: float
    solver_failure_fraction: float
    observed_start_indices: tuple[int, ...]
    episodes_by_start: tuple[tuple[int, int], ...]
    success_rate_by_start: tuple[tuple[int, float], ...]
    successful_canonical_laps: int

    @classmethod
    def from_track_audit(
        cls, name: str, audit: Mapping[str, Any],
    ) -> "MPCCRegimeEvidence":
        episodes = tuple(dict(item) for item in audit.get("episodes", ()))
        summary = dict(audit.get("summary") or {})
        by_start = dict(summary.get("by_start_gate") or {})
        observed = tuple(sorted(int(index) for index in by_start))
        episodes_by_start = tuple(
            (index, int(dict(by_start[str(index)]).get("episodes", 0)))
            for index in observed
        )
        success_by_start = tuple(
            (index, float(dict(by_start[str(index)]).get("success_rate", 0.0)))
            for index in observed
        )
        canonical_success = sum(
            int(bool(item.get("success")))
            for item in episodes if int(item.get("start_gate_index", 0)) == 0
        )
        return cls(
            name=str(name),
            episodes=len(episodes),
            success_rate=float(summary.get("success_rate", 0.0)),
            crash_rate=float(summary.get("crash_rate", 1.0)),
            solver_failure_fraction=float(
                summary.get("solver_failure_fraction", 1.0)
            ),
            observed_start_indices=observed,
            episodes_by_start=episodes_by_start,
            success_rate_by_start=success_by_start,
            successful_canonical_laps=canonical_success,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "crash_rate": self.crash_rate,
            "solver_failure_fraction": self.solver_failure_fraction,
            "observed_start_indices": list(self.observed_start_indices),
            "episodes_by_start": {
                str(index): count for index, count in self.episodes_by_start
            },
            "success_rate_by_start": {
                str(index): rate for index, rate in self.success_rate_by_start
            },
            "successful_canonical_laps": self.successful_canonical_laps,
        }


@dataclass(frozen=True, slots=True)
class MPCCAdmissionDecision:
    physically_feasible: bool
    teacher_labelable: bool
    robustly_labelable: bool
    expected_start_indices: tuple[int, ...]
    nominal: MPCCRegimeEvidence | None
    stress: MPCCRegimeEvidence | None
    reasons: tuple[str, ...]

    @property
    def rl_eligible(self) -> bool:
        return self.physically_feasible

    @property
    def dagger_eligible(self) -> bool:
        return self.robustly_labelable

    def to_mapping(self) -> dict[str, Any]:
        return {
            "contract": "starscream-mpcc-admission-v2",
            "physically_feasible": self.physically_feasible,
            "teacher_labelable": self.teacher_labelable,
            "robustly_labelable": self.robustly_labelable,
            "rl_eligible": self.rl_eligible,
            "dagger_eligible": self.dagger_eligible,
            "expected_start_indices": list(self.expected_start_indices),
            "nominal": None if self.nominal is None else self.nominal.to_mapping(),
            "stress": None if self.stress is None else self.stress.to_mapping(),
            "reasons": list(self.reasons),
        }


def evaluate_mpcc_admission(
    *,
    expected_start_indices: Sequence[int],
    nominal_audit: Mapping[str, Any] | None,
    stress_audit: Mapping[str, Any] | None,
    thresholds: MPCCAdmissionThresholds | None = None,
) -> MPCCAdmissionDecision:
    """Evaluate canonical feasibility and expert-prefix labelability."""

    cfg = thresholds or MPCCAdmissionThresholds()
    expected = tuple(sorted({int(item) for item in expected_start_indices}))
    if not expected or expected[0] < 0:
        raise ValueError("expected physical start indices must be nonempty and valid")
    nominal = (
        None if nominal_audit is None
        else MPCCRegimeEvidence.from_track_audit("nominal", nominal_audit)
    )
    stress = (
        None if stress_audit is None
        else MPCCRegimeEvidence.from_track_audit("randomized_dart_stress", stress_audit)
    )
    reasons: list[str] = []

    physically_feasible = bool(
        nominal is not None and nominal.successful_canonical_laps >= 1
    )
    if not physically_feasible:
        reasons.append("no-successful-nominal-canonical-lap")

    nominal_starts_ok = False
    nominal_quality_ok = False
    if nominal is None:
        reasons.append("missing-nominal-mpcc-audit")
    else:
        observed = set(nominal.observed_start_indices)
        missing = sorted(set(expected) - observed)
        if missing:
            reasons.append("nominal-start-coverage-incomplete")
        repeat_counts = dict(nominal.episodes_by_start)
        if any(
            repeat_counts.get(index, 0) < cfg.minimum_nominal_repeats_per_start
            for index in expected
        ):
            reasons.append("nominal-starts-under-sampled")
        rates = dict(nominal.success_rate_by_start)
        if any(
            rates.get(index, 0.0) < cfg.minimum_nominal_start_success_rate
            for index in expected
        ):
            reasons.append("nominal-start-phase-failure")
        if nominal.success_rate < cfg.minimum_nominal_success_rate:
            reasons.append("nominal-success-rate")
        if (
            nominal.solver_failure_fraction
            > cfg.maximum_nominal_solver_failure_fraction
        ):
            reasons.append("nominal-solver-failure-rate")
        nominal_starts_ok = bool(
            not missing
            and all(
                repeat_counts.get(index, 0)
                >= cfg.minimum_nominal_repeats_per_start for index in expected
            )
            and all(
                rates.get(index, 0.0)
                >= cfg.minimum_nominal_start_success_rate for index in expected
            )
        )
        nominal_quality_ok = bool(
            nominal.success_rate >= cfg.minimum_nominal_success_rate
            and nominal.solver_failure_fraction
            <= cfg.maximum_nominal_solver_failure_fraction
        )
    teacher_labelable = bool(
        physically_feasible and nominal_starts_ok and nominal_quality_ok
    )

    stress_ok = False
    if stress is None:
        reasons.append("missing-randomized-dart-mpcc-audit")
    else:
        observed = set(stress.observed_start_indices)
        missing = sorted(set(expected) - observed)
        if missing:
            reasons.append("stress-start-coverage-incomplete")
        repeat_counts = dict(stress.episodes_by_start)
        if any(
            repeat_counts.get(index, 0) < cfg.minimum_stress_repeats_per_start
            for index in expected
        ):
            reasons.append("stress-starts-under-sampled")
        if stress.success_rate < cfg.minimum_stress_success_rate:
            reasons.append("stress-success-rate")
        if stress.solver_failure_fraction > cfg.maximum_stress_solver_failure_fraction:
            reasons.append("stress-solver-failure-rate")
        stress_ok = bool(
            not missing
            and all(
                repeat_counts.get(index, 0)
                >= cfg.minimum_stress_repeats_per_start for index in expected
            )
            and stress.success_rate >= cfg.minimum_stress_success_rate
            and stress.solver_failure_fraction
            <= cfg.maximum_stress_solver_failure_fraction
        )
    robustly_labelable = bool(teacher_labelable and stress_ok)
    return MPCCAdmissionDecision(
        physically_feasible=physically_feasible,
        teacher_labelable=teacher_labelable,
        robustly_labelable=robustly_labelable,
        expected_start_indices=expected,
        nominal=nominal,
        stress=stress,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def evaluate_record_dynamic_qualification(
    record: Mapping[str, Any],
    *,
    expected_start_indices: Sequence[int],
    thresholds: MPCCAdmissionThresholds | None = None,
) -> MPCCAdmissionDecision:
    """Evaluate the ``dynamic_qualification`` field of a frozen manifest row."""

    qualification = dict(record.get("dynamic_qualification") or {})
    return evaluate_mpcc_admission(
        expected_start_indices=expected_start_indices,
        nominal_audit=qualification.get("nominal"),
        stress_audit=qualification.get("randomized_dart_stress"),
        thresholds=thresholds,
    )
