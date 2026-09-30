"""Adaptive task scheduling driven by learning progress, not course identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class TaskLearningState:
    task_id: str
    recent_returns: tuple[float, ...] = ()
    expert_return: float = 1.0
    learner_success: float = 0.0
    expert_success: float = 1.0
    novelty: float = 0.0
    rounds_since_sampled: int = 0
    mastered: bool = False

    @property
    def normalized_regret(self) -> float:
        if self.expert_success < 0.5:
            return 0.0
        current = self.recent_returns[-1] if self.recent_returns else 0.0
        scale = max(abs(float(self.expert_return)), 1.0)
        return float(np.clip((self.expert_return - current) / scale, 0.0, 2.0))

    @property
    def learning_progress(self) -> float:
        values = np.asarray(self.recent_returns[-8:], np.float64)
        if len(values) < 3:
            return 0.0
        x = np.linspace(-1.0, 1.0, len(values))
        slope = float((x @ (values - values.mean())) / max(x @ x, 1.0e-12))
        scale = max(abs(float(self.expert_return)), 1.0)
        return float(np.clip(slope / scale, -1.0, 1.0))


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0.0:
        return np.full(len(values), 1.0 / max(len(values), 1), np.float64)
    return values / total


def adaptive_task_probabilities(
    states: Sequence[TaskLearningState],
    *,
    coverage_weight: float = 0.45,
    learning_weight: float = 0.35,
    rehearsal_weight: float = 0.20,
) -> Mapping[str, float]:
    """Mix novelty, learnability/regret, and anti-forgetting rehearsal.

    Expert-infeasible tasks receive no probability.  Mastered tasks retain a
    rehearsal floor so a curriculum cannot silently erase prior competence.
    """

    if not states:
        raise ValueError("adaptive curriculum needs at least one task")
    if min(coverage_weight, learning_weight, rehearsal_weight) < 0.0:
        raise ValueError("curriculum mixture weights must be nonnegative")
    if coverage_weight + learning_weight + rehearsal_weight <= 0.0:
        raise ValueError("at least one curriculum mixture weight must be positive")
    feasible = np.asarray([item.expert_success >= 0.5 for item in states], np.bool_)
    if not feasible.any():
        raise ValueError("adaptive curriculum has no expert-feasible tasks")
    coverage = np.asarray([
        max(item.novelty, 0.0) + 0.05 / (1.0 + max(item.rounds_since_sampled, 0))
        for item in states
    ], np.float64)
    learning = np.asarray([
        max(item.learning_progress, 0.0) + item.normalized_regret
        for item in states
    ], np.float64)
    rehearsal = np.asarray([
        (1.0 + max(item.rounds_since_sampled, 0)) * (1.0 if item.mastered else 0.15)
        for item in states
    ], np.float64)
    for values in (coverage, learning, rehearsal):
        values[~feasible] = 0.0
    mixed = (
        coverage_weight * _normalize(coverage)
        + learning_weight * _normalize(learning)
        + rehearsal_weight * _normalize(rehearsal)
    )
    mixed[~feasible] = 0.0
    mixed = _normalize(mixed)
    return {item.task_id: float(value) for item, value in zip(states, mixed)}
