"""Competence-bounded admission for manifold curriculum candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def wilson_interval(successes: int, episodes: int, z: float = 1.96) -> tuple[float, float]:
    if episodes < 1 or not 0 <= successes <= episodes:
        raise ValueError("invalid Bernoulli counts")
    rate = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (rate + z * z / (2.0 * episodes)) / denominator
    half = z * np.sqrt(rate * (1.0 - rate) / episodes + z * z / (4.0 * episodes**2)) / denominator
    return float(max(0.0, center - half)), float(min(1.0, center + half))


@dataclass(frozen=True, slots=True)
class FrontierCalibration:
    status: str
    baseline_success: float
    target_success_band: tuple[float, float]
    selected_step: float | None
    selected_success: float | None
    selected_interval: tuple[float, float] | None
    sampling_mixture: dict[str, float]
    reason: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "baseline_success": self.baseline_success,
            "target_success_band": list(self.target_success_band),
            "selected_step": self.selected_step,
            "selected_success": self.selected_success,
            "selected_interval": None if self.selected_interval is None else list(self.selected_interval),
            "sampling_mixture": self.sampling_mixture,
            "reason": self.reason,
        }


def calibrate_frontier(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_mastered_success: float = 0.60,
    relative_band: tuple[float, float] = (0.60, 0.85),
) -> FrontierCalibration:
    """Choose a harder-but-alive rung from a closed-loop ladder.

    The mastered source must first clear ``minimum_mastered_success``. A
    candidate is selected only when its empirical completion lies inside a
    relative success band. Wilson intervals are reported so low-count screens
    cannot be mistaken for a precise trust-region estimate.
    """

    if not rows or "policy" not in rows[0]:
        raise ValueError("ladder rows need policy metrics")
    baseline = float(rows[0]["policy"].get("full_course_success", 0.0))
    low, high = relative_band[0] * baseline, relative_band[1] * baseline
    if baseline < minimum_mastered_success:
        return FrontierCalibration(
            status="baseline-not-mastered", baseline_success=baseline,
            target_success_band=(low, high), selected_step=None,
            selected_success=None, selected_interval=None,
            sampling_mixture={"core": 0.80, "frontier": 0.0, "recovery": 0.20},
            reason=(
                f"source success {baseline:.3f} is below the mastered threshold "
                f"{minimum_mastered_success:.3f}; do not expand the manifold"
            ),
        )
    candidates = []
    for row in rows[1:]:
        policy = row.get("policy", {})
        episodes = int(round(float(policy.get("episodes", 0.0))))
        success = float(policy.get("full_course_success", 0.0))
        if episodes < 1:
            continue
        successes = int(np.clip(round(success * episodes), 0, episodes))
        interval = wilson_interval(successes, episodes)
        if low <= success <= high:
            candidates.append((float(row["standardized_step"]), success, interval))
    if not candidates:
        return FrontierCalibration(
            status="refine-bracket", baseline_success=baseline,
            target_success_band=(low, high), selected_step=None,
            selected_success=None, selected_interval=None,
            sampling_mixture={"core": 0.65, "frontier": 0.15, "recovery": 0.20},
            reason="no evaluated rung lies in the target competence band; bisect the nearest bracket",
        )
    selected = max(candidates, key=lambda item: item[0])
    return FrontierCalibration(
        status="admit-frontier", baseline_success=baseline,
        target_success_band=(low, high), selected_step=selected[0],
        selected_success=selected[1], selected_interval=selected[2],
        sampling_mixture={"core": 0.50, "frontier": 0.30, "recovery": 0.20},
        reason="candidate is hard enough to create policy gradient while retaining closed-loop competence",
    )
