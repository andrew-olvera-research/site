"""Dense, auditable rewards for perception-aware autonomous drone racing.

The task term is a potential difference in ordered course progress.  Secondary
perception and command terms are normalized and bounded so they cannot make a
fast, clean lap worse than hovering or flying slowly with a level attitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike


def _quaternion_matrix(quaternion_wxyz: ArrayLike) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float32)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or norm < 1.0e-8 or not np.isfinite(norm):
        raise ValueError("reward attitude must be a finite non-zero wxyz quaternion")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True, slots=True)
class RaceRewardConfig:
    """Reward coefficients in physical units.

    ``course_progress`` is measured in metres.  Rate and action-difference
    penalties use dimensionless values normalized by the declared limits and
    are clipped to one before weighting.
    """

    progress_weight: float = 2.0
    progress_clip_m: float = 1.0
    perception_weight: float = 0.01
    perception_angle_scale: float = -2.0
    body_rate_weight: float = 0.01
    action_smoothness_weight: float = 0.01
    time_penalty: float = 0.01
    gate_pass_bonus: float = 5.0
    crash_penalty: float = 10.0
    body_rate_scale: float = 6.0
    collective_thrust_scale: float = 15.0

    def __post_init__(self) -> None:
        nonnegative = (
            self.progress_weight,
            self.perception_weight,
            self.body_rate_weight,
            self.action_smoothness_weight,
            self.time_penalty,
            self.gate_pass_bonus,
            self.crash_penalty,
        )
        if any(value < 0 or not np.isfinite(value) for value in nonnegative):
            raise ValueError("reward weights must be finite and non-negative")
        if min(self.progress_clip_m, self.body_rate_scale, self.collective_thrust_scale) <= 0:
            raise ValueError("reward normalization scales must be positive")
        if self.perception_angle_scale > 0 or not np.isfinite(self.perception_angle_scale):
            raise ValueError("perception_angle_scale must be finite and non-positive")


@dataclass(frozen=True, slots=True)
class RewardResult:
    total: float
    components: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class SkyDreamerRewardConfig:
    """SkyDreamer racing reward in physical units.

    The published controller uses ``5 * progress - rate + 30 * gate`` at
    90 Hz.  ``gate_clearance_m`` is the half-aperture after subtracting the
    frame; 0.8 m is the value reported in the paper.
    """

    progress_weight: float = 5.0
    gate_weight: float = 30.0
    gate_clearance_m: float = 0.8
    control_frequency_hz: float = 90.0
    body_rate_clip: float = 17.0
    body_rate_denominator: float = 1.0e5
    crash_penalty: float = 0.0

    def __post_init__(self) -> None:
        nonnegative = (
            self.progress_weight,
            self.gate_weight,
            self.crash_penalty,
        )
        if any(value < 0 or not np.isfinite(value) for value in nonnegative):
            raise ValueError("SkyDreamer reward weights must be finite and non-negative")
        positive = (
            self.gate_clearance_m,
            self.control_frequency_hz,
            self.body_rate_clip,
            self.body_rate_denominator,
        )
        if any(value <= 0 or not np.isfinite(value) for value in positive):
            raise ValueError("SkyDreamer reward scales must be finite and positive")


class SkyDreamerRaceReward:
    """State-estimation-friendly reward from the SkyDreamer formulation.

    Ordered course progress is used as the continuous potential because it is
    invariant to Starscream's active-gate switch.  The gate term preserves the
    paper's centered-crossing quality and the rate term uses measured angular
    velocity, rather than the requested CTBR command.
    """

    def __init__(self, config: SkyDreamerRewardConfig | None = None) -> None:
        self.config = config or SkyDreamerRewardConfig()
        self._previous_distance: float | None = None
        self._previous_course_progress: float | None = None

    def reset(
        self,
        position_world: ArrayLike,
        active_gate: Any,
        *,
        course_progress: float | None = None,
    ) -> None:
        position = np.asarray(position_world, dtype=np.float32)
        self._previous_distance = float(np.linalg.norm(active_gate.position - position))
        self._previous_course_progress = (
            None if course_progress is None else float(course_progress)
        )

    def __call__(
        self,
        *,
        state: ArrayLike,
        action: Any,
        active_gate: Any,
        gate_passed: bool,
        crashed: bool,
        course_progress: float | None = None,
    ) -> RewardResult:
        del action
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (25,):
            raise ValueError("reward state must use the 25-value Flightmare layout")
        if not np.all(np.isfinite(state)):
            raise ValueError("reward state must be finite")

        delta_world = state[0:3] - np.asarray(active_gate.position, np.float32)
        distance = float(np.linalg.norm(delta_world))
        if course_progress is not None:
            current_progress = float(course_progress)
            previous = (
                current_progress
                if self._previous_course_progress is None
                else self._previous_course_progress
            )
            progress_delta = current_progress - previous
        else:
            current_progress = None
            previous_distance = distance if self._previous_distance is None else self._previous_distance
            progress_delta = previous_distance - distance

        cfg = self.config
        rate_l1 = float(np.abs(state[10:13]).sum())
        rate_penalty = (
            np.exp(min(rate_l1, cfg.body_rate_clip)) - 1.0
        ) / (2.0 * cfg.control_frequency_hz * cfg.body_rate_denominator)
        gate_position = active_gate.directed_rotation.T @ delta_world
        gate_quality = (
            max(
                0.0,
                1.0
                - max(abs(float(gate_position[1])), abs(float(gate_position[2])))
                / cfg.gate_clearance_m,
            )
            if gate_passed
            else 0.0
        )
        progress = cfg.progress_weight * progress_delta
        gate = cfg.gate_weight * gate_quality
        crash = -cfg.crash_penalty if crashed else 0.0
        components = {
            "progress": float(progress),
            "progress_delta": float(progress_delta),
            "body_rate": float(-rate_penalty),
            "body_rate_l1": rate_l1,
            "gate_pass": float(gate),
            "gate_quality": float(gate_quality),
            "crash": float(crash),
            "gate_distance": distance,
            "course_progress": float(current_progress if current_progress is not None else 0.0),
            "course_progress_valid": float(current_progress is not None),
        }
        total = progress - rate_penalty + gate + crash
        self._previous_distance = None if gate_passed else distance
        self._previous_course_progress = current_progress
        return RewardResult(float(total), components)


class PerceptionAwareRaceReward:
    """Dense racing reward with ordered progress as the dominant objective.

    Environments should provide an unwrapped ``course_progress`` potential.  A
    distance-to-active-gate fallback remains for small test environments and
    older integrations; it detects gate changes and deliberately emits zero
    progress on the switch step instead of the old large negative discontinuity.
    """

    def __init__(self, config: RaceRewardConfig | None = None) -> None:
        self.config = config or RaceRewardConfig()
        self._previous_distance: float | None = None
        self._previous_course_progress: float | None = None
        self._active_gate_key: tuple[float, ...] | None = None
        self._previous_action = np.asarray([9.81, 0.0, 0.0, 0.0], dtype=np.float32)

    @staticmethod
    def _gate_key(active_gate: Any) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(active_gate.position, dtype=np.float64), 5))

    def reset(
        self,
        position_world: ArrayLike,
        active_gate: Any,
        *,
        course_progress: float | None = None,
    ) -> None:
        position = np.asarray(position_world, dtype=np.float32)
        self._previous_distance = float(np.linalg.norm(active_gate.position - position))
        self._previous_course_progress = (
            None if course_progress is None else float(course_progress)
        )
        self._active_gate_key = self._gate_key(active_gate)
        self._previous_action[:] = [9.81, 0.0, 0.0, 0.0]

    def __call__(
        self,
        *,
        state: ArrayLike,
        action: Any,
        active_gate: Any,
        gate_passed: bool,
        crashed: bool,
        course_progress: float | None = None,
    ) -> RewardResult:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (25,):
            raise ValueError("reward state must use the 25-value Flightmare layout")
        action_array = (
            np.asarray(action.as_array(), dtype=np.float32)
            if hasattr(action, "as_array")
            else np.asarray(action, dtype=np.float32)
        )
        if action_array.shape != (4,) or not np.all(np.isfinite(action_array)):
            raise ValueError("reward action must be finite CTBR with shape (4,)")

        delta_world = np.asarray(active_gate.position, np.float32) - state[0:3]
        distance = float(np.linalg.norm(delta_world))
        gate_key = self._gate_key(active_gate)
        target_changed = self._active_gate_key is not None and gate_key != self._active_gate_key

        if course_progress is not None:
            current_progress = float(course_progress)
            if not np.isfinite(current_progress):
                raise ValueError("course_progress must be finite")
            previous_progress = (
                current_progress
                if self._previous_course_progress is None
                else self._previous_course_progress
            )
            progress_delta = current_progress - previous_progress
        else:
            previous_distance = (
                distance if self._previous_distance is None or target_changed else self._previous_distance
            )
            progress_delta = previous_distance - distance
            current_progress = None

        # Camera optical axis is body +X in the Starscream camera contract.
        delta_body = _quaternion_matrix(state[3:7]).T @ delta_world
        cosine = (
            float(np.clip(delta_body[0] / distance, -1.0, 1.0))
            if distance > 1.0e-6
            else 1.0
        )
        camera_angle = float(np.arccos(cosine))

        cfg = self.config
        clipped_progress_delta = float(
            np.clip(progress_delta, -cfg.progress_clip_m, cfg.progress_clip_m)
        )
        progress = cfg.progress_weight * clipped_progress_delta
        perception = cfg.perception_weight * float(
            np.exp(cfg.perception_angle_scale * camera_angle**4)
        )
        normalized_rates = action_array[1:4] / cfg.body_rate_scale
        body_rate = -cfg.body_rate_weight * float(
            np.clip(np.mean(normalized_rates**2), 0.0, 1.0)
        )
        action_scale = np.asarray(
            [cfg.collective_thrust_scale, cfg.body_rate_scale, cfg.body_rate_scale, cfg.body_rate_scale],
            dtype=np.float32,
        )
        normalized_delta = (action_array - self._previous_action) / action_scale
        smoothness = -cfg.action_smoothness_weight * float(
            np.clip(np.mean(normalized_delta**2), 0.0, 1.0)
        )
        time = -cfg.time_penalty
        gate = cfg.gate_pass_bonus if gate_passed else 0.0
        crash = -cfg.crash_penalty if crashed else 0.0
        components = {
            "progress": float(progress),
            "perception": float(perception),
            "body_rate": float(body_rate),
            "smoothness": float(smoothness),
            "time": float(time),
            "gate_pass": float(gate),
            "crash": float(crash),
            "camera_angle": camera_angle,
            "gate_distance": distance,
            "course_progress": float(current_progress if current_progress is not None else 0.0),
            "course_progress_valid": float(current_progress is not None),
            "progress_delta": clipped_progress_delta,
        }
        total = sum(
            components[key]
            for key in ("progress", "perception", "body_rate", "smoothness", "time", "gate_pass", "crash")
        )
        self._previous_distance = None if gate_passed else distance
        self._previous_course_progress = current_progress
        self._active_gate_key = None if gate_passed else gate_key
        self._previous_action = action_array.copy()
        return RewardResult(float(total), components)
