"""State-estimator contracts used by collection and reconstruction targets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from .types import Proprioception
from .tracks import matrix_quaternion, quaternion_matrix


F32 = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class StateEstimate:
    state: F32
    standard_deviation: F32
    valid: bool = True


class StateEstimator(Protocol):
    def reset(self, seed: int | None = None) -> None: ...
    def estimate(self, proprioception: Proprioception) -> StateEstimate: ...


class SimulatorStateEstimator:
    """Ground-truth feedthrough estimator used until VIO/EKF is connected.

    It intentionally writes a separate estimate field, so future real estimator
    outputs can replace it without changing the dataset or model contract.
    """

    def reset(self, seed: int | None = None) -> None:
        return None

    def estimate(self, proprioception: Proprioception) -> StateEstimate:
        return StateEstimate(
            state=proprioception.state.copy(),
            standard_deviation=np.zeros(25, dtype=np.float32),
            valid=True,
        )


@dataclass(frozen=True, slots=True)
class StateEstimatorRandomizationConfig:
    """Mild motion-capture/VIO error envelope for deployment-facing control."""

    enabled: bool = True
    latency_seconds: tuple[float, float] = (0.0, 0.022)
    position_bias_std_m: float = 0.015
    position_noise_std_m: float = 0.008
    velocity_bias_std_mps: float = 0.025
    velocity_noise_std_mps: float = 0.040
    attitude_bias_std_degrees: float = 0.20
    attitude_noise_std_degrees: float = 0.12
    body_rate_bias_std_rps: float = 0.010
    body_rate_noise_std_rps: float = 0.015

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None,
    ) -> "StateEstimatorRandomizationConfig | None":
        return None if value is None else cls(**dict(value))

    def __post_init__(self) -> None:
        low, high = self.latency_seconds
        scales = (
            self.position_bias_std_m, self.position_noise_std_m,
            self.velocity_bias_std_mps, self.velocity_noise_std_mps,
            self.attitude_bias_std_degrees, self.attitude_noise_std_degrees,
            self.body_rate_bias_std_rps, self.body_rate_noise_std_rps,
        )
        if low < 0.0 or high < low or any(value < 0.0 for value in scales):
            raise ValueError("state-estimator randomization scales must be non-negative")


class RandomizedStateEstimator:
    """Deterministic per-episode delayed, biased and noisy state estimator."""

    def __init__(
        self, config: StateEstimatorRandomizationConfig, *, control_dt: float,
    ) -> None:
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        self.config = config
        self.control_dt = float(control_dt)
        self._rng = np.random.default_rng(0)
        self._states: deque[np.ndarray] = deque()
        self._delay_steps = 0
        self._position_bias = np.zeros(3, np.float32)
        self._velocity_bias = np.zeros(3, np.float32)
        self._attitude_bias = np.zeros(3, np.float32)
        self._body_rate_bias = np.zeros(3, np.float32)
        self.reset(0)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(0 if seed is None else int(seed) ^ 0x51A7E)
        self._states.clear()
        config = self.config
        latency = float(self._rng.uniform(*config.latency_seconds)) if config.enabled else 0.0
        self._delay_steps = int(round(latency / self.control_dt))
        self._position_bias = self._rng.normal(0.0, config.position_bias_std_m, 3).astype(np.float32)
        self._velocity_bias = self._rng.normal(0.0, config.velocity_bias_std_mps, 3).astype(np.float32)
        self._attitude_bias = self._rng.normal(
            0.0, np.radians(config.attitude_bias_std_degrees), 3,
        ).astype(np.float32)
        self._body_rate_bias = self._rng.normal(0.0, config.body_rate_bias_std_rps, 3).astype(np.float32)

    @staticmethod
    def _error_rotation(angles: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = angles
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.asarray([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], np.float32)

    def estimate(self, proprioception: Proprioception) -> StateEstimate:
        self._states.append(proprioception.state.copy())
        while len(self._states) > self._delay_steps + 1:
            self._states.popleft()
        state = self._states[0].copy()
        config = self.config
        if config.enabled:
            state[0:3] += self._position_bias + self._rng.normal(
                0.0, config.position_noise_std_m, 3,
            )
            state[7:10] += self._velocity_bias + self._rng.normal(
                0.0, config.velocity_noise_std_mps, 3,
            )
            angles = self._attitude_bias + self._rng.normal(
                0.0, np.radians(config.attitude_noise_std_degrees), 3,
            )
            state[3:7] = matrix_quaternion(
                quaternion_matrix(state[3:7]) @ self._error_rotation(angles)
            )
            state[10:13] += self._body_rate_bias + self._rng.normal(
                0.0, config.body_rate_noise_std_rps, 3,
            )
        standard_deviation = np.zeros(25, np.float32)
        standard_deviation[0:3] = config.position_noise_std_m
        standard_deviation[3:7] = np.radians(config.attitude_noise_std_degrees)
        standard_deviation[7:10] = config.velocity_noise_std_mps
        standard_deviation[10:13] = config.body_rate_noise_std_rps
        return StateEstimate(state.astype(np.float32), standard_deviation, True)
