"""Simple collection policies for pipeline debugging before expert MPC is connected."""

from __future__ import annotations

import numpy as np


class HoverPolicy:
    def __call__(self, observation) -> np.ndarray:
        return np.asarray([9.81, 0.0, 0.0, 0.0], dtype=np.float32)


class SmoothExcitationPolicy:
    """Deterministic low-amplitude CTBR excitation for dynamics smoke datasets."""

    def __init__(self, control_dt: float = 0.02, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.control_dt = float(control_dt)
        self.step = 0
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=4)

    def __call__(self, observation) -> np.ndarray:
        t = self.step * self.control_dt
        self.step += 1
        thrust = 9.81 + 0.8 * np.sin(0.7 * t + self.phases[0])
        rates = np.asarray([
            0.25 * np.sin(0.9 * t + self.phases[1]),
            0.25 * np.sin(0.6 * t + self.phases[2]),
            0.35 * np.sin(0.4 * t + self.phases[3]),
        ])
        return np.concatenate(([thrust], rates)).astype(np.float32)


class GateChasePolicy:
    """Privileged geometric teacher for diverse prototype racing rollouts.

    It is intentionally a collection baseline, not a claim of expert MPC quality.
    """

    def __init__(self, rate_gain: float = 2.0, forward_pitch: float = 0.35) -> None:
        self.rate_gain = float(rate_gain)
        self.forward_pitch = float(forward_pitch)

    def __call__(self, observation) -> np.ndarray:
        gate = np.asarray(observation["gates"]["position"][0], dtype=np.float32)
        distance_xy = max(float(np.hypot(gate[0], gate[1])), 0.25)
        yaw_error = float(np.arctan2(gate[1], max(gate[0], 0.1)))
        vertical_error = float(np.arctan2(gate[2], distance_xy))
        forward = float(np.clip(gate[0] / 8.0, 0.0, 1.0))
        rates = np.asarray(
            [
                -self.rate_gain * gate[1] / max(float(np.linalg.norm(gate)), 1.0),
                self.rate_gain * (self.forward_pitch * forward - vertical_error),
                self.rate_gain * yaw_error,
            ],
            dtype=np.float32,
        )
        thrust = 9.81 + np.clip(1.5 * gate[2], -2.0, 4.0) + 2.0 * forward
        return np.concatenate(([thrust], np.clip(rates, -4.0, 4.0))).astype(np.float32)
