"""Expert-controller contracts and deterministic collection fallbacks.

The geometric controller is deliberately identified as a fallback.  It exists
to smoke the exact dataset path that a licensed Agilicious MPCC adapter uses;
it is not a replacement for expert MPCC trajectories in training data.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from ..tracks import Track, matrix_quaternion, quaternion_matrix
from ..types import CTBRAction, ControllerCommand


@dataclass(slots=True)
class TrajectorySpawnSampler:
    track: Track
    seed: int = 0
    approach_distance: tuple[float, float] = (2.5, 5.0)
    lateral_offset: tuple[float, float] = (-0.6, 0.6)
    vertical_offset: tuple[float, float] = (-0.4, 0.4)
    forward_speed: tuple[float, float] = (0.0, 2.0)

    def sample(self, index: int, gate_index: int | None = None) -> tuple[np.ndarray, dict]:
        rng = np.random.default_rng(self.seed + 1009 * int(index))
        selected = int(index % len(self.track.gates) if gate_index is None else gate_index)
        gate = self.track.gates[selected % len(self.track.gates)]
        distance = float(rng.uniform(*self.approach_distance))
        lateral = float(rng.uniform(*self.lateral_offset))
        vertical = float(rng.uniform(*self.vertical_offset))
        speed = float(rng.uniform(*self.forward_speed))
        position = (
            gate.position
            - distance * gate.normal
            + lateral * gate.lateral
            + vertical * gate.up
        )
        margin = np.asarray([0.5, 0.5, 0.5], np.float32)
        position = np.clip(position, self.track.bounds[:, 0] + margin, self.track.bounds[:, 1] - margin)
        state = np.zeros(25, np.float32)
        state[0:3] = position
        state[3:7] = matrix_quaternion(gate.directed_rotation)
        state[7:10] = speed * gate.normal
        state[10:13] = rng.uniform(-0.1, 0.1, 3)
        metadata = {
            "gate_index": selected,
            "approach_distance": distance,
            "lateral_offset": lateral,
            "vertical_offset": vertical,
            "forward_speed": speed,
            "sampler": "trajectory-gate-frame-v1",
        }
        return state, metadata


class GeometricExpertPolicy:
    """Privileged attitude/velocity controller for end-to-end collector smoke tests."""

    def __init__(
        self,
        track: Track,
        *,
        target_speed: float = 5.0,
        velocity_gain: float = 1.8,
        attitude_gain: float = 5.0,
    ) -> None:
        self.track = track
        self.target_speed = float(target_speed)
        self.velocity_gain = float(velocity_gain)
        self.attitude_gain = float(attitude_gain)

    @staticmethod
    def _vee(matrix: np.ndarray) -> np.ndarray:
        return np.asarray(
            [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
            np.float32,
        ) * 0.5

    def __call__(self, observation) -> ControllerCommand:
        started = time.perf_counter()
        state = np.asarray(observation["privileged"]["state"], np.float32)
        progress = np.asarray(observation["privileged"]["progress"], np.float32)
        gate_index = int(round(float(progress[3]))) % len(self.track.gates)
        gate = self.track.gates[gate_index]
        position, velocity = state[0:3], state[7:10]
        to_gate = gate.position - position
        distance = float(np.linalg.norm(to_gate))
        direction = to_gate / max(distance, 1e-6)
        # Close to the aperture, prioritize crossing along its directed normal.
        blend = float(np.clip(1.0 - distance / 4.0, 0.0, 1.0))
        direction = (1.0 - blend) * direction + blend * gate.normal
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        desired_speed = self.target_speed * float(np.clip(distance / 1.5, 0.55, 1.0))
        reference_velocity = desired_speed * direction
        acceleration = self.velocity_gain * (reference_velocity - velocity)
        force = acceleration + np.asarray([0.0, 0.0, 9.81], np.float32)
        desired_z = force / max(float(np.linalg.norm(force)), 1e-6)
        heading = direction.copy()
        heading[2] = 0.0
        if np.linalg.norm(heading) < 1e-4:
            heading = gate.normal.copy()
        desired_y = np.cross(desired_z, heading)
        if np.linalg.norm(desired_y) < 1e-4:
            desired_y = np.cross(desired_z, np.asarray([0.0, 1.0, 0.0]))
        desired_y /= max(float(np.linalg.norm(desired_y)), 1e-6)
        desired_x = np.cross(desired_y, desired_z)
        desired_x /= max(float(np.linalg.norm(desired_x)), 1e-6)
        desired_rotation = np.stack([desired_x, desired_y, desired_z], axis=1)
        desired_quaternion = matrix_quaternion(desired_rotation)
        rotation = quaternion_matrix(state[3:7])
        attitude_error = self._vee(desired_rotation.T @ rotation - rotation.T @ desired_rotation)
        body_rates = np.clip(-self.attitude_gain * attitude_error, -5.5, 5.5)
        collective = float(np.clip(force @ rotation[:, 2], 2.0, 25.0))
        action = CTBRAction(collective, body_rates)
        reference_state = state.copy()
        reference_state[0:3] = gate.position
        reference_state[3:7] = desired_quaternion
        reference_state[7:10] = reference_velocity
        reference_state[10:13] = body_rates
        bounds_margin = float(
            np.min(np.concatenate([position - self.track.bounds[:, 0], self.track.bounds[:, 1] - position]))
        )
        timestamp = float(observation["timestamp"]["sim"])
        return ControllerCommand(
            action=action,
            source_timestamp=timestamp,
            receive_timestamp=timestamp,
            reference_state=reference_state,
            reference_action=action.as_array(),
            reference_progress=float(gate_index),
            solver_status=1,
            solve_time=float(time.perf_counter() - started),
            constraint_margin=bounds_margin,
            valid=True,
            source="geometric-expert-smoke-v1",
        )
