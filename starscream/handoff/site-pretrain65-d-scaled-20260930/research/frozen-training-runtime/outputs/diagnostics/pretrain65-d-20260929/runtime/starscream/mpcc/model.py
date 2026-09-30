"""NumPy port of Flightmare's pinned CTBR prediction chain.

The implementation follows uzh-rpg/flightmare commit d4218ae: body-rate P
control, allocation, rotor clipping, motor first-order response, and RK4 state
integration with a maximum 2.5 ms physics step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..env.tracks import quaternion_matrix
from ..env.types import CTBRAction
from .config import VehicleModelConfig


F64 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ModelState:
    state: F64
    motor_omega: F64

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float64)
        motors = np.asarray(self.motor_omega, dtype=np.float64)
        if state.shape != (25,) or motors.shape != (4,):
            raise ValueError("ModelState requires state=(25,) and motor_omega=(4,)")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(motors)):
            raise ValueError("model state must be finite")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "motor_omega", motors)


def _skew_cross(left: F64, right: F64) -> F64:
    return np.cross(left, right)


def _quaternion_derivative(quaternion: F64, omega: F64) -> F64:
    w, x, y, z = quaternion
    wx, wy, wz = omega
    return 0.5 * np.asarray(
        [
            -x * wx - y * wy - z * wz,
            w * wx + y * wz - z * wy,
            w * wy - x * wz + z * wx,
            w * wz + x * wy - y * wx,
        ],
        dtype=np.float64,
    )


class FlightmareCTBRModel:
    """Deterministic prediction model with the simulator's 25-state layout."""

    def __init__(self, config: VehicleModelConfig | None = None, *, implementation: str = "numpy") -> None:
        self.config = config or VehicleModelConfig()
        cfg = self.config
        self.inertia = cfg.inertia
        self.inertia_inverse = 1.0 / self.inertia
        arm = cfg.arm_length * np.sqrt(0.5)
        rotor_moment = arm * np.asarray(
            [[1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]],
            dtype=np.float64,
        )
        self.allocation = np.vstack(
            [
                np.ones(4, dtype=np.float64),
                rotor_moment,
                cfg.kappa * np.asarray([1.0, -1.0, 1.0, -1.0]),
            ]
        )
        self.allocation_inverse = np.linalg.inv(self.allocation)
        self.rate_gain = np.asarray(cfg.rate_gain, dtype=np.float64)
        self.body_rate_max = np.asarray(cfg.body_rate_max, dtype=np.float64)
        self.thrust_map = np.asarray(cfg.thrust_map, dtype=np.float64)
        self.thrust_min = 0.0
        self.thrust_max = float(self.motor_omega_to_thrust(cfg.effective_thrust_omega_max))
        self.collective_acceleration_max = 4.0 * self.thrust_max / cfg.mass
        if implementation not in {"numpy", "native", "native_exact"}:
            raise ValueError("prediction implementation must be numpy, native or native_exact")
        self.implementation = implementation
        self.native_prediction = None
        self.native_full_step = False
        if implementation != "numpy":
            from .native_prediction import NativePrediction
            self.native_prediction = NativePrediction(self)

    def motor_omega_to_thrust(self, omega: ArrayLike) -> F64:
        value = np.asarray(omega, dtype=np.float64)
        a, b, c = self.thrust_map
        return a * value * value + b * value + c

    def motor_thrust_to_omega(self, thrust: ArrayLike) -> F64:
        value = np.asarray(thrust, dtype=np.float64)
        a, b, c = self.thrust_map
        discriminant = np.maximum(b * b - 4.0 * a * (c - value), 0.0)
        return (-b + np.sqrt(discriminant)) / (2.0 * a)

    def _commanded_motor_thrusts(self, state: F64, action: CTBRAction) -> F64:
        cfg = self.config
        rates = np.clip(action.body_rates.astype(np.float64), -self.body_rate_max, self.body_rate_max)
        collective = float(
            np.clip(action.collective_thrust, 0.0, self.collective_acceleration_max)
        )
        omega = state[10:13]
        desired_torque = (
            self.inertia * self.rate_gain * (rates - omega)
            + (self.native_prediction.cross(omega, self.inertia * omega) if self.implementation == "native_exact" else _skew_cross(omega, self.inertia * omega))
        )
        wrench = np.concatenate(([cfg.mass * collective], desired_torque))
        desired = self.allocation_inverse @ wrench
        return np.clip(desired, self.thrust_min, self.thrust_max)

    def _derivative(self, state: F64) -> F64:
        derivative = np.zeros(25, dtype=np.float64)
        derivative[0:3] = state[7:10]
        derivative[3:7] = _quaternion_derivative(state[3:7], state[10:13])
        derivative[7:10] = state[13:16]
        omega = state[10:13]
        derivative[10:13] = self.inertia_inverse * (
            state[16:19] - _skew_cross(omega, self.inertia * omega)
        )
        return derivative

    def _rk4(self, state: F64, dt: float) -> F64:
        if self.implementation == "native_exact":
            return self.native_prediction.rk4(state,dt)
        k1 = self._derivative(state)
        k2 = self._derivative(state + 0.5 * dt * k1)
        k3 = self._derivative(state + 0.5 * dt * k2)
        k4 = self._derivative(state + dt * k3)
        return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, model_state: ModelState, action: ArrayLike | CTBRAction, dt: float) -> ModelState:
        """Advance one control step using the same substep ordering as Flightmare."""

        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and positive")
        command = action if isinstance(action, CTBRAction) else CTBRAction.from_array(action)
        if self.native_full_step:
            if self.implementation != 'native_exact':
                raise ValueError('full exact step requires native_exact prediction')
            state, motors = self.native_prediction.step_exact(
                model_state.state, model_state.motor_omega,
                np.asarray([command.collective_thrust, *command.body_rates], np.float64), dt)
            return ModelState(state, motors)
        if self.implementation == "native":
            state, motors = self.native_prediction.step(model_state.state, model_state.motor_omega, command.as_array(), dt)
            return ModelState(state, motors)
        state = model_state.state.copy()
        motors = model_state.motor_omega.copy()
        # Flightmare's Scalar is float32.  For a 20 ms control step the
        # repeated 2.5 ms subtraction leaves a tiny positive ninth substep;
        # that substep refreshes acceleration/torque telemetry at the final
        # attitude even though it has negligible integration duration.
        remaining = np.float32(dt)
        maximum_substep = np.float32(self.config.integration_dt_max)
        while remaining > 0.0:
            sim_dt_scalar = np.minimum(remaining, maximum_substep)
            sim_dt = float(sim_dt_scalar)
            motor_thrust_desired = self._commanded_motor_thrusts(state, command)
            motor_omega_desired = np.clip(
                self.motor_thrust_to_omega(motor_thrust_desired),
                self.config.motor_omega_min,
                self.config.motor_omega_max,
            )
            decay = np.exp(-sim_dt / self.config.motor_tau)
            motors = decay * motors + (1.0 - decay) * motor_omega_desired
            motor_thrusts = np.clip(
                self.motor_omega_to_thrust(motors), self.thrust_min, self.thrust_max
            )
            wrench = self.allocation @ motor_thrusts
            rotation = quaternion_matrix(state[3:7]).astype(np.float64)
            air_velocity_body = rotation.T @ (
                state[7:10] - np.asarray(self.config.wind_world, np.float64)
            )
            drag_force = (
                -np.asarray(self.config.linear_drag) * air_velocity_body
                -np.asarray(self.config.quadratic_drag)
                * np.abs(air_velocity_body) * air_velocity_body
                -float(np.sum(motors)) * np.asarray(self.config.rotor_drag)
                * air_velocity_body
            )
            body_force = np.asarray([0.0, 0.0, wrench[0]], dtype=np.float64) + drag_force
            state[13:16] = (
                rotation @ body_force
                / self.config.mass
                + np.asarray([0.0, 0.0, -self.config.gravity])
            )
            center_of_mass = np.asarray(self.config.center_of_mass, np.float64)
            aerodynamic_torque = (
                -np.asarray(self.config.angular_drag) * state[10:13]
                + (self.native_prediction.cross(center_of_mass, np.asarray([0.0, 0.0, wrench[0]])) if self.implementation == "native_exact" else np.cross(center_of_mass, np.asarray([0.0, 0.0, wrench[0]])))
            )
            state[16:19] = wrench[1:4] + aerodynamic_torque
            state = self._rk4(state, sim_dt)
            remaining = np.float32(remaining - sim_dt_scalar)
        # Flightmare's pinned normalization is accidentally applied to the old
        # state before assignment. Normalize here only if drift threatens the
        # rotation calculation; normal rollouts remain bit-close without it.
        norm = float(np.linalg.norm(state[3:7]))
        if abs(norm - 1.0) > 1e-5:
            state[3:7] /= max(norm, 1e-12)
        return ModelState(state, motors)

    def rollout(
        self,
        initial: ModelState,
        actions: Iterable[ArrayLike | CTBRAction],
        time_steps: Iterable[float],
    ) -> tuple[F64, F64, F64]:
        actions_list = list(actions)
        steps = np.asarray(list(time_steps), dtype=np.float64)
        if len(actions_list) != len(steps):
            raise ValueError("actions and time_steps must have equal length")
        states = [initial.state.copy()]
        motors = [initial.motor_omega.copy()]
        current = initial
        applied_actions: list[F64] = []
        for action, dt in zip(actions_list, steps, strict=True):
            parsed = action if isinstance(action, CTBRAction) else CTBRAction.from_array(action)
            applied_actions.append(parsed.as_array().astype(np.float64))
            current = self.step(current, parsed, float(dt))
            states.append(current.state.copy())
            motors.append(current.motor_omega.copy())
        return np.stack(states), np.stack(motors), np.stack(applied_actions)
