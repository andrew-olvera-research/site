"""Recovery-aware MPCC policy producing Starscream's native CTBR contract."""

from __future__ import annotations

from dataclasses import replace
import copy
from enum import IntEnum
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..env.tracks import Track, matrix_quaternion, quaternion_matrix
from ..env.types import CTBRAction, ControllerCommand
from .acados_backend import (
    MOTOR_OMEGA_SCALE,
    AcadosRTIBackend,
    AcadosResult,
    acados_available,
)
from .config import MPCCConfig, VehicleModelConfig
from .model import FlightmareCTBRModel, ModelState
from .racing_line import Projection, RacingLine


F64 = NDArray[np.float64]


class MPCCMode(IntEnum):
    NOMINAL = 0
    RECOVERY = 1
    INFEASIBLE = 2


def _vee(matrix: F64) -> F64:
    return 0.5 * np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
    )


class MPCCController:
    """Privileged expert controller with acados RTI and deterministic fallback.

    The fallback uses the exact prediction model and the same spatial reference;
    it is intended for model/debug benchmarks, not expert-data qualification.
    """

    def __init__(
        self,
        track: Track,
        racing_line: RacingLine,
        *,
        config: MPCCConfig | None = None,
        vehicle: VehicleModelConfig | None = None,
        build_directory: str | Path | None = None,
        backend_instance: AcadosRTIBackend | None = None,
        diagnostics_level: str = "full",
        prediction_backend: str = "numpy",
        native_reference: bool = False,
        native_projection: bool = False,
        cache_runtime_vehicle: bool = False,
        native_model_step: bool = False,
    ) -> None:
        if racing_line.track_fingerprint != track.fingerprint:
            raise ValueError("racing line does not belong to the supplied track")
        self.track = track
        # Runtime accelerators belong to this controller, not every controller
        # that happens to share the same immutable spline coefficients.
        self.line = copy.copy(racing_line) if native_reference or native_projection else racing_line
        if native_reference:
            from .native_prediction import NativeCubicTriplet
            self.line._native_triplet = NativeCubicTriplet(self.line._position_spline)
        if native_projection:
            from .native_prediction import NativeProgressProjection
            self.line._native_projection = NativeProgressProjection(self.line)
        self.cache_runtime_vehicle = bool(cache_runtime_vehicle)
        self._runtime_vehicle_key = None
        self._runtime_vehicle_value = None
        self._runtime_vehicle_defaults = None
        self.config = config or MPCCConfig()
        if diagnostics_level not in {"full", "targets", "minimal"}:
            raise ValueError(
                "diagnostics_level must be 'full', 'targets', or 'minimal'"
            )
        self.diagnostics_level = diagnostics_level
        self.vehicle = vehicle or VehicleModelConfig()
        self._vehicle_defaults = self.vehicle
        self.prediction_backend = prediction_backend
        if native_model_step and prediction_backend != 'native_exact':
            raise ValueError('native_model_step requires native_exact prediction')
        self.native_model_step = bool(native_model_step)
        self.model = FlightmareCTBRModel(self.vehicle, implementation=prediction_backend)
        self.model.native_full_step = self.native_model_step
        self._last_progress: float | None = None
        self._last_gate_index: int | None = None
        self._last_position: F64 | None = None
        self._missed_gate_recovery = False
        self._last_solver_success = False
        self._previous_action = np.asarray([self.vehicle.gravity, 0.0, 0.0, 0.0])
        self._speed_progress, self._speed_profile = self._build_speed_profile()
        self._backend: AcadosRTIBackend | None = backend_instance
        if self._backend is not None:
            self._backend.set_world_bounds(track.bounds)
            self._backend.set_vehicle(self.vehicle)
        wants_acados = self.config.backend == "acados" or (
            self.config.backend == "auto" and acados_available()
        )
        if wants_acados and self._backend is None:
            if build_directory is None:
                root = Path(os.environ.get("STARSCREAM_ACADOS_CACHE", "/tmp/starscream-acados"))
                build_directory = root / f"{track.fingerprint[:12]}-{self.config.fingerprint[:12]}"
            try:
                self._backend = AcadosRTIBackend(
                    self.config,
                    self.vehicle,
                    track.bounds.astype(np.float64),
                    build_directory=build_directory,
                )
            except Exception:
                if self.config.backend == "acados":
                    raise
                self._backend = None

    @property
    def backend_instance(self) -> AcadosRTIBackend | None:
        return self._backend

    def _synchronize_runtime_vehicle(
        self, observation: dict[str, Any], privileged: dict[str, Any]
    ) -> None:
        dynamics = privileged.get("dynamics")
        aerodynamics = privileged.get("aerodynamics")
        if dynamics is None or aerodynamics is None:
            return
        timestamp = observation.get("timestamp", {}).get(
            "sim", observation.get("time", 0.0)
        )
        key = None
        if self.cache_runtime_vehicle:
            dynamics = np.asarray(dynamics, np.float64)
            aerodynamics = np.asarray(aerodynamics, np.float64)
            if dynamics.shape != (15,) or aerodynamics.shape != (27,):
                raise ValueError("runtime vehicle parameters require dynamics=(15,), aerodynamics=(27,)")
            # Effective wind, not time or array identity: gusts and in-place
            # domain-randomization edits must immediately invalidate this cache.
            wind = aerodynamics[12:15] + aerodynamics[15:18] * np.sin(
                2.0 * np.pi * aerodynamics[18:21] * float(timestamp) + aerodynamics[21:24])
            key = (dynamics.tobytes(), aerodynamics.tobytes(), wind.tobytes())
            if (key == self._runtime_vehicle_key
                    and self.vehicle is self._runtime_vehicle_value
                    and self._vehicle_defaults is self._runtime_vehicle_defaults):
                return
            if (self._runtime_vehicle_key is not None
                    and key[:2] == self._runtime_vehicle_key[:2]
                    and self.vehicle is self._runtime_vehicle_value
                    and self._vehicle_defaults is self._runtime_vehicle_defaults):
                # A gust changes only effective wind, not mass/inertia, rotor
                # allocation, thrust maps, BLAS bindings or integration decay.
                # Keep those exact derived values instead of rebuilding the
                # entire prediction model and hashing two dataclasses per tick.
                self.vehicle = replace(self.vehicle, wind_world=tuple(float(x) for x in wind))
                self.model.config = self.vehicle
                if self.model.native_prediction is not None:
                    self.model.native_prediction.parameters[51:54] = wind
                if self._backend is not None:
                    self._backend.set_vehicle(self.vehicle)
                self._runtime_vehicle_key = key
                self._runtime_vehicle_value = self.vehicle
                return
        vehicle = VehicleModelConfig.from_observation_parameters(
            np.asarray(dynamics, np.float64),
            np.asarray(aerodynamics, np.float64),
            time=float(timestamp),
            defaults=self._vehicle_defaults,
        )
        if vehicle.fingerprint == self.vehicle.fingerprint:
            self._runtime_vehicle_key = key
            self._runtime_vehicle_value = self.vehicle
            self._runtime_vehicle_defaults = self._vehicle_defaults
            return
        self.vehicle = vehicle
        self.model = FlightmareCTBRModel(vehicle, implementation=self.prediction_backend)
        self.model.native_full_step = self.native_model_step
        if self._backend is not None:
            self._backend.set_vehicle(vehicle)
        self._runtime_vehicle_key = key
        self._runtime_vehicle_value = self.vehicle
        self._runtime_vehicle_defaults = self._vehicle_defaults

    @property
    def source(self) -> str:
        return "starscream-acados-mpcc-v1" if self._backend is not None else "starscream-predictive-mpcc-v1"

    def reset(self) -> None:
        self._last_progress = None
        self._last_gate_index = None
        self._last_position = None
        self._missed_gate_recovery = False
        self._last_solver_success = False
        self._previous_action = np.asarray([self.vehicle.gravity, 0.0, 0.0, 0.0])
        if self._backend is not None:
            self._backend.reset()

    def set_nominal_speed(self, speed: float) -> None:
        """Retarget the spatial speed profile without rebuilding ACADOS code."""

        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("nominal speed must be finite and positive")
        self.config = replace(
            self.config,
            nominal_speed=float(speed),
            track_speed_overrides=(),
        )
        self._speed_progress, self._speed_profile = self._build_speed_profile()
        self.reset()

    def observe_executed_action(self, action: NDArray[np.floating]) -> None:
        """Synchronize delay/slew state when MPCC labels another policy's rollout.

        Interactive imitation queries MPCC at learner-visited states without
        necessarily executing the expert command. The next query must use the
        action that actually entered the plant, not MPCC's recommendation.
        """

        value = np.asarray(action, np.float64)
        if value.shape != (4,) or not np.all(np.isfinite(value)):
            raise ValueError("executed CTBR action must be finite with shape (4,)")
        self._previous_action = value.copy()

    def _active_gate_index(self, observation: dict[str, Any], position: F64) -> int:
        progress = observation.get("privileged", {}).get("progress")
        if progress is not None and np.asarray(progress).shape[0] >= 4:
            return int(round(float(np.asarray(progress)[3]))) % len(self.track.gates)
        gates = observation.get("gates", {})
        indices = gates.get("index")
        if indices is not None and len(indices):
            return int(indices[0]) % len(self.track.gates)
        return self.track.closest_gate_index(position)

    def _project(self, position: F64, gate_index: int) -> Projection:
        self._missed_gate_recovery = False
        if self._last_progress is not None and self._last_gate_index is not None:
            gate_delta = (gate_index - self._last_gate_index) % len(self.track.gates)
            if gate_delta in {0, 1}:
                hint = self._last_progress
                projection = self.line.project(position, hint_progress=hint, search_radius=10.0)
            else:
                hint = float(self.line.gate_progress[gate_index] - 2.5)
                projection = self.line.project(position, hint_progress=hint, search_radius=8.0)
        else:
            hint = float(self.line.gate_progress[gate_index] - 2.5)
            projection = self.line.project(position, hint_progress=hint, search_radius=8.0)

        # The gate tracker is authoritative.  If geometric projection has
        # advanced beyond the still-active gate, the vehicle crossed its plane
        # outside the aperture.  Continuing around the spline would leave that
        # gate permanently unclaimed and cause a lap timeout.  Anchor recovery
        # on the approach side so MPCC explicitly rejoins through the missed
        # aperture.
        if self.track.loop:
            expected_segment = (gate_index - 1) % len(self.track.gates)
            active_gate = self.track.gates[gate_index]
            gate_local = active_gate.directed_rotation.T @ (
                position - active_gate.position.astype(np.float64)
            )
            outside_aperture = bool(
                abs(gate_local[1]) > 0.5 * active_gate.size[0]
                or abs(gate_local[2]) > 0.5 * active_gate.size[1]
            )
            # Exit-side position alone is not a missed crossing: an intended
            # incoming segment can circle around a gate from its exit side.
            # Require the ordered path projection to have left that incoming
            # segment, as well as the geometric missed-aperture condition.
            missed_gate = (
                gate_local[0] > 0.05 and outside_aperture
                and projection.gate_index != expected_segment
            )
            if missed_gate:
                self._missed_gate_recovery = True
                previous_progress = float(self.line.gate_progress[expected_segment])
                gate_progress = float(self.line.gate_progress[gate_index])
                gap = (gate_progress - previous_progress) % self.line.length
                approach_distance = float(np.clip(0.25 * gap, 0.75, 2.0))
                target_progress = float(self.line.wrap(gate_progress - approach_distance))
                frame = self.line.evaluate(target_progress)
                center = np.asarray(frame["position"], dtype=np.float64)
                tangent = np.asarray(frame["tangent"], dtype=np.float64)
                residual = position - center
                lag = float(residual @ tangent)
                contour = residual - lag * tangent
                projection = Projection(
                    progress=target_progress,
                    position=center,
                    tangent=tangent,
                    lateral=np.asarray(frame["lateral"], dtype=np.float64),
                    up=np.asarray(frame["up"], dtype=np.float64),
                    contour_error=contour,
                    lag_error=lag,
                    distance=float(
                        np.clip(
                            np.linalg.norm(residual),
                            self.config.recovery_distance + 0.05,
                            self.config.hard_recovery_distance - 0.05,
                        )
                    ),
                    gate_index=expected_segment,
                )
        # The spline is periodic, but the progress state inside acados is not.
        # Keep that state on a continuous (unwrapped) coordinate so crossing
        # the final-to-first gate boundary does not inject a full-lap jump into
        # the SQP state and its shifted warm start.  RacingLine.evaluate/project
        # already wrap queries internally, so only the scalar state changes.
        if self.track.loop and self._last_progress is not None:
            wrapped = float(self.line.wrap(projection.progress))
            previous_wrapped = float(self.line.wrap(self._last_progress))
            delta = float(
                (wrapped - previous_wrapped + 0.5 * self.line.length)
                % self.line.length
                - 0.5 * self.line.length
            )
            projection = replace(
                projection,
                progress=float(self._last_progress + delta),
            )
        return projection

    def _warm_start_valid(self, position: F64, progress: float, gate_index: int) -> bool:
        if (
            not self._last_solver_success
            or self._last_progress is None
            or self._last_position is None
            or self._last_gate_index is None
        ):
            return False
        position_delta = float(np.linalg.norm(position - self._last_position))
        progress_delta = abs(float((progress - self._last_progress + 0.5 * self.line.length) % self.line.length - 0.5 * self.line.length))
        gate_delta = (gate_index - self._last_gate_index) % len(self.track.gates)
        return bool(
            position_delta <= self.config.warm_start_position_tolerance
            and progress_delta <= self.config.warm_start_progress_tolerance
            and gate_delta in {0, 1}
        )

    def _build_speed_profile(self) -> tuple[F64, F64]:
        """Build a cyclic speed limit with forward acceleration and braking passes."""

        progress = np.asarray(self.line.progress[:-1], dtype=np.float64)
        curvature = np.asarray(self.line.evaluate(progress)["curvature"], dtype=np.float64)
        nominal = min(
            self.config.nominal_speed_for_track(self.track.name),
            self.config.max_progress_speed,
        )
        speed = np.minimum(
            nominal,
            np.sqrt(
                self.config.maximum_acceleration
                / np.maximum(curvature, 0.025)
            ),
        )
        if len(progress) < 2:
            return progress, speed
        segment = np.diff(np.concatenate([progress, [self.line.length]]))
        if not self.track.loop:
            speed[-1] = min(speed[-1], self.config.terminal_speed)
        passes = max(4, len(progress) // 8)
        for _ in range(passes):
            changed = False
            last = len(speed) if self.track.loop else len(speed) - 1
            for index in range(last - 1, -1, -1):
                following = (index + 1) % len(speed)
                allowed = np.sqrt(
                    speed[following] ** 2
                    + 2.0 * self.config.maximum_braking_acceleration * segment[index]
                )
                if speed[index] > allowed:
                    speed[index] = allowed
                    changed = True
            for index in range(last):
                following = (index + 1) % len(speed)
                allowed = np.sqrt(
                    speed[index] ** 2
                    + 2.0 * self.config.maximum_longitudinal_acceleration * segment[index]
                )
                if speed[following] > allowed:
                    speed[following] = allowed
                    changed = True
            if not changed:
                break
        return progress, speed

    def _target_speed(self, progress: float) -> float:
        query = float(self.line.wrap(progress))
        if self.track.loop:
            old = getattr(self, "_periodic_speed_cache_key", None)
            if old is None or old[0] is not self._speed_progress or old[1] is not self._speed_profile or old[2] != self.line.length:
                self._periodic_speed_samples = np.concatenate([self._speed_progress, [self.line.length]])
                self._periodic_speed_values = np.concatenate([self._speed_profile, self._speed_profile[:1]])
                self._periodic_speed_cache_key = (self._speed_progress, self._speed_profile, self.line.length)
            samples, values = self._periodic_speed_samples, self._periodic_speed_values
        else:
            samples, values = self._speed_progress, self._speed_profile
        return float(np.interp(query, samples, values))

    def _mode(self, projection: Projection, position: F64) -> MPCCMode:
        bounds_margin = float(
            np.min(np.concatenate([position - self.track.bounds[:, 0], self.track.bounds[:, 1] - position]))
        )
        if projection.distance > self.config.hard_recovery_distance or bounds_margin < -0.05:
            return MPCCMode.INFEASIBLE
        if projection.distance > self.config.recovery_distance:
            return MPCCMode.RECOVERY
        return MPCCMode.NOMINAL

    def _reference(
        self,
        projection: Projection,
        mode: MPCCMode,
        current_velocity: F64,
    ) -> dict[str, F64]:
        count = self.config.horizon + 1
        progress = np.zeros(count, dtype=np.float64)
        progress_speed = np.zeros(count, dtype=np.float64)
        progress[0] = projection.progress
        progress_speed[0] = float(
            np.clip(current_velocity @ projection.tangent, 0.0, self.config.max_progress_speed)
        )
        speed_scale = 1.0
        if self._missed_gate_recovery:
            speed_scale = 0.0
        elif mode == MPCCMode.RECOVERY:
            speed_scale = float(np.clip(1.0 - projection.distance / self.config.hard_recovery_distance, 0.18, 0.65))
        elif mode == MPCCMode.INFEASIBLE:
            speed_scale = 0.0
        if self.prediction_backend == "native_exact":
            # Populate the cached periodic arrays without changing interp math.
            self._target_speed(progress[0])
            xp, fp = ((self._periodic_speed_samples, self._periodic_speed_values) if self.track.loop else (self._speed_progress, self._speed_profile))
            self.model.native_prediction.reference_progress(progress, progress_speed, self.config.time_steps, xp, fp,
                [self.line.length, self.track.loop, self.config.max_progress_speed,
                 self.config.time_optimal_progress_fraction, self.config.progress_reference_mode == "time_optimal", speed_scale])
        for index, dt in enumerate(() if self.prediction_backend == "native_exact" else self.config.time_steps):
            if self.config.progress_reference_mode == "time_optimal":
                target_speed = (
                    self.config.max_progress_speed
                    * self.config.time_optimal_progress_fraction
                    * speed_scale
                )
                # This is a path linearization schedule, not the progress-rate
                # target. Keep its frames plant-feasible while the coupled
                # theta state is optimized toward the faster target above.
                linearization_speed = (
                    self._target_speed(progress[index]) * speed_scale
                )
            else:
                target_speed = self._target_speed(progress[index]) * speed_scale
                linearization_speed = target_speed
            progress_speed[index + 1] = target_speed
            progress[index + 1] = progress[index] + linearization_speed * dt
        if (
            self.config.terminal_speed_envelope
            and mode == MPCCMode.NOMINAL
            and count > 1
        ):
            # MPCC++ uses a feasible periodic terminal set.  This first-party
            # approximation retains the free-progress stage objective while
            # asking the horizon endpoint to re-enter the plant-feasible
            # curvature/braking envelope.  It prevents the short horizon from
            # blindly accelerating into a turn just beyond its field of view.
            progress_speed[-1] = min(
                progress_speed[-1], self._target_speed(progress[-1])
            )
        if count > 1:
            progress_speed[0] = progress_speed[1]
        frame = self.line.evaluate(progress)
        velocity = np.asarray(frame["tangent"]) * progress_speed[:, None]
        return {
            "progress": progress,
            "position": np.asarray(frame["position"]),
            "tangent": np.asarray(frame["tangent"]),
            "lateral": np.asarray(frame["lateral"]),
            "up": np.asarray(frame["up"]),
            "curvature": np.asarray(frame["curvature"]),
            "corridor_lateral": np.maximum(
                np.asarray(frame["corridor_lateral"]) - self.config.corridor_margin, 0.08
            ),
            "corridor_vertical": np.maximum(
                np.asarray(frame["corridor_vertical"]) - self.config.corridor_margin, 0.08
            ),
            "velocity": velocity,
            "progress_speed": progress_speed,
        }

    def _feedback_action(
        self,
        state: F64,
        reference_position: F64,
        reference_velocity: F64,
        tangent: F64,
        previous_action: F64,
        mode: MPCCMode,
    ) -> F64:
        position, velocity = state[0:3], state[7:10]
        kp = 5.0 if mode == MPCCMode.RECOVERY else 3.4
        kd = 3.4 if mode == MPCCMode.RECOVERY else 2.5
        desired_acceleration = kp * (reference_position - position) + kd * (reference_velocity - velocity)
        desired_force = desired_acceleration + np.asarray([0.0, 0.0, self.vehicle.gravity])
        desired_z = desired_force / max(float(np.linalg.norm(desired_force)), 1e-8)
        heading = tangent - desired_z * float(desired_z @ tangent)
        if np.linalg.norm(heading) < 1e-5:
            heading = np.asarray([1.0, 0.0, 0.0])
        desired_x = heading / max(float(np.linalg.norm(heading)), 1e-8)
        desired_y = np.cross(desired_z, desired_x)
        desired_y /= max(float(np.linalg.norm(desired_y)), 1e-8)
        desired_x = np.cross(desired_y, desired_z)
        desired_rotation = np.stack([desired_x, desired_y, desired_z], axis=1)
        rotation = quaternion_matrix(state[3:7]).astype(np.float64)
        attitude_error = _vee(desired_rotation.T @ rotation - rotation.T @ desired_rotation)
        rates = np.clip(-6.5 * attitude_error, -np.asarray(self.vehicle.body_rate_max), self.vehicle.body_rate_max)
        collective = float(np.clip(desired_force @ rotation[:, 2], self.config.minimum_collective_thrust, self.config.maximum_collective_thrust))
        raw = np.concatenate([[collective], rates])
        smoothing = 0.35 if mode == MPCCMode.NOMINAL else 0.15
        smoothed = smoothing * previous_action + (1.0 - smoothing) * raw
        return self._limit_action_slew(smoothed, previous_action)

    def _limit_action_slew(self, action: F64, previous_action: F64) -> F64:
        """Apply the final finite, actuator-aware CTBR command envelope."""

        slew = np.asarray(
            [
                self.config.collective_slew_limit,
                *([self.config.body_rate_slew_limit] * 3),
            ],
            dtype=np.float64,
        )
        limited = np.clip(action, previous_action - slew, previous_action + slew)
        limited[0] = np.clip(
            limited[0],
            self.config.minimum_collective_thrust,
            self.config.maximum_collective_thrust,
        )
        limited[1:4] = np.clip(
            limited[1:4],
            -np.asarray(self.vehicle.body_rate_max),
            np.asarray(self.vehicle.body_rate_max),
        )
        return limited.astype(np.float64)

    def _predictive_solve(
        self,
        state: F64,
        motors: F64,
        reference: dict[str, F64],
        mode: MPCCMode,
    ) -> tuple[F64, F64, F64]:
        current = ModelState(state, motors)
        states = [state.copy()]
        motor_horizon = [motors.copy()]
        actions: list[F64] = []
        previous = self._previous_action.copy()
        for index, dt in enumerate(self.config.time_steps):
            action = self._feedback_action(
                current.state,
                reference["position"][index + 1],
                reference["velocity"][index + 1],
                reference["tangent"][index + 1],
                previous,
                mode,
            )
            current = self.model.step(current, action, dt)
            actions.append(action)
            states.append(current.state.copy())
            motor_horizon.append(current.motor_omega.copy())
            previous = action
        return np.stack(states), np.stack(motor_horizon), np.stack(actions)

    def _unpack_acados(self, result: AcadosResult) -> tuple[F64, F64, F64]:
        states = np.zeros((len(result.states), 25), dtype=np.float64)
        states[:, 0:3] = result.states[:, 0:3]
        states[:, 3:7] = result.states[:, 3:7]
        states[:, 7:10] = result.states[:, 7:10]
        states[:, 10:13] = result.states[:, 10:13]
        motors = MOTOR_OMEGA_SCALE * result.states[:, 13:17]
        actions = result.controls[:, 0:4]
        return states, motors, actions

    def _diagnostics(
        self,
        predicted_states: F64,
        predicted_motors: F64,
        predicted_actions: F64,
        reference: dict[str, F64],
        mode: MPCCMode,
        warm_start: bool,
        projection: Projection,
        result: AcadosResult | None,
    ) -> dict[str, np.ndarray]:
        errors = predicted_states[:, 0:3] - reference["position"]
        lag = np.sum(errors * reference["tangent"], axis=1)
        contour = errors - lag[:, None] * reference["tangent"]
        contour_norm = np.linalg.norm(contour, axis=1)
        action_delta = np.diff(np.vstack([self._previous_action, predicted_actions]), axis=0)
        world_margin = np.min(
            np.concatenate(
                [
                    predicted_states[:, 0:3] - self.track.bounds[:, 0],
                    self.track.bounds[:, 1] - predicted_states[:, 0:3],
                ],
                axis=1,
            ),
            axis=1,
        )
        lateral_error = np.sum(errors * reference["lateral"], axis=1)
        vertical_error = np.sum(errors * reference["up"], axis=1)
        corridor_margin = np.minimum(
            reference["corridor_lateral"] - np.abs(lateral_error),
            reference["corridor_vertical"] - np.abs(vertical_error),
        )
        costs = np.asarray(
            [
                np.mean(contour_norm**2),
                np.mean(lag**2),
                -float(reference["progress"][-1] - reference["progress"][0]),
                np.mean((predicted_states[:, 7:10] - reference["velocity"]) ** 2),
                np.mean(predicted_states[:, 10:13] ** 2),
                np.mean(predicted_actions**2),
                np.mean(action_delta**2),
                max(0.0, -float(np.min(corridor_margin))) ** 2,
            ],
            dtype=np.float32,
        )
        residuals = np.zeros(8, dtype=np.float32)
        residuals[0:4] = np.asarray(result.residuals[:4] if result is not None else 0.0)
        residuals[4:] = [
            float(np.max(contour_norm)),
            float(np.max(np.abs(lag))),
            float(np.min(corridor_margin)),
            float(np.min(world_margin)),
        ]
        return {
            "predicted_state_horizon": predicted_states.astype(np.float32),
            "predicted_action_horizon": predicted_actions.astype(np.float32),
            "predicted_progress_horizon": reference["progress"].astype(np.float32),
            "predicted_motor_omega": predicted_motors.astype(np.float32),
            "contour_lag_error": np.stack([contour_norm, lag], axis=1).astype(np.float32),
            "cost_components": costs,
            "constraint_residuals": residuals,
            "solver_iterations": np.asarray(result.iterations if result is not None else 1, np.int32),
            "warm_start_valid": np.asarray(warm_start),
            "mode": np.asarray(int(mode), np.int8),
            "projection_distance": np.asarray(projection.distance, np.float32),
            "racing_line_hash_u64": np.asarray(int(self.line.fingerprint[:16], 16), np.uint64),
            "controller_hash_u64": np.asarray(int(self.config.fingerprint[:16], 16), np.uint64),
        }

    def _minimum_constraint_margin(
        self,
        predicted_states: F64,
        reference: dict[str, F64],
    ) -> float:
        """Return the command qualification margin without horizon diagnostics."""

        errors = predicted_states[:, 0:3] - reference["position"]
        lateral_error = np.sum(errors * reference["lateral"], axis=1)
        vertical_error = np.sum(errors * reference["up"], axis=1)
        corridor_margin = np.minimum(
            reference["corridor_lateral"] - np.abs(lateral_error),
            reference["corridor_vertical"] - np.abs(vertical_error),
        )
        world_margin = np.min(
            np.concatenate(
                [
                    predicted_states[:, 0:3] - self.track.bounds[:, 0],
                    self.track.bounds[:, 1] - predicted_states[:, 0:3],
                ],
                axis=1,
            ),
            axis=1,
        )
        return float(min(np.min(corridor_margin), np.min(world_margin)))

    def __call__(self, observation: dict[str, Any]) -> ControllerCommand:
        started = time.perf_counter()
        privileged = observation.get("privileged", {})
        self._synchronize_runtime_vehicle(observation, privileged)
        synchronized = time.perf_counter()
        state_value = privileged.get("state", observation.get("state"))
        if state_value is None:
            raise ValueError("MPCC requires a privileged 25-value simulator or estimator state")
        state = np.asarray(state_value, dtype=np.float64)
        if state.shape != (25,) or not np.all(np.isfinite(state)):
            raise ValueError("MPCC state must be finite with shape (25,)")
        motors = np.asarray(
            observation.get("measured", {}).get("motor_omega", observation.get("motor_omega", np.zeros(4))),
            dtype=np.float64,
        )
        if self.config.actuation_delay > 0.0:
            delay_compensated = self.model.step(
                ModelState(state, motors),
                self._previous_action,
                self.config.actuation_delay,
            )
            state = delay_compensated.state
            motors = delay_compensated.motor_omega
        position = state[0:3]
        gate_index = self._active_gate_index(observation, position)
        projection = self._project(position, gate_index)
        mode = self._mode(projection, position)
        warm_start = self._warm_start_valid(position, projection.progress, gate_index)
        if not warm_start and self._backend is not None:
            self._backend.reset()
        reference = self._reference(projection, mode, state[7:10])
        reference_ready = time.perf_counter()
        result: AcadosResult | None = None
        if self._backend is not None and mode != MPCCMode.INFEASIBLE:
            initial = self._backend.pack_state(state, motors, projection.progress)
            guess_states = guess_controls = None
            if not warm_start:
                seed_states, seed_motors, seed_actions = self._predictive_solve(
                    state, motors, reference, mode
                )
                guess_states = np.stack(
                    [
                        self._backend.pack_state(
                            seed_states[index], seed_motors[index], reference["progress"][index]
                        )
                        for index in range(self.config.horizon + 1)
                    ]
                )
                progress_rates = np.diff(reference["progress"]) / np.asarray(self.config.time_steps)
                guess_controls = np.column_stack([seed_actions, progress_rates])
            result = self._backend.solve(
                initial,
                reference,
                self._previous_action,
                warm_start=warm_start,
                initial_guess_states=guess_states,
                initial_guess_controls=guess_controls,
            )
        if result is not None and result.status == 0 and np.all(np.isfinite(result.controls)):
            predicted_states, predicted_motors, predicted_actions = self._unpack_acados(result)
            solver_status = 0
            self._last_solver_success = True
        else:
            if self._backend is not None:
                self._backend.reset()
            predicted_states, predicted_motors, predicted_actions = self._predictive_solve(
                state, motors, reference, mode
            )
            solver_status = -100 if result is None else -max(1, result.status)
            self._last_solver_success = self._backend is None
        backend_ready = time.perf_counter()
        if mode == MPCCMode.INFEASIBLE:
            # A finite braking/leveling action is returned for the simulator,
            # while valid=False explicitly excludes it from accepted expert data.
            action_array = self._feedback_action(
                state,
                position,
                np.zeros(3),
                projection.tangent,
                self._previous_action,
                mode,
            )
        else:
            action_array = self._limit_action_slew(
                predicted_actions[0], self._previous_action
            )
        action = CTBRAction.from_array(action_array.astype(np.float32))
        solve_time = float(result.solve_time if result is not None else time.perf_counter() - started)
        reference_state = state.copy()
        reference_state[0:3] = reference["position"][1]
        reference_state[7:10] = reference["velocity"][1]
        reference_state[10:13] = action.body_rates
        tangent = reference["tangent"][1]
        reference_state[3:7] = matrix_quaternion(
            np.stack(
                [
                    tangent / max(np.linalg.norm(tangent), 1e-8),
                    reference["lateral"][1],
                    reference["up"][1],
                ],
                axis=1,
            )
        )
        if self.diagnostics_level in {"minimal", "targets"}:
            margin = self._minimum_constraint_margin(predicted_states, reference)
            diagnostics = {
                "solver_iterations": np.asarray(
                    result.iterations if result is not None else 1, np.int32
                ),
                "warm_start_valid": np.asarray(warm_start),
                "mode": np.asarray(int(mode), np.int8),
                "projection_distance": np.asarray(projection.distance, np.float32),
            }
            if self.diagnostics_level == "targets":
                # DAgger's trajectory/dynamics objectives need only these three
                # horizons. Avoid the contour, cost, motor, and constraint
                # arrays computed by full experiment diagnostics.
                diagnostics.update({
                    "predicted_state_horizon": predicted_states.astype(np.float32),
                    "predicted_action_horizon": predicted_actions.astype(np.float32),
                    "predicted_progress_horizon": reference["progress"].astype(np.float32),
                })
        else:
            diagnostics = self._diagnostics(
                predicted_states,
                predicted_motors,
                predicted_actions,
                reference,
                mode,
                warm_start,
                projection,
                result,
            )
            margin = float(min(
                diagnostics["constraint_residuals"][6],
                diagnostics["constraint_residuals"][7],
            ))
        # projection.distance is deliberately clipped for recovery mode selection.
        # Preserve that controller behavior; expose the actual reference error
        # at the same delay-compensated state for trajectory admission.
        diagnostics["trajectory_reference_distance"] = np.asarray(np.linalg.norm(position - projection.position), np.float32)
        diagnostics["missed_gate_recovery"] = np.asarray(self._missed_gate_recovery)
        diagnostics["controller_wall_breakdown"] = np.asarray(
            [
                synchronized - started,
                reference_ready - synchronized,
                backend_ready - reference_ready,
                time.perf_counter() - backend_ready,
            ],
            np.float32,
        )
        diagnostics["backend_wall_breakdown"] = np.asarray(
            result.wall_breakdown if result is not None else np.zeros(4),
            np.float32,
        )
        timestamp = float(observation.get("timestamp", {}).get("sim", observation.get("time", 0.0)))
        self._last_progress = projection.progress
        self._last_gate_index = gate_index
        self._last_position = position.copy()
        self._previous_action = action.as_array().astype(np.float64)
        return ControllerCommand(
            action=action,
            source_timestamp=timestamp,
            receive_timestamp=timestamp,
            reference_state=reference_state.astype(np.float32),
            reference_action=action.as_array(),
            reference_progress=float(projection.progress),
            solver_status=solver_status,
            solve_time=solve_time,
            constraint_margin=margin,
            valid=mode != MPCCMode.INFEASIBLE and np.all(np.isfinite(action.as_array())),
            source=self.source,
            diagnostics=diagnostics,
        )
