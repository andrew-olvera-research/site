"""Gymnasium wrapper around Starscream's extended Flightmare binding."""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from PIL import Image, ImageChops, ImageDraw
import yaml

from .tracks import GateTracker, Track, load_track, matrix_quaternion, quaternion_matrix
from .types import CTBRAction, Proprioception
from .estimation import (
    RandomizedStateEstimator, SimulatorStateEstimator, StateEstimator,
    StateEstimatorRandomizationConfig,
)
from .dynamics_randomization import (
    AERODYNAMIC_PARAMETER_NAMES, DynamicsDomain, DynamicsRandomizationConfig,
    sample_dynamics_domain,
)
from starscream.rewards import PerceptionAwareRaceReward


class FlightmareUnavailable(RuntimeError):
    """Raised when the native extension or renderer is unavailable."""


class FlightmareEnv(gym.Env):
    """Single-vehicle CTBR Flightmare environment with rich data observations.

    This bypasses the limited upstream ``QuadrotorEnv_v1`` API. Actions are
    ``[collective_thrust_mps2, body_rate_x, body_rate_y, body_rate_z]``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        track: Track | str = "figure8",
        *,
        config_path: str | Path | None = None,
        next_gates: int = 3,
        control_dt: float = 0.02,
        image_size: tuple[int, int] = (640, 480),
        fov_degrees: float = 90.0,
        render_observations: bool = False,
        mask_source: str = "geometry",
        geometry_renderer: str = "exact",
        mask_size: tuple[int, int] = (160, 128),
        retain_render_images: bool = False,
        image_delay: float = 0.033,
        action_delay: float = 0.011,
        scene_id: int = 1,
        state_estimator: StateEstimator | None = None,
        state_estimator_randomization: Mapping[str, Any] | None = None,
        policy_state_source: str = "truth",
        flight_plan_randomization: Mapping[str, Any] | None = None,
        reward_function: PerceptionAwareRaceReward | None = None,
        terminate_on_collision: bool = True,
        camera_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        camera_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        dynamics_randomization: DynamicsRandomizationConfig | Mapping[str, Any] | None = None,
        action_delay_range: tuple[float, float] | None = None,
        maximum_collective_thrust: float = 30.0,
        collection_observation: bool = False,
        plant_settings_observation: bool = False,
    ) -> None:
        super().__init__()
        try:
            import flightgym  # type: ignore[import-not-found]
        except ImportError as error:
            raise FlightmareUnavailable(
                "flightgym is not compiled; build and enter the Docker container"
            ) from error
        if not hasattr(flightgym, "RichQuadrotorEnv_v0"):
            raise FlightmareUnavailable(
                "flightgym lacks RichQuadrotorEnv_v0; use this project's Docker image"
            )
        self.track = load_track(track) if isinstance(track, (str, Path)) else track
        self.tracker = GateTracker(self.track)
        # Optional finite-horizon route contract. When set at reset, route
        # lookahead rolls across intermediate lap boundaries and repeats the
        # final gate after the requested episode route is exhausted.
        self._route_plan_total_gates: int | None = None
        gate_centres = np.stack([gate.position for gate in self.track.gates]).astype(np.float64)
        gate_segments = np.linalg.norm(np.diff(gate_centres, axis=0), axis=1)
        self._gate_cumulative_distance = np.concatenate([[0.0], np.cumsum(gate_segments)])
        self._course_length = float(self._gate_cumulative_distance[-1])
        if self.track.loop and len(gate_centres) > 1:
            self._course_length += float(np.linalg.norm(gate_centres[0] - gate_centres[-1]))
        self.next_gate_count = int(next_gates)
        self.control_dt = float(control_dt)
        self.plant_settings_observation = bool(plant_settings_observation)
        self._plant_settings_static = None
        self.render_observations = bool(render_observations)
        self.collection_observation = bool(collection_observation)
        if self.collection_observation and (render_observations or mask_source != 'none'):
            raise ValueError('compact collection observations require non-rendering mask_source=none')
        if mask_source not in {"geometry", "unity", "none"}:
            raise ValueError("mask_source must be 'geometry', 'unity', or 'none'")
        if geometry_renderer not in {"exact", "fast_polygon", "native_exact"}:
            raise ValueError(
                "geometry_renderer must be 'exact', 'fast_polygon', or 'native_exact'"
            )
        self.mask_source = mask_source
        self.geometry_renderer = geometry_renderer
        self.mask_size = tuple(int(value) for value in mask_size)
        self.retain_render_images = bool(retain_render_images)
        self.image_delay = float(image_delay)
        self.action_delay = float(action_delay)
        self._nominal_action_delay = float(action_delay)
        self.action_delay_range = (
            None if action_delay_range is None
            else tuple(float(value) for value in action_delay_range)
        )
        self.maximum_collective_thrust = float(maximum_collective_thrust)
        if not np.isfinite(self.maximum_collective_thrust) or self.maximum_collective_thrust <= 0.0:
            raise ValueError("maximum_collective_thrust must be finite and positive")
        if min(self.mask_size) < 1 or self.image_delay < 0 or self.action_delay < 0:
            raise ValueError("mask dimensions and sensor/action delays must be non-negative")
        if self.mask_source == "unity" and not self.render_observations:
            raise ValueError("Unity masks require render_observations=True")
        if state_estimator is not None and state_estimator_randomization is not None:
            raise ValueError("provide a state estimator or its randomization config, not both")
        estimator_config = StateEstimatorRandomizationConfig.from_mapping(
            state_estimator_randomization
        )
        self.state_estimator = (
            state_estimator
            or (
                RandomizedStateEstimator(estimator_config, control_dt=self.control_dt)
                if estimator_config is not None
                else SimulatorStateEstimator()
            )
        )
        self.policy_state_source = str(policy_state_source)
        if self.policy_state_source not in {"truth", "estimate"}:
            raise ValueError("policy_state_source must be 'truth' or 'estimate'")
        self.flight_plan_randomization = dict(flight_plan_randomization or {})
        self._policy_gate_position_error = np.zeros(
            (len(self.track.gates), 3), np.float32
        )
        self._policy_gate_orientation_error = np.zeros(
            (len(self.track.gates), 3), np.float32
        )
        self._policy_gate_size_scale = np.ones(len(self.track.gates), np.float32)
        if self.action_delay_range is not None:
            if (
                len(self.action_delay_range) != 2
                or self.action_delay_range[0] < 0.0
                or self.action_delay_range[1] < self.action_delay_range[0]
            ):
                raise ValueError("action_delay_range must be an ordered non-negative pair")
        self.reward_function = reward_function or PerceptionAwareRaceReward()
        self.terminate_on_collision = bool(terminate_on_collision)
        self.camera_extrinsics = np.concatenate(
            [
                np.asarray(camera_translation, np.float32),
                np.asarray(camera_quaternion_wxyz, np.float32),
            ]
        )
        if self.camera_extrinsics.shape != (7,) or not np.all(np.isfinite(self.camera_extrinsics)):
            raise ValueError("camera extrinsics must be finite translation(3)+quaternion_wxyz(4)")
        self.state_parameter_names = (
            *(f"world_position_{axis}" for axis in "xyz"),
            "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z",
            *(f"world_velocity_{axis}" for axis in "xyz"),
            *(f"body_rate_{axis}" for axis in "xyz"),
            *(f"world_acceleration_{axis}" for axis in "xyz"),
            *(f"body_torque_{axis}" for axis in "xyz"),
            *(f"gyro_bias_{axis}" for axis in "xyz"),
            *(f"accelerometer_bias_{axis}" for axis in "xyz"),
        )
        self.sky_state_parameter_names = (
            *(f"world_position_{axis}" for axis in "xyz"),
            *(f"gate_position_{axis}" for axis in "xyz"),
            *(f"world_velocity_{axis}" for axis in "xyz"),
            *(f"gate_velocity_{axis}" for axis in "xyz"),
            "world_roll", "world_pitch", "world_yaw", "gate_yaw",
            *(f"true_body_rate_{axis}" for axis in "xyz"),
            *(f"true_motor_omega_{index}" for index in range(4)),
        )
        self._connected = False
        if self.next_gate_count < 1 or self.control_dt <= 0:
            raise ValueError("next_gates and control_dt must be positive")
        if config_path is None:
            root = os.environ.get("FLIGHTMARE_PATH")
            if not root:
                raise FlightmareUnavailable("FLIGHTMARE_PATH is not configured")
            config_path = Path(root) / "flightlib" / "configs" / "quadrotor_env.yaml"
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as stream:
            config_document = yaml.safe_load(stream) or {}
        dynamics = config_document.get("quadrotor_dynamics", {})
        mass = float(dynamics.get("mass", 1.0))
        arm_length = float(dynamics.get("arm_l", 0.2))
        inertia = mass / 12.0 * arm_length * arm_length * np.asarray([4.5, 4.5, 7.0])
        self.dynamics_parameter_names = (
            "mass", "arm_length", "inertia_xx", "inertia_yy", "inertia_zz",
            "motor_omega_min", "motor_omega_max", "motor_tau",
            "thrust_map_quadratic", "thrust_map_linear", "thrust_map_constant",
            "kappa", "body_rate_max_x", "body_rate_max_y", "body_rate_max_z",
        )
        self.dynamics_parameters = np.asarray(
            [
                mass, arm_length, *inertia,
                float(dynamics.get("motor_omega_min", 150.0)),
                float(dynamics.get("motor_omega_max", 2000.0)),
                float(dynamics.get("motor_tau", 0.05)),
                *dynamics.get("thrust_map", [1.32982535e-6, 0.00383608, -1.76899868]),
                float(dynamics.get("kappa", 0.016)),
                *dynamics.get("omega_max", [6.0, 6.0, 6.0]),
            ],
            dtype=np.float32,
        )
        self._nominal_dynamics_parameters = self.dynamics_parameters.copy()
        self.dynamics_randomization = (
            dynamics_randomization
            if isinstance(dynamics_randomization, DynamicsRandomizationConfig)
            else DynamicsRandomizationConfig.from_mapping(dynamics_randomization)
        )
        self.aerodynamics_parameter_names = AERODYNAMIC_PARAMETER_NAMES
        self.aerodynamics_parameters = np.zeros(
            len(self.aerodynamics_parameter_names), np.float32
        )
        self._dynamics_domain: DynamicsDomain | None = None
        rotor_xy = arm_length / np.sqrt(2.0) * np.asarray(
            [[1, -1], [-1, -1], [-1, 1], [1, 1]], dtype=np.float32
        )
        rotor_positions = np.pad(rotor_xy, ((0, 0), (0, 1))).reshape(-1)
        omega_min, omega_max, motor_tau = self.dynamics_parameters[5:8]
        thrust_map = self.dynamics_parameters[8:11]
        thrust_limits = thrust_map[0] * np.asarray([omega_min, omega_max]) ** 2
        thrust_limits += thrust_map[1] * np.asarray([omega_min, omega_max]) + thrust_map[2]
        self.actuator_parameter_names = (
            "mass", "arm_length", "inertia_xx", "inertia_yy", "inertia_zz",
            *(f"rotor_{rotor}_{axis}" for rotor in range(4) for axis in ("x", "y", "z")),
            *(f"rotor_{rotor}_spin" for rotor in range(4)),
            "motor_omega_min", "motor_omega_max", "motor_tau",
            "thrust_map_quadratic", "thrust_map_linear", "thrust_map_constant",
            "kappa", "motor_thrust_min", "motor_thrust_max",
        )
        self.actuator_parameters = np.asarray(
            [
                mass, arm_length, *inertia, *rotor_positions, 1.0, -1.0, 1.0, -1.0,
                omega_min, omega_max, motor_tau, *thrust_map,
                self.dynamics_parameters[11], *thrust_limits,
            ],
            dtype=np.float32,
        )
        width, height = image_size
        self.image_size = (int(width), int(height))
        self.fov_degrees = float(fov_degrees)
        # Flightmare forwards the ROS camera transform directly to a Unity
        # Camera, whose optical forward/up axes are local +Z/+Y.  Starscream's
        # camera contract is the robotics convention +X forward, +Z up.  Adapt
        # at the native boundary so the persisted semantic extrinsic can remain
        # identity for a body-aligned camera.
        native_from_optical = np.asarray(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        native_camera_quaternion = matrix_quaternion(
            quaternion_matrix(self.camera_extrinsics[3:7]) @ native_from_optical
        )
        self._native = flightgym.RichQuadrotorEnv_v0(
            str(config_path), width, height, fov_degrees,
            self.retain_render_images or self.mask_source == "unity",
            self.retain_render_images,
            self.retain_render_images,
            self.camera_extrinsics[:3], native_camera_quaternion,
        )
        self._flightgym = flightgym
        self._gate_positions = np.ascontiguousarray(
            np.stack([gate.position for gate in self.track.gates]), dtype=np.float32
        )
        self._gate_laterals = np.ascontiguousarray(
            np.stack([gate.lateral for gate in self.track.gates]), dtype=np.float32
        )
        self._gate_ups = np.ascontiguousarray(
            np.stack([gate.up for gate in self.track.gates]), dtype=np.float32
        )
        self._gate_sizes = np.ascontiguousarray(
            np.stack([gate.size for gate in self.track.gates]), dtype=np.float32
        )
        native_dynamics = self._native.get_dynamics()
        self.actuator_parameters[-2:] = np.asarray(
            [native_dynamics["motor_thrust_min"], native_dynamics["motor_thrust_max"]],
            dtype=np.float32,
        )
        self._native.set_world_box(self.track.bounds.astype(np.float32))
        if self.dynamics_randomization is not None and (
            not hasattr(self._native, "set_dynamics")
            or not hasattr(self._native, "set_aerodynamics")
        ):
            raise FlightmareUnavailable(
                "full dynamics randomization requires the project-owned aerodynamic FlightGym binding"
            )
        # The Unity ``rpg_gate`` prefab lies in its local X/Y plane (normal Z),
        # whereas Starscream's semantic gate frame is X-forward, Y-lateral,
        # Z-up.  Flightmare also swaps ROS Y/Z scale components on the way to
        # Unity.  Keep planning coordinates semantic and adapt only the render
        # pose/scale here.
        prefab_from_semantic = np.asarray(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        for index, gate in enumerate(self.track.gates):
            if not gate.render:
                continue
            render_quaternion = matrix_quaternion(gate.rotation @ prefab_from_semantic)
            # The prefab's unit scale has an approximately 5 m square aperture.
            size = np.asarray(
                [gate.size[0] / 5.0, 0.2, gate.size[1] / 5.0], dtype=np.float32
            )
            # Mask-only collection uses Unity as an occlusion oracle and does
            # not instantiate the mismatched stock prefab.  Debug RGB retains
            # the prefab for visual inspection only.
            if self.mask_source != "unity" or self.retain_render_images:
                self._native.add_gate(
                    f"{self.track.name}_gate_{index:02d}",
                    gate.position,
                    render_quaternion,
                    size,
                )
        self._unity_gate_render_quaternions = tuple(
            matrix_quaternion(gate.rotation @ prefab_from_semantic)
            for gate in self.track.gates
        )
        if self.render_observations:
            self.connect_unity(scene_id)

        self.action_space = gym.spaces.Box(
            low=np.asarray([0.0, -6.0, -6.0, -6.0], dtype=np.float32),
            high=np.asarray(
                [self.maximum_collective_thrust, 6.0, 6.0, 6.0],
                dtype=np.float32,
            ),
        )
        gate_spaces = {
            "position": gym.spaces.Box(-np.inf, np.inf, (self.next_gate_count, 3), np.float32),
            "normal": gym.spaces.Box(-1.0, 1.0, (self.next_gate_count, 3), np.float32),
            "up": gym.spaces.Box(-1.0, 1.0, (self.next_gate_count, 3), np.float32),
            "size": gym.spaces.Box(0.0, np.inf, (self.next_gate_count, 2), np.float32),
            "distance": gym.spaces.Box(0.0, np.inf, (self.next_gate_count,), np.float32),
            "index": gym.spaces.Box(0, len(self.track.gates) - 1, (self.next_gate_count,), np.int64),
            "mask": gym.spaces.MultiBinary(self.next_gate_count),
            "enter_from_opposite_side": gym.spaces.MultiBinary(self.next_gate_count),
        }
        spaces: dict[str, gym.Space] = {
            "state": gym.spaces.Box(-np.inf, np.inf, (25,), np.float32),
            "motor_thrusts": gym.spaces.Box(0.0, np.inf, (4,), np.float32),
            "motor_omega": gym.spaces.Box(0.0, np.inf, (4,), np.float32),
            "state_estimate": gym.spaces.Box(-np.inf, np.inf, (25,), np.float32),
            "state_estimate_std": gym.spaces.Box(0.0, np.inf, (25,), np.float32),
            "state_estimate_valid": gym.spaces.Discrete(2),
            "time": gym.spaces.Box(0.0, np.inf, (), np.float32),
            "collision": gym.spaces.Discrete(2),
            "previous_action": self.action_space,
            "gates": gym.spaces.Dict(gate_spaces),
            "flight_plan": gym.spaces.Dict(
                {
                    "records": gym.spaces.Box(
                        -np.inf, np.inf, (self.next_gate_count, 13), np.float32
                    ),
                    "position": gate_spaces["position"],
                    "normal": gate_spaces["normal"],
                    "up": gate_spaces["up"],
                    "size": gate_spaces["size"],
                    "index": gate_spaces["index"],
                    "mask": gate_spaces["mask"],
                    "enter_from_opposite_side": gate_spaces["enter_from_opposite_side"],
                }
            ),
            "task_state": gym.spaces.Box(-np.inf, np.inf, (19,), np.float32),
            "measured": gym.spaces.Dict(
                {
                    "body_rates": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                    "motor_omega": gym.spaces.Box(0.0, np.inf, (4,), np.float32),
                }
            ),
            "timestamp": gym.spaces.Dict(
                {
                    name: gym.spaces.Box(-np.inf, np.inf, (), np.float64)
                    for name in ("sim", "camera", "body_rates", "motor_omega", "state_estimate", "previous_action")
                }
            ),
            "age": gym.spaces.Dict(
                {
                    name: gym.spaces.Box(0.0, np.inf, (), np.float32)
                    for name in ("camera", "body_rates", "motor_omega", "state_estimate", "previous_action")
                }
            ),
            "valid": gym.spaces.Dict(
                {
                    name: gym.spaces.Discrete(2)
                    for name in ("camera", "body_rates", "motor_omega", "state_estimate", "previous_action")
                }
            ),
            "privileged": gym.spaces.Dict(
                {
                    "state": gym.spaces.Box(-np.inf, np.inf, (25,), np.float32),
                    "gate_state": gym.spaces.Box(-np.inf, np.inf, (19,), np.float32),
                    "dynamics": gym.spaces.Box(-np.inf, np.inf, (15,), np.float32),
                    "aerodynamics": gym.spaces.Box(
                        -np.inf, np.inf, (len(AERODYNAMIC_PARAMETER_NAMES),), np.float32
                    ),
                    "camera_extrinsics": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
                    "camera_intrinsics": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
                    "sky_state": gym.spaces.Box(-np.inf, np.inf, (23,), np.float32),
                    "disturbance": gym.spaces.Box(-np.inf, np.inf, (10,), np.float32),
                    "progress": gym.spaces.Box(-np.inf, np.inf, (6,), np.float32),
                    "actuator": gym.spaces.Box(-np.inf, np.inf, (30,), np.float32),
                }
            ),
        }
        if self.mask_source != "none":
            mask_width, mask_height = self.mask_size
            spaces["gate_mask"] = gym.spaces.Box(0, 255, (mask_height, mask_width), np.uint8)
        if self.retain_render_images:
            spaces.update(
                rgb=gym.spaces.Box(0, 255, (height, width, 3), np.uint8),
                depth=gym.spaces.Box(0.0, np.inf, (height, width), np.float32),
                segmentation=gym.spaces.Box(0, 255, (height, width, 3), np.uint8),
                optical_flow=gym.spaces.Box(0, 255, (height, width, 3), np.uint8),
            )
        self.observation_space = gym.spaces.Dict(spaces)
        self._previous_action = np.zeros(4, dtype=np.float32)
        self._previous_action_valid = False
        self._last_observation: dict[str, Any] | None = None
        self._mask_queue: deque[tuple[float, np.ndarray]] = deque()
        self._action_queue: deque[tuple[float, np.ndarray]] = deque()
        self._applied_action = np.asarray([9.81, 0.0, 0.0, 0.0], np.float32)
        self._applied_action_timestamp = -self.action_delay

    def _apply_dynamics_domain(self, seed: int) -> None:
        assert self.dynamics_randomization is not None
        nominal = self._nominal_dynamics_parameters
        domain = sample_dynamics_domain(
            self.dynamics_randomization,
            seed=int(seed),
            nominal_mass=float(nominal[0]),
            nominal_arm_length=float(nominal[1]),
            nominal_motor_omega_min=float(nominal[5]),
            nominal_motor_omega_max=float(nominal[6]),
            nominal_motor_tau=float(nominal[7]),
            nominal_thrust_map=tuple(float(item) for item in nominal[8:11]),
            nominal_kappa=float(nominal[11]),
            body_rate_max=tuple(float(item) for item in nominal[12:15]),
        )
        native = self._native.set_dynamics(domain.native_dynamics())
        self._native.set_aerodynamics(domain.native_aerodynamics())
        self._dynamics_domain = domain
        self.dynamics_parameters = domain.dynamics_vector
        self.aerodynamics_parameters = domain.aerodynamics_vector
        arm = domain.arm_length
        rotor_xy = arm / np.sqrt(2.0) * np.asarray(
            [[1, -1], [-1, -1], [-1, 1], [1, 1]], dtype=np.float32
        )
        rotor_positions = np.pad(rotor_xy, ((0, 0), (0, 1))).reshape(-1)
        self.actuator_parameters = np.asarray([
            domain.mass, domain.arm_length, *domain.inertia,
            *rotor_positions, 1.0, -1.0, 1.0, -1.0,
            domain.motor_omega_min, domain.motor_omega_max, domain.motor_tau,
            *domain.thrust_map, domain.kappa,
            float(native["motor_thrust_min"]), float(native["motor_thrust_max"]),
        ], np.float32)

    @property
    def dynamics_domain(self) -> DynamicsDomain | None:
        return self._dynamics_domain

    def connect_unity(self, scene_id: int = 1) -> None:
        if not self._native.connect_unity(int(scene_id)):
            raise FlightmareUnavailable(
                "could not connect to Unity; start the renderer before creating a vision env"
            )
        self._connected = True

    @staticmethod
    def _proprio(raw: dict[str, Any]) -> Proprioception:
        return Proprioception(raw["state"], raw["motor_thrusts"], raw["motor_omega"])

    def _ordered_course_progress(
        self, position: np.ndarray, gate_index: int, lap: int
    ) -> float:
        """Continuous race potential: completed route length minus gate distance.

        The cumulative centre-to-centre distance makes switching from a passed
        gate to the next gate approximately continuous, unlike raw active-gate
        distance.  Missing a gate and flying away reduces this potential.
        """

        index = int(gate_index) % len(self.track.gates)
        gate_distance = float(
            np.linalg.norm(np.asarray(position) - self.track.gates[index].position)
        )
        return (
            float(lap) * self._course_length
            + float(self._gate_cumulative_distance[index])
            - gate_distance
        )

    def _reset_flight_plan_randomization(self, seed: int | None) -> None:
        raw = self.flight_plan_randomization
        enabled = bool(raw.get("enabled", False))
        rng = np.random.default_rng((0 if seed is None else int(seed)) ^ 0x6A7E)
        position_std = float(raw.get("position_bias_std_m", 0.0)) if enabled else 0.0
        orientation_std = np.radians(
            float(raw.get("orientation_bias_std_degrees", 0.0)) if enabled else 0.0
        )
        size_std = float(raw.get("size_scale_std", 0.0)) if enabled else 0.0
        if min(position_std, orientation_std, size_std) < 0.0:
            raise ValueError("flight-plan randomization scales must be non-negative")
        self._policy_gate_position_error = rng.normal(
            0.0, position_std, (len(self.track.gates), 3)
        ).astype(np.float32)
        self._policy_gate_orientation_error = rng.normal(
            0.0, orientation_std, (len(self.track.gates), 3)
        ).astype(np.float32)
        self._policy_gate_size_scale = np.clip(
            1.0 + rng.normal(0.0, size_std, len(self.track.gates)), 0.90, 1.10
        ).astype(np.float32)

    def _randomized_flight_plan(
        self, plan: dict[str, np.ndarray], task_state: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if not bool(self.flight_plan_randomization.get("enabled", False)):
            return plan, task_state
        indices = np.asarray(plan["index"], np.int64)
        active = int(indices[0])
        position = np.asarray(plan["position"], np.float32).copy()
        position += (
            self._policy_gate_position_error[indices]
            - self._policy_gate_position_error[active]
        )
        normal = np.asarray(plan["normal"], np.float32).copy()
        up = np.asarray(plan["up"], np.float32).copy()
        normal += self._policy_gate_orientation_error[indices]
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-6)
        up += self._policy_gate_orientation_error[indices]
        up -= normal * np.sum(up * normal, axis=1, keepdims=True)
        up /= np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1.0e-6)
        size = np.asarray(plan["size"], np.float32).copy()
        size *= self._policy_gate_size_scale[indices, None]
        randomized = dict(plan)
        randomized.update(position=position, normal=normal, up=up, size=size)
        randomized["records"] = np.concatenate([
            position, normal, up, size,
            np.asarray(plan["enter_from_opposite_side"], np.float32)[:, None],
            np.asarray(plan["mask"], np.float32)[:, None],
        ], axis=-1).astype(np.float32)
        task = task_state.copy()
        task[0:3] -= self._policy_gate_position_error[active]
        return randomized, task

    def _observation(self, raw: dict[str, Any], include_images: bool) -> dict[str, Any]:
        proprio = self._proprio(raw)
        estimate = self.state_estimator.estimate(proprio)
        policy_proprio = (
            Proprioception(estimate.state, proprio.motor_thrusts, proprio.motor_omega)
            if self.policy_state_source == "estimate" else proprio
        )
        route_remaining = (
            None
            if self._route_plan_total_gates is None
            else max(self._route_plan_total_gates - self.tracker.passed_count, 1)
        )
        gates = None if self.collection_observation else self.tracker.relative_gates(
            policy_proprio.position, policy_proprio.quaternion_wxyz,
            self.next_gate_count, remaining=route_remaining,
        )
        flight_plan = self.track.flight_plan(
            self.tracker.index, self.next_gate_count, remaining=route_remaining
        )
        active_gate = self.track.gates[self.tracker.index]
        gate_from_world = active_gate.directed_rotation.T
        policy_world_from_body = quaternion_matrix(policy_proprio.quaternion_wxyz)
        policy_gate_from_body = gate_from_world @ policy_world_from_body
        task_state = np.concatenate(
            [
                gate_from_world @ (policy_proprio.position - active_gate.position),
                gate_from_world @ policy_proprio.linear_velocity,
                policy_gate_from_body[:, :2].T.reshape(-1),
                policy_proprio.body_rates,
                policy_proprio.motor_omega,
            ]
        ).astype(np.float32)
        flight_plan, task_state = self._randomized_flight_plan(
            flight_plan, task_state
        )
        sim_time = float(raw.get("time", 0.0))
        gate_position = gate_from_world @ (proprio.position - active_gate.position)
        gate_velocity = gate_from_world @ proprio.linear_velocity
        rotation = quaternion_matrix(proprio.quaternion_wxyz)
        plant_descriptor = None
        if self.plant_settings_observation:
            from starscream.plant_privileged import plant_static_settings, plant_settings
            if self._plant_settings_static is None:
                self._plant_settings_static = plant_static_settings(
                    self.dynamics_parameters, self.aerodynamics_parameters,
                    self.actuator_parameters, self.action_delay, self.control_dt)
            plant_descriptor = plant_settings(self._plant_settings_static,
                self.aerodynamics_parameters, sim_time, rotation)
        true_gate_from_body = gate_from_world @ rotation
        true_task_state = np.concatenate([
            gate_position, gate_velocity,
            true_gate_from_body[:, :2].T.reshape(-1),
            proprio.body_rates, proprio.motor_omega,
        ]).astype(np.float32)
        if self.collection_observation:
            # Private teacher-collection view, NOT the production observation
            # schema. Keep exactly the same estimator/RNG, task and plan math.
            # Exclude only unused camera/body-relative/world-route/sky payloads.
            observation = {
                'state': proprio.state, 'motor_thrusts': proprio.motor_thrusts,
                'motor_omega': proprio.motor_omega,
                'state_estimate': np.asarray(estimate.state, np.float32),
                'state_estimate_std': np.asarray(estimate.standard_deviation, np.float32),
                'state_estimate_valid': np.int64(estimate.valid),
                'time': np.asarray(sim_time, np.float32),
                'collision': np.int64(raw.get('collision', False)),
                'previous_action': self._previous_action.copy(),
                'flight_plan': flight_plan, 'task_state': task_state,
                'age': {'previous_action': np.asarray(max(0., sim_time-self._applied_action_timestamp), np.float32)},
                'valid': {'previous_action': np.int64(self._previous_action_valid)},
                'timestamp': {'sim': np.asarray(sim_time, np.float64)},
                'measured': {'body_rates': proprio.body_rates.copy(), 'motor_omega': proprio.motor_omega.copy()},
                'privileged': {'state': proprio.state.copy(), 'gate_state': true_task_state,
                    'dynamics': self.dynamics_parameters.copy(),
                    'aerodynamics': self.aerodynamics_parameters.copy(),
                    'actuator': self.actuator_parameters.copy(),
                    'progress': np.asarray([*gate_position, self.tracker.index,
                        self.tracker.lap, self.tracker.passed_count], np.float32)},
            }
            if plant_descriptor is not None:
                observation['privileged']['plant_settings'] = plant_descriptor
            self._last_observation = observation
            return observation
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2]))
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        yaw_gate = np.arctan2(true_gate_from_body[1, 0], true_gate_from_body[0, 0])
        sky_state = np.concatenate(
            [
                proprio.position, gate_position, proprio.linear_velocity, gate_velocity,
                np.asarray([roll, pitch, yaw, yaw_gate], np.float32),
                proprio.body_rates, proprio.motor_omega,
            ]
        ).astype(np.float32)
        focal = self.image_size[0] / (2.0 * np.tan(np.radians(self.fov_degrees) * 0.5))
        camera_intrinsics = np.asarray(
            [focal, focal, self.image_size[0] * 0.5, self.image_size[1] * 0.5], np.float32
        )
        mask = None
        camera_timestamp = sim_time
        camera_valid = False
        if self.mask_source != "none":
            captured = self._capture_gate_mask(proprio)
            self._mask_queue.append((sim_time, captured))
            target_time = sim_time - self.image_delay
            eligible = [sample for sample in self._mask_queue if sample[0] <= target_time + 1e-9]
            camera_timestamp, mask = eligible[-1] if eligible else self._mask_queue[0]
            while len(self._mask_queue) > 1 and self._mask_queue[1][0] <= camera_timestamp:
                self._mask_queue.popleft()
            camera_valid = sim_time - camera_timestamp + 1e-9 >= self.image_delay
        observation: dict[str, Any] = {
            "state": proprio.state,
            "motor_thrusts": proprio.motor_thrusts,
            "motor_omega": proprio.motor_omega,
            "state_estimate": np.asarray(estimate.state, dtype=np.float32),
            "state_estimate_std": np.asarray(estimate.standard_deviation, dtype=np.float32),
            "state_estimate_valid": np.int64(estimate.valid),
            "time": np.asarray(raw.get("time", 0.0), dtype=np.float32),
            "collision": np.int64(raw.get("collision", False)),
            "previous_action": self._previous_action.copy(),
            "gates": gates,
            "flight_plan": flight_plan,
            "task_state": task_state,
            "measured": {
                "body_rates": proprio.body_rates.copy(),
                "motor_omega": proprio.motor_omega.copy(),
            },
            "timestamp": {
                "sim": np.asarray(sim_time, np.float64),
                "camera": np.asarray(camera_timestamp, np.float64),
                "body_rates": np.asarray(sim_time, np.float64),
                "motor_omega": np.asarray(sim_time, np.float64),
                "state_estimate": np.asarray(sim_time, np.float64),
                "previous_action": np.asarray(self._applied_action_timestamp, np.float64),
            },
            "age": {
                "camera": np.asarray(sim_time - camera_timestamp, np.float32),
                "body_rates": np.asarray(0.0, np.float32),
                "motor_omega": np.asarray(0.0, np.float32),
                "state_estimate": np.asarray(0.0, np.float32),
                "previous_action": np.asarray(max(0.0, sim_time - self._applied_action_timestamp), np.float32),
            },
            "valid": {
                "camera": np.int64(camera_valid),
                "body_rates": np.int64(True),
                "motor_omega": np.int64(True),
                "state_estimate": np.int64(estimate.valid),
                "previous_action": np.int64(self._previous_action_valid),
            },
            "privileged": {
                "world_route_records": np.concatenate([
                    np.stack([self.track.gates[int(i)].position for i in flight_plan["index"]]),
                    np.stack([self.track.gates[int(i)].normal for i in flight_plan["index"]]),
                    np.stack([self.track.gates[int(i)].up for i in flight_plan["index"]]),
                    np.stack([self.track.gates[int(i)].size for i in flight_plan["index"]]),
                    flight_plan["records"][:, 11:13],
                ], axis=-1).astype(np.float32),
                "state": proprio.state.copy(),
                "gate_state": true_task_state,
                    "dynamics": self.dynamics_parameters.copy(),
                    "aerodynamics": self.aerodynamics_parameters.copy(),
                    "camera_extrinsics": self.camera_extrinsics.copy(),
                    "camera_intrinsics": camera_intrinsics,
                    "sky_state": sky_state,
                    "disturbance": np.zeros(10, np.float32),
                "progress": np.asarray(
                    [*gate_position, self.tracker.index, self.tracker.lap, self.tracker.passed_count],
                    np.float32,
                ),
                "actuator": self.actuator_parameters.copy(),
            },
        }
        if mask is not None:
            observation["gate_mask"] = mask
        if include_images and self.retain_render_images:
            images = self._native.render()
            observation.update(
                rgb=np.asarray(images["bgr"])[..., ::-1].copy(),
                depth=np.asarray(images["depth"], dtype=np.float32),
                segmentation=np.asarray(images["segmentation_bgr"])[..., ::-1].copy(),
                optical_flow=np.asarray(images["optical_flow"]).copy(),
            )
        if plant_descriptor is not None:
            observation['privileged']['plant_settings'] = plant_descriptor
        self._last_observation = observation
        return observation

    @staticmethod
    def _nearest_resize(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        width, height = size
        source_height, source_width = mask.shape
        rows = np.minimum((np.arange(height) * source_height / height).astype(int), source_height - 1)
        cols = np.minimum((np.arange(width) * source_width / width).astype(int), source_width - 1)
        return np.asarray(mask[np.ix_(rows, cols)], dtype=np.uint8)

    def _capture_gate_mask(self, proprio: Proprioception) -> np.ndarray:
        if self.mask_source == "geometry":
            geometry = self._render_gate_mask(proprio)
            return self._nearest_resize(geometry, self.mask_size)
        images = self._native.render()
        scene_depth = np.asarray(images["depth"], dtype=np.float32)
        _, gate_depth = self._render_gate_mask_and_depth(proprio)
        valid_scene = np.isfinite(scene_depth) & (scene_depth > 0.0)
        visible = np.isfinite(gate_depth) & (
            ~valid_scene | (gate_depth <= scene_depth + 0.15)
        )
        return self._nearest_resize(np.where(visible, 255, 0).astype(np.uint8), self.mask_size)

    def _render_gate_mask(self, proprio: Proprioception) -> np.ndarray:
        if self.geometry_renderer == "exact":
            mask, _ = self._render_gate_mask_and_depth(proprio)
            return mask
        if self.geometry_renderer == "native_exact":
            if not hasattr(self._flightgym, "rasterize_gate_mask"):
                raise FlightmareUnavailable(
                    "flightgym lacks the native exact mask rasterizer; rebuild the container"
                )
            width, height = self.image_size
            return np.asarray(
                self._flightgym.rasterize_gate_mask(
                    proprio.state, self.camera_extrinsics,
                    self._gate_positions, self._gate_laterals,
                    self._gate_ups, self._gate_sizes,
                    width, height, self.fov_degrees, 0.72,
                ),
                dtype=np.uint8,
            )
        return self._render_gate_mask_fast(proprio)

    def _render_gate_mask_fast(self, proprio: Proprioception) -> np.ndarray:
        """Rasterize the union of visible semantic gate rings.

        The depth-producing implementation below is retained for Unity
        occlusion checks. Geometry-only collection does not need per-pixel
        depth, so use Pillow's compiled polygon rasterizer instead of building
        several full NumPy coordinate grids for every gate and simulator tick.
        Each ring is composed independently before union, preserving a farther
        gate that is visible through the aperture of a nearer one.
        """

        width, height = self.image_size
        focal = width / (2.0 * np.tan(np.radians(self.fov_degrees) * 0.5))
        center = np.asarray([width * 0.5, height * 0.5], dtype=np.float32)
        world_from_body = quaternion_matrix(proprio.quaternion_wxyz)
        body_from_camera = quaternion_matrix(self.camera_extrinsics[3:7])
        camera_from_world = (world_from_body @ body_from_camera).T
        camera_position = (
            proprio.position + world_from_body @ self.camera_extrinsics[:3]
        )
        composite = Image.new("L", (width, height), 0)

        for gate in self.track.gates:
            if not gate.render:
                continue
            half_width, half_height = gate.size * 0.5
            outer_world = np.stack(
                [
                    gate.position
                    + sy * half_width * gate.lateral
                    + sz * half_height * gate.up
                    for sy, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                ]
            )
            inner_world = gate.position + (outer_world - gate.position) * 0.72

            def project(points: np.ndarray) -> list[tuple[int, int]] | None:
                camera = (camera_from_world @ (points - camera_position).T).T
                if np.any(camera[:, 0] <= 0.05):
                    return None
                pixels = np.stack(
                    [
                        center[0] + focal * camera[:, 1] / camera[:, 0],
                        center[1] - focal * camera[:, 2] / camera[:, 0],
                    ],
                    axis=-1,
                )
                rounded = np.rint(pixels).astype(np.int32)
                return [(int(x), int(y)) for x, y in rounded]

            outer, inner = project(outer_world), project(inner_world)
            if outer is None or inner is None:
                continue
            ring = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(ring)
            draw.polygon(outer, fill=255)
            draw.polygon(inner, fill=0)
            composite = ImageChops.lighter(composite, ring)
        return np.asarray(composite, dtype=np.uint8).copy()

    def _render_gate_mask_and_depth(
        self, proprio: Proprioception
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize the nearest semantic gate ring and its optical-axis depth."""

        width, height = self.image_size
        focal = width / (2.0 * np.tan(np.radians(self.fov_degrees) * 0.5))
        center = np.asarray([width * 0.5, height * 0.5], dtype=np.float32)
        world_from_body = quaternion_matrix(proprio.quaternion_wxyz)
        body_from_camera = quaternion_matrix(self.camera_extrinsics[3:7])
        world_from_camera = world_from_body @ body_from_camera
        camera_from_world = world_from_camera.T
        camera_position = proprio.position + world_from_body @ self.camera_extrinsics[:3]
        gate_depth = np.full((height, width), np.inf, dtype=np.float32)

        def convex_pixels(vertices: np.ndarray) -> np.ndarray:
            result = np.zeros((height, width), dtype=np.bool_)
            minimum = np.maximum(np.floor(vertices.min(0)).astype(int), 0)
            maximum = np.minimum(np.ceil(vertices.max(0)).astype(int), [width - 1, height - 1])
            if np.any(maximum < minimum):
                return result
            xs = np.arange(minimum[0], maximum[0] + 1, dtype=np.float32) + 0.5
            ys = np.arange(minimum[1], maximum[1] + 1, dtype=np.float32) + 0.5
            grid_x, grid_y = np.meshgrid(xs, ys)
            points = np.stack([grid_x, grid_y], axis=-1)
            edges = np.roll(vertices, -1, axis=0) - vertices
            relative = points[..., None, :] - vertices[None, None, :, :]
            crosses = edges[None, None, :, 0] * relative[..., 1] - edges[None, None, :, 1] * relative[..., 0]
            inside = np.all(crosses >= 0, axis=-1) | np.all(crosses <= 0, axis=-1)
            result[minimum[1] : maximum[1] + 1, minimum[0] : maximum[0] + 1] = inside
            return result

        for gate in self.track.gates:
            if not gate.render:
                continue
            half_width, half_height = gate.size * 0.5
            outer_world = np.stack(
                [
                    gate.position + sy * half_width * gate.lateral + sz * half_height * gate.up
                    for sy, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                ]
            )
            inner_scale = 0.72
            inner_world = gate.position + (outer_world - gate.position) * inner_scale

            def project(points):
                camera = (camera_from_world @ (points - camera_position).T).T
                if np.any(camera[:, 0] <= 0.05):
                    return None
                pixels = np.stack(
                    [center[0] + focal * camera[:, 1] / camera[:, 0],
                     center[1] - focal * camera[:, 2] / camera[:, 0]],
                    axis=-1,
                )
                return np.rint(pixels).astype(np.int32), float(np.mean(camera[:, 0]))

            outer_result, inner_result = project(outer_world), project(inner_world)
            if outer_result is not None and inner_result is not None:
                outer, axial_depth = outer_result
                inner, _ = inner_result
                ring = convex_pixels(outer.astype(np.float32)) & ~convex_pixels(
                    inner.astype(np.float32)
                )
                gate_depth[ring] = np.minimum(gate_depth[ring], axial_depth)
        return np.where(np.isfinite(gate_depth), 255, 0).astype(np.uint8), gate_depth

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Explicit reset is the recovery path after a failed prepared native step.
        self._step_pending = False
        super().reset(seed=seed)
        options = options or {}
        route_plan_total = options.get("route_plan_total_gates")
        self._route_plan_total_gates = (
            None if route_plan_total is None else int(route_plan_total)
        )
        if self._route_plan_total_gates is not None and self._route_plan_total_gates < 1:
            raise ValueError("route_plan_total_gates must be positive")
        self._reset_flight_plan_randomization(seed)
        if self.action_delay_range is not None:
            delay_rng = np.random.default_rng(
                (0 if seed is None else int(seed)) ^ 0xAC710
            )
            self.action_delay = float(delay_rng.uniform(*self.action_delay_range))
        else:
            self.action_delay = self._nominal_action_delay
        self._plant_settings_static = None
        if self.dynamics_randomization is not None:
            dynamics_seed = int(options.get(
                "dynamics_seed",
                seed if seed is not None else self.np_random.integers(0, 2**31 - 1),
            ))
            self._apply_dynamics_domain(dynamics_seed)
        if "state" in options:
            state = np.asarray(self._native.reset_to(options["state"], 0.0), dtype=np.float32)
        else:
            start_index = int(options.get("gate_index", 0))
            gate = self.track.gates[start_index % len(self.track.gates)]
            state = np.zeros(25, dtype=np.float32)
            state[0:3] = gate.position - 3.0 * gate.normal
            state[2] = max(float(state[2]), 1.0)
            state[3:7] = matrix_quaternion(gate.directed_rotation)
            state = np.asarray(self._native.reset_to(state, 0.0), dtype=np.float32)
        start_index = int(options.get("gate_index", 0))
        self.tracker.reset(state[0:3], start_index)
        archived_action = options.get("previous_action")
        if archived_action is None:
            self._previous_action.fill(0.0)
            self._previous_action_valid = False
            initial_applied = np.asarray([9.81, 0.0, 0.0, 0.0], np.float32)
        else:
            initial_applied = np.asarray(archived_action, np.float32)
            if initial_applied.shape != (4,) or not np.all(np.isfinite(initial_applied)):
                raise ValueError("reset previous_action must be a finite CTBR vector")
            initial_applied = np.clip(
                initial_applied, [0.0, -6.0, -6.0, -6.0],
                [self.maximum_collective_thrust, 6.0, 6.0, 6.0]
            ).astype(np.float32)
            self._previous_action = initial_applied.copy()
            self._previous_action_valid = True
        self._mask_queue.clear()
        self._action_queue.clear()
        self._applied_action = initial_applied
        self._applied_action_timestamp = 0.0 if archived_action is not None else -self.action_delay
        try:
            self.state_estimator.reset(seed)
        except TypeError:
            # Preserve compatibility with externally supplied estimators using
            # the original no-argument protocol.
            self.state_estimator.reset()
        initial_progress = self._ordered_course_progress(
            state[0:3], self.tracker.index, self.tracker.lap
        )
        self.reward_function.reset(
            state[0:3],
            self.track.gates[self.tracker.index],
            course_progress=initial_progress,
        )
        raw = self._native.get_proprioception()
        observation = self._observation(raw, self.render_observations)
        spawn_metadata = dict(options.get("spawn", {}))
        if self._dynamics_domain is not None:
            spawn_metadata["dynamics_domain_seed"] = self._dynamics_domain.seed
        return observation, {
            "gate_index": self.tracker.index,
            "track": self.track.name,
            "track_fingerprint": self.track.fingerprint,
            "seed": -1 if seed is None else int(seed),
            "spawn": spawn_metadata,
        }

    def step(self, action):
        prepared = self.prepare_step(action)
        raw = self._native.step_ctbr(float(prepared[0]), prepared[1:], self.control_dt)
        return self.finish_step(raw)

    def prepare_step(self, action):
        """Resolve the delay queue before a native batch; must pair with finish_step."""
        if getattr(self, '_step_pending', False):
            raise RuntimeError('previous prepared step has not completed')
        command = CTBRAction.from_array(action)
        action_timestamp = float(self._last_observation["time"]) if self._last_observation else 0.0
        self._action_queue.append((action_timestamp, command.as_array()))
        target_time = action_timestamp - self.action_delay
        eligible = [sample for sample in self._action_queue if sample[0] <= target_time + 1e-9]
        if eligible:
            self._applied_action_timestamp, self._applied_action = eligible[-1]
            while len(self._action_queue) > 1 and self._action_queue[1][0] <= self._applied_action_timestamp:
                self._action_queue.popleft()
        applied_command = CTBRAction.from_array(self._applied_action)
        self._step_pending = (action_timestamp, applied_command)
        return applied_command.as_array()

    def finish_step(self, raw):
        """Apply the original gate/reward/observation contract to native telemetry."""
        if not getattr(self, '_step_pending', False):
            raise RuntimeError('finish_step without prepare_step')
        action_timestamp, applied_command = self._step_pending
        self._step_pending = False
        self._previous_action = applied_command.as_array()
        self._previous_action_valid = True
        proprio = self._proprio(raw)
        active_gate_index = self.tracker.index
        active_gate = self.track.gates[active_gate_index]
        course_progress = self._ordered_course_progress(
            proprio.position, active_gate_index, self.tracker.lap
        )
        gate_passed = self.tracker.update(proprio.position)
        ground_contact = bool(proprio.position[2] <= 0.02)
        unity_collision = bool(raw.get("collision", False))
        terminated = bool(ground_contact or (self.terminate_on_collision and unity_collision))
        reward_result = self.reward_function(
            state=proprio.state,
            action=applied_command,
            active_gate=active_gate,
            gate_passed=gate_passed,
            crashed=ground_contact or unity_collision,
            course_progress=course_progress,
        )
        observation = self._observation(raw, self.render_observations)
        reward = reward_result.total
        applied_motor_thrusts = np.asarray(raw["motor_thrusts"], dtype=np.float32)
        thrust_min, thrust_max = self.actuator_parameters[-2:]
        applied_motor_normalized = np.clip(
            (applied_motor_thrusts - thrust_min) / max(float(thrust_max - thrust_min), 1e-6),
            0.0,
            1.0,
        ).astype(np.float32)
        info = {
            "gate_index": self.tracker.index,
            "gate_passed": gate_passed,
            "course_progress": course_progress,
            "time": float(raw["time"]),
            "ctbr_command": np.asarray(raw["ctbr_command"], dtype=np.float32),
            "reward_components": dict(reward_result.components),
            "unity_collision": unity_collision,
            "ground_contact": ground_contact,
            "action_command_timestamp": action_timestamp,
            "action_applied_timestamp": self._applied_action_timestamp,
            "applied_ctbr": np.asarray(raw["ctbr_command"], dtype=np.float32),
            "applied_motor_thrusts": applied_motor_thrusts,
            "applied_motor_omega": np.asarray(raw["motor_omega"], dtype=np.float32),
            "applied_motor_normalized": applied_motor_normalized,
        }
        return observation, reward, terminated, False, info

    def render(self):
        if not self._connected:
            raise FlightmareUnavailable("Unity is not connected")
        return np.asarray(self._native.render()["bgr"])[..., ::-1].copy()

    def close(self) -> None:
        if self._connected:
            self._native.disconnect_unity()
            self._connected = False
        if hasattr(self._native, "close"):
            self._native.close()

    def collection_metadata(self) -> dict[str, Any]:
        return {
            "contract": "starscream-schema-v4",
            "control_dt": self.control_dt,
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
            "fov_degrees": self.fov_degrees,
            "track": self.track.name,
            "track_fingerprint": self.track.fingerprint,
            "coordinate_convention": "world-z-up_body-x-forward_quaternion-wxyz",
            "action_convention": "mass-normalized-ctbr",
            "maximum_collective_thrust_mps2": self.maximum_collective_thrust,
            "dynamics_parameter_names": ",".join(self.dynamics_parameter_names),
            "aerodynamics_parameter_names": ",".join(self.aerodynamics_parameter_names),
            "dynamics_randomization": int(
                self.dynamics_randomization is not None
                and self.dynamics_randomization.enabled
            ),
            "dynamics_randomization_fingerprint": (
                self.dynamics_randomization.fingerprint
                if self.dynamics_randomization is not None else "none"
            ),
            "actuator_parameter_names": ",".join(self.actuator_parameter_names),
            "state_parameter_names": ",".join(self.state_parameter_names),
            "sky_state_parameter_names": ",".join(self.sky_state_parameter_names),
            "mask_source": self.mask_source,
            "mask_width": self.mask_size[0],
            "mask_height": self.mask_size[1],
            "image_delay_requested": self.image_delay,
            "action_delay_requested": self.action_delay,
            "action_delay_range": (
                "none" if self.action_delay_range is None
                else ",".join(str(value) for value in self.action_delay_range)
            ),
            "policy_state_source": self.policy_state_source,
            "retains_direct_images": int(self.retain_render_images),
            "unity_mask_method": (
                "semantic_gate_rasterization_with_unity_depth_occlusion"
                if self.mask_source == "unity"
                else "not_applicable"
            ),
        }
