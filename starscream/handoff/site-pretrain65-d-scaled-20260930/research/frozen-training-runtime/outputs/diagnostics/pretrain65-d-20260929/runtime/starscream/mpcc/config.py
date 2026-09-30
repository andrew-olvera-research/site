"""Configuration contracts for the MPCC model, planner, and solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True, slots=True)
class VehicleModelConfig:
    """Pinned Flightmare quadrotor and low-level rate-controller parameters."""

    mass: float = 0.73
    arm_length: float = 0.17
    motor_omega_min: float = 150.0
    motor_omega_max: float = 3000.0
    motor_tau: float = 0.0001
    thrust_map: tuple[float, float, float] = (
        1.3298253500372892e-6,
        0.0038360810526746033,
        -1.7689986848125325,
    )
    kappa: float = 0.016
    body_rate_max: tuple[float, float, float] = (6.0, 6.0, 6.0)
    rate_gain: tuple[float, float, float] = (16.6, 16.6, 5.0)
    gravity: float = 9.81
    integration_dt_max: float = 0.0025
    # Flightmare d4218ae leaves this at the constructor's 2000 rad/s for its
    # ordinary YAML path. The project randomization setter explicitly refreshes
    # the envelope and ``from_observation_parameters`` switches to its sampled
    # motor maximum; this default preserves old-checkpoint rollout parity.
    effective_thrust_omega_max: float = 2000.0
    linear_drag: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quadratic_drag: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotor_drag: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_drag: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wind_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_flightmare_yaml(cls, path: str | Path) -> "VehicleModelConfig":
        defaults = cls()
        with Path(path).open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        values = document.get("quadrotor_dynamics", {})
        return cls(
            mass=float(values.get("mass", defaults.mass)),
            arm_length=float(values.get("arm_l", defaults.arm_length)),
            motor_omega_min=float(values.get("motor_omega_min", defaults.motor_omega_min)),
            motor_omega_max=float(values.get("motor_omega_max", defaults.motor_omega_max)),
            motor_tau=float(values.get("motor_tau", defaults.motor_tau)),
            thrust_map=tuple(float(x) for x in values.get("thrust_map", defaults.thrust_map)),
            kappa=float(values.get("kappa", defaults.kappa)),
            body_rate_max=tuple(float(x) for x in values.get("omega_max", defaults.body_rate_max)),
        )

    @classmethod
    def from_observation_parameters(
        cls,
        dynamics: np.ndarray,
        aerodynamics: np.ndarray,
        *,
        time: float = 0.0,
        defaults: "VehicleModelConfig | None" = None,
    ) -> "VehicleModelConfig":
        base = defaults or cls()
        dynamics = np.asarray(dynamics, np.float64)
        aerodynamics = np.asarray(aerodynamics, np.float64)
        if dynamics.shape != (15,) or aerodynamics.shape != (27,):
            raise ValueError("runtime vehicle parameters require dynamics=(15,), aerodynamics=(27,)")
        wind = aerodynamics[12:15] + aerodynamics[15:18] * np.sin(
            2.0 * np.pi * aerodynamics[18:21] * float(time) + aerodynamics[21:24]
        )
        return cls(
            mass=float(dynamics[0]),
            arm_length=float(dynamics[1]),
            motor_omega_min=float(dynamics[5]),
            motor_omega_max=float(dynamics[6]),
            motor_tau=float(dynamics[7]),
            thrust_map=tuple(float(item) for item in dynamics[8:11]),
            kappa=float(dynamics[11]),
            body_rate_max=tuple(float(item) for item in dynamics[12:15]),
            rate_gain=base.rate_gain,
            gravity=base.gravity,
            integration_dt_max=base.integration_dt_max,
            effective_thrust_omega_max=float(dynamics[6]),
            linear_drag=tuple(float(item) for item in aerodynamics[0:3]),
            quadratic_drag=tuple(float(item) for item in aerodynamics[3:6]),
            rotor_drag=tuple(float(item) for item in aerodynamics[6:9]),
            angular_drag=tuple(float(item) for item in aerodynamics[9:12]),
            wind_world=tuple(float(item) for item in wind),
            center_of_mass=tuple(float(item) for item in aerodynamics[24:27]),
        )

    @property
    def inertia(self) -> np.ndarray:
        return (
            self.mass
            / 12.0
            * self.arm_length**2
            * np.asarray([4.5, 4.5, 7.0], dtype=np.float64)
        )

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class MPCCWeights:
    contour: float = 52.557252608935876
    lag: float = 3.619793616954658
    progress: float = 3.1098212243821344
    progress_tracking: float = 0.2
    velocity: float = 2.298947894406991
    attitude: float = 3.0
    body_rate: float = 0.035
    action: float = 0.012
    action_smoothness: float = 0.015133307188757177
    slack: float = 9185.334361902462
    terminal: float = 12.0


def _default_time_steps() -> tuple[float, ...]:
    # Dense near the feedback action and gradually coarser in the tail.
    return tuple([0.02] * 12 + [0.03] * 12 + [0.04] * 10)


@dataclass(frozen=True, slots=True)
class MPCCConfig:
    time_steps: tuple[float, ...] = field(default_factory=_default_time_steps)
    weights: MPCCWeights = field(default_factory=MPCCWeights)
    backend: str = "auto"
    max_progress_speed: float = 24.0
    # ``profile`` reproduces the accuracy-first teacher by tracking the
    # curvature-limited speed envelope. ``time_optimal`` makes the envelope a
    # terminal feasibility cue and asks the OCP to maximize progress inside the
    # spatial tunnel, matching the central MPCC++ formulation.
    progress_reference_mode: str = "profile"
    time_optimal_progress_fraction: float = 1.0
    terminal_speed_envelope: bool = False
    # First-order spatial MPCC: evaluate lag error against a path point that
    # moves with the optimized progress state instead of a fixed timestamped
    # reference. Kept opt-in so existing collection configs retain their
    # objective structure.
    progress_coupled_contouring: bool = False
    # Qualified accuracy baseline.  Raise this only through the speed sweep
    # after the all-gates/all-starts chaining suite remains perfect.
    nominal_speed: float = 6.0
    track_speed_overrides: tuple[tuple[str, float], ...] = ()
    # Accuracy-first spatial speed-profile limits. Lateral acceleration retains
    # thrust reserve for gravity and feedback instead of consuming the entire
    # collective envelope in the nominal reference.
    maximum_acceleration: float = 12.0
    maximum_longitudinal_acceleration: float = 6.0
    maximum_braking_acceleration: float = 8.0
    minimum_collective_thrust: float = 0.0
    maximum_collective_thrust: float = 21.0
    collective_slew_limit: float = 4.0
    body_rate_slew_limit: float = 2.0
    corridor_margin: float = 0.28422954428223957
    recovery_distance: float = 1.5
    hard_recovery_distance: float = 5.0
    reverse_progress_speed: float = 2.0
    warm_start_position_tolerance: float = 2.0
    warm_start_progress_tolerance: float = 4.0
    terminal_speed: float = 8.0
    actuation_delay: float = 0.0

    def __post_init__(self) -> None:
        steps = np.asarray(self.time_steps, dtype=np.float64)
        if steps.ndim != 1 or len(steps) < 2 or np.any(steps <= 0):
            raise ValueError("time_steps must be a positive one-dimensional grid")
        if self.backend not in {"auto", "acados", "predictive"}:
            raise ValueError("backend must be auto, acados, or predictive")
        if self.progress_reference_mode not in {"profile", "time_optimal"}:
            raise ValueError("progress_reference_mode must be profile or time_optimal")
        if not 0 < self.time_optimal_progress_fraction <= 1.0:
            raise ValueError("time_optimal_progress_fraction must be in (0,1]")
        if self.minimum_collective_thrust < 0 or (
            self.maximum_collective_thrust <= self.minimum_collective_thrust
        ):
            raise ValueError("invalid collective-thrust bounds")
        if self.collective_slew_limit <= 0 or self.body_rate_slew_limit <= 0:
            raise ValueError("action slew limits must be positive")
        if min(
            self.maximum_acceleration,
            self.maximum_longitudinal_acceleration,
            self.maximum_braking_acceleration,
        ) <= 0:
            raise ValueError("acceleration limits must be positive")
        if any(not name or speed <= 0 for name, speed in self.track_speed_overrides):
            raise ValueError("track speed overrides must contain names and positive speeds")
        if self.actuation_delay < 0:
            raise ValueError("actuation_delay cannot be negative")

    @property
    def horizon(self) -> int:
        return len(self.time_steps)

    @property
    def horizon_seconds(self) -> float:
        return float(sum(self.time_steps))

    def nominal_speed_for_track(self, track_name: str) -> float:
        return float(dict(self.track_speed_overrides).get(track_name, self.nominal_speed))

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()
