"""Deterministic Flightmare plant and aerodynamic domain randomization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np


AERODYNAMIC_PARAMETER_NAMES = (
    *(f"linear_drag_{axis}" for axis in "xyz"),
    *(f"quadratic_drag_{axis}" for axis in "xyz"),
    *(f"rotor_drag_{axis}" for axis in "xyz"),
    *(f"angular_drag_{axis}" for axis in "xyz"),
    *(f"wind_mean_{axis}" for axis in "xyz"),
    *(f"wind_gust_amplitude_{axis}" for axis in "xyz"),
    *(f"wind_gust_frequency_{axis}" for axis in "xyz"),
    *(f"wind_gust_phase_{axis}" for axis in "xyz"),
    *(f"center_of_mass_{axis}" for axis in "xyz"),
)


@dataclass(frozen=True, slots=True)
class DynamicsRandomizationConfig:
    """Moderate system-identification envelope for agile 5-inch aircraft.

    The ranges intentionally cover every plant parameter exposed by pinned
    Flightmare plus the project-owned aerodynamic extension.  They are broad
    enough to prevent nominal-plant memorization without making a perfect MPCC
    lap physically meaningless.
    """

    enabled: bool = True
    # Kept at zero for historical-replay compatibility. New sim-to-real
    # experiments must opt in explicitly (the v6.3 contract uses 0.10). A
    # continuous aerodynamic distribution otherwise assigns exactly zero
    # probability to the nominal zero-drag/zero-wind plant.
    nominal_probability: float = 0.0
    mass_scale: tuple[float, float] = (0.85, 1.15)
    arm_length_scale: tuple[float, float] = (0.92, 1.08)
    motor_tau_scale: tuple[float, float] = (0.75, 1.40)
    motor_omega_min_scale: tuple[float, float] = (0.90, 1.10)
    motor_omega_max_scale: tuple[float, float] = (0.95, 1.05)
    thrust_map_scale: tuple[float, float] = (0.90, 1.10)
    kappa_scale: tuple[float, float] = (0.85, 1.15)
    # Keep the sampled native limit inside the public policy action contract
    # (|body_rate| <= 6 rad/s) while still varying Flightmare's final clamp.
    body_rate_max_scale: tuple[float, float] = (0.85, 1.00)
    linear_drag_xy: tuple[float, float] = (0.04, 0.16)
    linear_drag_z: tuple[float, float] = (0.06, 0.22)
    quadratic_drag_xy: tuple[float, float] = (0.008, 0.030)
    quadratic_drag_z: tuple[float, float] = (0.010, 0.040)
    rotor_drag_xy: tuple[float, float] = (3.0e-6, 12.0e-6)
    rotor_drag_z: tuple[float, float] = (0.0, 2.0e-6)
    angular_drag: tuple[float, float] = (0.001, 0.008)
    wind_horizontal: tuple[float, float] = (-1.2, 1.2)
    wind_vertical: tuple[float, float] = (-0.25, 0.25)
    gust_horizontal: tuple[float, float] = (0.0, 0.80)
    gust_vertical: tuple[float, float] = (0.0, 0.20)
    gust_frequency_hz: tuple[float, float] = (0.15, 0.80)
    center_of_mass_xy: tuple[float, float] = (-0.006, 0.006)
    center_of_mass_z: tuple[float, float] = (-0.003, 0.003)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.nominal_probability) <= 1.0:
            raise ValueError("nominal_probability must lie in [0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DynamicsRandomizationConfig | None":
        if value is None:
            return None
        return cls(**dict(value))

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class DynamicsDomain:
    seed: int
    mass: float
    arm_length: float
    motor_omega_min: float
    motor_omega_max: float
    motor_tau: float
    thrust_map: tuple[float, float, float]
    kappa: float
    body_rate_max: tuple[float, float, float]
    linear_drag: tuple[float, float, float]
    quadratic_drag: tuple[float, float, float]
    rotor_drag: tuple[float, float, float]
    angular_drag: tuple[float, float, float]
    wind_mean: tuple[float, float, float]
    wind_gust_amplitude: tuple[float, float, float]
    wind_gust_frequency: tuple[float, float, float]
    wind_gust_phase: tuple[float, float, float]
    center_of_mass: tuple[float, float, float]

    @property
    def inertia(self) -> np.ndarray:
        return (
            self.mass / 12.0 * self.arm_length**2
            * np.asarray([4.5, 4.5, 7.0], np.float64)
        )

    @property
    def dynamics_vector(self) -> np.ndarray:
        return np.asarray([
            self.mass, self.arm_length, *self.inertia,
            self.motor_omega_min, self.motor_omega_max, self.motor_tau,
            *self.thrust_map, self.kappa, *self.body_rate_max,
        ], np.float32)

    @property
    def aerodynamics_vector(self) -> np.ndarray:
        return np.asarray([
            *self.linear_drag, *self.quadratic_drag, *self.rotor_drag,
            *self.angular_drag, *self.wind_mean, *self.wind_gust_amplitude,
            *self.wind_gust_frequency, *self.wind_gust_phase,
            *self.center_of_mass,
        ], np.float32)

    def native_dynamics(self) -> dict[str, Any]:
        return {
            "mass": self.mass,
            "arm_length": self.arm_length,
            "motor_omega_min": self.motor_omega_min,
            "motor_omega_max": self.motor_omega_max,
            "motor_tau": self.motor_tau,
            "thrust_map": self.thrust_map,
            "kappa": self.kappa,
            "body_rate_max": self.body_rate_max,
        }

    def native_aerodynamics(self) -> dict[str, Any]:
        return {
            "linear_drag": self.linear_drag,
            "quadratic_drag": self.quadratic_drag,
            "rotor_drag": self.rotor_drag,
            "angular_drag": self.angular_drag,
            "wind_mean": self.wind_mean,
            "wind_gust_amplitude": self.wind_gust_amplitude,
            "wind_gust_frequency": self.wind_gust_frequency,
            "wind_gust_phase": self.wind_gust_phase,
            "center_of_mass": self.center_of_mass,
        }


def _uniform(rng: np.random.Generator, interval: tuple[float, float], size: int | None = None):
    return rng.uniform(float(interval[0]), float(interval[1]), size=size)


def _deterministic_probability_event(seed: int, probability: float, *, salt: str) -> bool:
    """Draw a stable mixture event without perturbing legacy parameter draws."""
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    digest = hashlib.sha256(f"{int(seed)}:{salt}".encode()).digest()
    unit = int.from_bytes(digest[:8], "little") / float(2**64)
    return unit < probability


def sample_dynamics_domain(
    config: DynamicsRandomizationConfig,
    *,
    seed: int,
    nominal_mass: float,
    nominal_arm_length: float,
    nominal_motor_omega_min: float,
    nominal_motor_omega_max: float,
    nominal_motor_tau: float,
    nominal_thrust_map: tuple[float, float, float],
    nominal_kappa: float,
    body_rate_max: tuple[float, float, float],
) -> DynamicsDomain:
    rng = np.random.default_rng(int(seed))
    if not config.enabled or _deterministic_probability_event(
        seed, float(config.nominal_probability), salt="nominal-dynamics-v1"
    ):
        return DynamicsDomain(
            int(seed), nominal_mass, nominal_arm_length,
            nominal_motor_omega_min, nominal_motor_omega_max, nominal_motor_tau,
            nominal_thrust_map, nominal_kappa, body_rate_max,
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        )
    thrust_scale = float(_uniform(rng, config.thrust_map_scale))
    return DynamicsDomain(
        seed=int(seed),
        mass=nominal_mass * float(_uniform(rng, config.mass_scale)),
        arm_length=nominal_arm_length * float(_uniform(rng, config.arm_length_scale)),
        motor_omega_min=nominal_motor_omega_min * float(_uniform(rng, config.motor_omega_min_scale)),
        motor_omega_max=nominal_motor_omega_max * float(_uniform(rng, config.motor_omega_max_scale)),
        motor_tau=nominal_motor_tau * float(_uniform(rng, config.motor_tau_scale)),
        thrust_map=tuple(float(value * thrust_scale) for value in nominal_thrust_map),
        kappa=nominal_kappa * float(_uniform(rng, config.kappa_scale)),
        body_rate_max=tuple(
            float(item * scale)
            for item, scale in zip(
                body_rate_max,
                _uniform(rng, config.body_rate_max_scale, size=3),
                strict=True,
            )
        ),
        linear_drag=tuple(float(item) for item in [
            *_uniform(rng, config.linear_drag_xy, size=2),
            _uniform(rng, config.linear_drag_z),
        ]),
        quadratic_drag=tuple(float(item) for item in [
            *_uniform(rng, config.quadratic_drag_xy, size=2),
            _uniform(rng, config.quadratic_drag_z),
        ]),
        rotor_drag=tuple(float(item) for item in [
            *_uniform(rng, config.rotor_drag_xy, size=2),
            _uniform(rng, config.rotor_drag_z),
        ]),
        angular_drag=tuple(float(item) for item in _uniform(rng, config.angular_drag, size=3)),
        wind_mean=tuple(float(item) for item in [
            *_uniform(rng, config.wind_horizontal, size=2),
            _uniform(rng, config.wind_vertical),
        ]),
        wind_gust_amplitude=tuple(float(item) for item in [
            *_uniform(rng, config.gust_horizontal, size=2),
            _uniform(rng, config.gust_vertical),
        ]),
        wind_gust_frequency=tuple(float(item) for item in _uniform(
            rng, config.gust_frequency_hz, size=3
        )),
        wind_gust_phase=tuple(float(item) for item in rng.uniform(-np.pi, np.pi, size=3)),
        center_of_mass=tuple(float(item) for item in [
            *_uniform(rng, config.center_of_mass_xy, size=2),
            _uniform(rng, config.center_of_mass_z),
        ]),
    )
