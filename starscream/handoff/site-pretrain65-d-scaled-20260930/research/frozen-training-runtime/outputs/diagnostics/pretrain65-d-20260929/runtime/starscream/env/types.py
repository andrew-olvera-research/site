"""Stable array contracts shared by simulators, agents, and datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


F32 = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class CTBRAction:
    """Mass-normalized collective thrust [m/s^2] and body rates [rad/s]."""

    collective_thrust: float
    body_rates: F32

    def __post_init__(self) -> None:
        rates = np.asarray(self.body_rates, dtype=np.float32)
        if rates.shape != (3,) or not np.all(np.isfinite(rates)):
            raise ValueError("body_rates must be a finite (3,) vector")
        if not np.isfinite(self.collective_thrust):
            raise ValueError("collective_thrust must be finite")
        object.__setattr__(self, "body_rates", rates)

    @classmethod
    def from_array(cls, value: ArrayLike) -> "CTBRAction":
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (4,):
            raise ValueError("CTBR action must have shape (4,): [c, wx, wy, wz]")
        return cls(float(array[0]), array[1:])

    def as_array(self) -> F32:
        return np.concatenate(
            (np.asarray([self.collective_thrust], dtype=np.float32), self.body_rates)
        )


@dataclass(frozen=True, slots=True)
class ControllerCommand:
    """One expert command plus the diagnostics needed to reproduce collection.

    Timestamps use simulator time unless an adapter explicitly records another
    clock domain in dataset metadata.  Reference arrays have fixed shapes so
    HDF5 episodes remain directly batchable.
    """

    action: CTBRAction
    source_timestamp: float = 0.0
    receive_timestamp: float = 0.0
    reference_state: F32 = field(default_factory=lambda: np.zeros(25, np.float32))
    reference_action: F32 = field(default_factory=lambda: np.zeros(4, np.float32))
    reference_progress: float = 0.0
    solver_status: int = 0
    solve_time: float = 0.0
    constraint_margin: float = 0.0
    valid: bool = True
    source: str = "unknown"
    diagnostics: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = np.asarray(self.reference_state, dtype=np.float32)
        reference_action = np.asarray(self.reference_action, dtype=np.float32)
        scalars = np.asarray(
            [
                self.source_timestamp,
                self.receive_timestamp,
                self.reference_progress,
                self.solve_time,
                self.constraint_margin,
            ],
            dtype=np.float64,
        )
        if state.shape != (25,) or reference_action.shape != (4,):
            raise ValueError("controller references must have shapes state=(25,), action=(4,)")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(reference_action)):
            raise ValueError("controller references must be finite")
        if not np.all(np.isfinite(scalars)):
            raise ValueError("controller timing and diagnostics must be finite")
        diagnostics: dict[str, np.ndarray] = {}
        for name, value in self.diagnostics.items():
            if not name or "/" in name:
                raise ValueError("controller diagnostic names must be non-empty path components")
            array = np.asarray(value)
            if array.dtype.kind not in {"b", "i", "u", "f"} or not np.all(np.isfinite(array)):
                raise ValueError(f"controller diagnostic {name!r} must be finite and numeric")
            diagnostics[name] = array
        object.__setattr__(self, "reference_state", state)
        object.__setattr__(self, "reference_action", reference_action)
        object.__setattr__(self, "diagnostics", diagnostics)

    @classmethod
    def from_action(
        cls,
        value: ArrayLike | CTBRAction,
        *,
        timestamp: float = 0.0,
        reference_state: ArrayLike | None = None,
        source: str = "unknown",
    ) -> "ControllerCommand":
        action = value if isinstance(value, CTBRAction) else CTBRAction.from_array(value)
        return cls(
            action=action,
            source_timestamp=float(timestamp),
            receive_timestamp=float(timestamp),
            reference_state=(
                np.zeros(25, np.float32)
                if reference_state is None
                else np.asarray(reference_state, np.float32)
            ),
            reference_action=action.as_array(),
            source=source,
        )


@dataclass(frozen=True, slots=True)
class Proprioception:
    """Flightmare's complete 25-value quadrotor state plus actuator telemetry."""

    state: F32
    motor_thrusts: F32
    motor_omega: F32

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32)
        thrusts = np.asarray(self.motor_thrusts, dtype=np.float32)
        omega = np.asarray(self.motor_omega, dtype=np.float32)
        if state.shape != (25,):
            raise ValueError("state must use Flightmare's 25-value QuadState layout")
        if thrusts.shape != (4,) or omega.shape != (4,):
            raise ValueError("motor telemetry must have shape (4,)")
        if not all(np.all(np.isfinite(x)) for x in (state, thrusts, omega)):
            raise ValueError("proprioception must be finite")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "motor_thrusts", thrusts)
        object.__setattr__(self, "motor_omega", omega)

    @property
    def position(self) -> F32:
        return self.state[0:3]

    @property
    def quaternion_wxyz(self) -> F32:
        return self.state[3:7]

    @property
    def linear_velocity(self) -> F32:
        return self.state[7:10]

    @property
    def body_rates(self) -> F32:
        return self.state[10:13]

    @property
    def linear_acceleration(self) -> F32:
        return self.state[13:16]

    @property
    def body_torque(self) -> F32:
        return self.state[16:19]
