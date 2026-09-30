"""Dependency-light boundary for collecting Agilicious CTBR commands.

The licensed Agilicious stack stays outside this Python package. A ROS callback
or in-process binding feeds messages into :class:`AgiliciousCommandBuffer`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Condition
import time
from typing import Any, Callable

import numpy as np

from .types import CTBRAction, ControllerCommand


@dataclass(frozen=True, slots=True)
class TimedCTBR:
    timestamp: float
    action: CTBRAction
    received_timestamp: float = 0.0
    source: str = "agilicious"
    reference_state: np.ndarray | None = None
    reference_action: np.ndarray | None = None
    reference_progress: float = 0.0
    solver_status: int = 1
    solve_time: float = 0.0
    constraint_margin: float = 0.0
    valid: bool = True

    def controller_command(self, observation: dict[str, Any]) -> ControllerCommand:
        state = (
            np.asarray(observation["privileged"]["state"], np.float32)
            if self.reference_state is None
            else np.asarray(self.reference_state, np.float32)
        )
        reference_action = (
            self.action.as_array()
            if self.reference_action is None
            else np.asarray(self.reference_action, np.float32)
        )
        return ControllerCommand(
            action=self.action,
            source_timestamp=self.timestamp,
            receive_timestamp=self.received_timestamp,
            reference_state=state,
            reference_action=reference_action,
            reference_progress=self.reference_progress,
            solver_status=self.solver_status,
            solve_time=self.solve_time,
            constraint_margin=self.constraint_margin,
            valid=self.valid,
            source=self.source,
        )


def ctbr_from_message(message: Any) -> CTBRAction:
    """Extract CTBR from common Agilicious/ROS message object layouts.

    Adapter code should be explicit about thrust units: this project stores
    mass-normalized collective thrust in m/s^2, matching Flightmare Command.
    """

    thrust = getattr(message, "collective_thrust", getattr(message, "thrust", None))
    rates = getattr(message, "body_rates", getattr(message, "omega", None))
    if rates is None or thrust is None:
        raise ValueError("message needs collective_thrust/thrust and body_rates/omega")
    if hasattr(rates, "x"):
        rates = [rates.x, rates.y, rates.z]
    return CTBRAction(float(thrust), np.asarray(rates, dtype=np.float32))


class AgiliciousCommandBuffer:
    """Thread-safe latest-command source suitable for a ROS subscriber callback."""

    def __init__(
        self,
        converter: Callable[[Any], CTBRAction | ControllerCommand] = ctbr_from_message,
        maxlen: int = 4096,
    ) -> None:
        self.converter = converter
        self._messages: deque[TimedCTBR] = deque(maxlen=maxlen)
        self._condition = Condition()

    def callback(self, message: Any, timestamp: float | None = None) -> None:
        received = time.monotonic()
        converted = self.converter(message)
        if isinstance(converted, ControllerCommand):
            sample = TimedCTBR(
                timestamp=float(converted.source_timestamp),
                action=converted.action,
                received_timestamp=float(converted.receive_timestamp),
                source=converted.source,
                reference_state=converted.reference_state,
                reference_action=converted.reference_action,
                reference_progress=converted.reference_progress,
                solver_status=converted.solver_status,
                solve_time=converted.solve_time,
                constraint_margin=converted.constraint_margin,
                valid=converted.valid,
            )
        else:
            sample = TimedCTBR(
                time.monotonic() if timestamp is None else float(timestamp),
                converted,
                received_timestamp=received,
            )
        with self._condition:
            self._messages.append(sample)
            self._condition.notify_all()

    def latest(self, timeout: float | None = None) -> TimedCTBR:
        with self._condition:
            if not self._messages and not self._condition.wait_for(
                lambda: bool(self._messages), timeout
            ):
                raise TimeoutError("no Agilicious command received")
            return self._messages[-1]

    def drain(self) -> list[TimedCTBR]:
        with self._condition:
            messages = list(self._messages)
            self._messages.clear()
            return messages

    def latest_after(self, timestamp: float, timeout: float | None = None) -> TimedCTBR:
        """Wait for a command newer than ``timestamp``; never reuse stale control."""

        with self._condition:
            predicate = lambda: bool(self._messages) and self._messages[-1].timestamp > timestamp
            if not predicate() and not self._condition.wait_for(predicate, timeout):
                raise TimeoutError("no fresh Agilicious command received")
            return self._messages[-1]


class AgiliciousExpertPolicy:
    """Synchronous collection policy around a ROS/in-process Agilicious bridge.

    ``state_publisher`` must publish simulator state and the active trajectory
    reference to the licensed controller.  Its command subscriber should call
    ``buffer.callback`` with timestamps in the same simulator clock domain.
    """

    def __init__(
        self,
        buffer: AgiliciousCommandBuffer,
        state_publisher: Callable[[dict[str, Any]], None],
        *,
        timeout: float = 0.1,
    ) -> None:
        self.buffer = buffer
        self.state_publisher = state_publisher
        self.timeout = float(timeout)
        self._last_timestamp = -np.inf

    def reset(self) -> None:
        self._last_timestamp = -np.inf
        self.buffer.drain()

    def __call__(self, observation: dict[str, Any]) -> ControllerCommand:
        self.state_publisher(observation)
        sample = self.buffer.latest_after(self._last_timestamp, self.timeout)
        self._last_timestamp = sample.timestamp
        command = sample.controller_command(observation)
        if not command.valid:
            raise RuntimeError("Agilicious returned an invalid controller command")
        return command
