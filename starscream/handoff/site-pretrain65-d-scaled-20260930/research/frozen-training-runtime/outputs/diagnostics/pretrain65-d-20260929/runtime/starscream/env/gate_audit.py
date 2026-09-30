"""Independent ordered gate-crossing audits for recorded flight trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .tracks import Gate, Track, quaternion_matrix


DEFAULT_VEHICLE_HALF_EXTENTS = (0.25, 0.25, 0.10)


@dataclass(frozen=True, slots=True)
class GateCrossingEvent:
    transition: int
    gate_index: int
    alpha: float
    lateral: float
    vertical: float
    center_clearance: float
    body_clearance: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GateTrajectoryAudit:
    transitions: int
    expected_passes: int
    recorded_passes: int
    replayed_passes: int
    start_gate_index: int
    end_gate_index: int
    label_mismatch_transitions: tuple[int, ...]
    recorded_events: tuple[GateCrossingEvent, ...]
    replayed_gate_indices: tuple[int, ...]
    minimum_center_clearance: float
    minimum_body_clearance: float

    @property
    def labels_consistent(self) -> bool:
        return not self.label_mismatch_transitions

    @property
    def complete(self) -> bool:
        return self.replayed_passes == self.expected_passes

    @property
    def center_clear(self) -> bool:
        return bool(self.recorded_events) and self.minimum_center_clearance >= 0.0

    @property
    def body_clear(self) -> bool:
        return bool(self.recorded_events) and self.minimum_body_clearance >= 0.0

    @property
    def qualified(self) -> bool:
        return self.labels_consistent and self.complete and self.center_clear and self.body_clear

    def as_dict(self) -> dict[str, Any]:
        return {
            "transitions": self.transitions,
            "expected_passes": self.expected_passes,
            "recorded_passes": self.recorded_passes,
            "replayed_passes": self.replayed_passes,
            "start_gate_index": self.start_gate_index,
            "end_gate_index": self.end_gate_index,
            "label_mismatch_transitions": list(self.label_mismatch_transitions),
            "recorded_events": [event.as_dict() for event in self.recorded_events],
            "replayed_gate_indices": list(self.replayed_gate_indices),
            "minimum_center_clearance": self.minimum_center_clearance,
            "minimum_body_clearance": self.minimum_body_clearance,
            "labels_consistent": self.labels_consistent,
            "complete": self.complete,
            "center_clear": self.center_clear,
            "body_clear": self.body_clear,
            "qualified": self.qualified,
        }


def _directed_plane_intersection(
    gate: Gate,
    previous_position: np.ndarray,
    current_position: np.ndarray,
) -> tuple[float, float, float] | None:
    previous_delta = previous_position - gate.position
    current_delta = current_position - gate.position
    previous_side = float(previous_delta @ gate.normal)
    current_side = float(current_delta @ gate.normal)
    if not previous_side < 0.0 <= current_side:
        return None
    denominator = previous_side - current_side
    if denominator == 0.0:
        return None
    alpha = previous_side / denominator
    intersection = previous_delta + alpha * (current_position - previous_position)
    return (
        float(alpha),
        float(intersection @ gate.lateral),
        float(intersection @ gate.up),
    )


def _interpolated_quaternion(previous: np.ndarray, current: np.ndarray, alpha: float) -> np.ndarray:
    current = current.copy()
    if float(previous @ current) < 0.0:
        current *= -1.0
    quaternion = (1.0 - alpha) * previous + alpha * current
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("trajectory contains a zero quaternion at a gate crossing")
    return quaternion / norm


def _oriented_box_clearance(
    gate: Gate,
    quaternion_wxyz: np.ndarray,
    half_extents: np.ndarray,
    lateral: float,
    vertical: float,
) -> float:
    world_from_body = quaternion_matrix(quaternion_wxyz).astype(np.float64)
    lateral_extent = float(np.abs(gate.lateral @ world_from_body) @ half_extents)
    vertical_extent = float(np.abs(gate.up @ world_from_body) @ half_extents)
    return min(
        0.5 * float(gate.size[0]) - abs(lateral) - lateral_extent,
        0.5 * float(gate.size[1]) - abs(vertical) - vertical_extent,
    )


def audit_gate_trajectory(
    track: Track,
    states: ArrayLike,
    recorded_passes: ArrayLike,
    *,
    start_gate_index: int = 0,
    expected_passes: int | None = None,
    vehicle_half_extents: ArrayLike = DEFAULT_VEHICLE_HALF_EXTENTS,
) -> GateTrajectoryAudit:
    """Replay ordered crossings from T+1 states and compare them with T saved labels.

    The replay uses the exact segment/plane intersection and the declared physical
    aperture, without ``GateTracker``'s runtime tolerance.  Body clearance uses
    the oriented 0.5 x 0.5 x 0.2 m Flightmare collision box by default.
    """

    trajectory = np.asarray(states, dtype=np.float64)
    labels = np.asarray(recorded_passes, dtype=np.bool_)
    half_extents = np.asarray(vehicle_half_extents, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[0] != len(labels) + 1 or trajectory.shape[1] < 7:
        raise ValueError("states must have shape (T+1, >=7) for T recorded pass labels")
    if not np.all(np.isfinite(trajectory[:, :7])):
        raise ValueError("trajectory positions and quaternions must be finite")
    if half_extents.shape != (3,) or np.any(half_extents < 0.0):
        raise ValueError("vehicle_half_extents must be a non-negative xyz vector")
    expected = len(track.gates) if expected_passes is None else int(expected_passes)
    if expected < 0:
        raise ValueError("expected_passes cannot be negative")

    replay_index = int(start_gate_index) % len(track.gates)
    recorded_index = replay_index
    replayed_indices: list[int] = []
    recorded_events: list[GateCrossingEvent] = []
    mismatches: list[int] = []

    for transition, recorded in enumerate(labels):
        previous_position = trajectory[transition, :3]
        current_position = trajectory[transition + 1, :3]

        replay_gate = track.gates[replay_index]
        replay_crossing = _directed_plane_intersection(
            replay_gate, previous_position, current_position
        )
        replay_valid = False
        if replay_crossing is not None:
            _, lateral, vertical = replay_crossing
            replay_valid = bool(
                abs(lateral) <= 0.5 * float(replay_gate.size[0])
                and abs(vertical) <= 0.5 * float(replay_gate.size[1])
            )
        if replay_valid:
            replayed_indices.append(replay_index)
            replay_index = (replay_index + 1) % len(track.gates)

        if bool(recorded) != replay_valid:
            mismatches.append(transition)

        if recorded:
            recorded_gate = track.gates[recorded_index]
            crossing = _directed_plane_intersection(
                recorded_gate, previous_position, current_position
            )
            if crossing is None:
                center_clearance = body_clearance = float("-inf")
                alpha = lateral = vertical = float("nan")
            else:
                alpha, lateral, vertical = crossing
                center_clearance = min(
                    0.5 * float(recorded_gate.size[0]) - abs(lateral),
                    0.5 * float(recorded_gate.size[1]) - abs(vertical),
                )
                quaternion = _interpolated_quaternion(
                    trajectory[transition, 3:7], trajectory[transition + 1, 3:7], alpha
                )
                # Invisible route checkpoints constrain the planner/policy
                # centreline but are not physical apertures around the airframe.
                # Applying the 0.5 m collision box to a 0.1 m over/flag proxy
                # made every otherwise valid special manoeuvre fail expert
                # qualification by construction.
                body_clearance = (
                    _oriented_box_clearance(
                        recorded_gate, quaternion, half_extents, lateral, vertical
                    )
                    if recorded_gate.kind == "gate" else float("inf")
                )
            recorded_events.append(
                GateCrossingEvent(
                    transition=transition,
                    gate_index=recorded_index,
                    alpha=alpha,
                    lateral=lateral,
                    vertical=vertical,
                    center_clearance=float(center_clearance),
                    body_clearance=float(body_clearance),
                )
            )
            recorded_index = (recorded_index + 1) % len(track.gates)

    center_clearances = [event.center_clearance for event in recorded_events]
    body_clearances = [event.body_clearance for event in recorded_events]
    return GateTrajectoryAudit(
        transitions=len(labels),
        expected_passes=expected,
        recorded_passes=int(np.sum(labels)),
        replayed_passes=len(replayed_indices),
        start_gate_index=int(start_gate_index) % len(track.gates),
        end_gate_index=replay_index,
        label_mismatch_transitions=tuple(mismatches),
        recorded_events=tuple(recorded_events),
        replayed_gate_indices=tuple(replayed_indices),
        minimum_center_clearance=(min(center_clearances) if center_clearances else float("-inf")),
        minimum_body_clearance=(min(body_clearances) if body_clearances else float("-inf")),
    )
