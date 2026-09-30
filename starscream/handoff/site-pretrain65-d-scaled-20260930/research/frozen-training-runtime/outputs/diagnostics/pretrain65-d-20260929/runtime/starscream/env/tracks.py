"""Racing-track geometry independent of perception and simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
import yaml


F32 = NDArray[np.float32]
TRACK_DIR = Path(__file__).resolve().parents[1] / "assets" / "tracks"
# Diagnostic escape hatch for exact cached/uncached regression comparisons.
GEOMETRY_CACHE_ENABLED = os.environ.get("STARSCREAM_GEOMETRY_CACHE", "1") != "0"


def _unit_quaternion(value: ArrayLike) -> F32:
    quaternion = np.asarray(value, dtype=np.float32)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must be finite wxyz with shape (4,)")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-7:
        raise ValueError("quaternion cannot be zero")
    return quaternion / norm


def quaternion_matrix(quaternion_wxyz: ArrayLike) -> F32:
    """Return a body-to-world rotation matrix for a wxyz quaternion."""

    w, x, y, z = _unit_quaternion(quaternion_wxyz)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def euler_degrees_quaternion(roll: float, pitch: float, yaw: float) -> F32:
    """Convert intrinsic XYZ roll/pitch/yaw degrees to wxyz."""

    r, p, y = np.radians([roll, pitch, yaw]) * 0.5
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return _unit_quaternion(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def matrix_quaternion(rotation: ArrayLike) -> F32:
    """Convert a proper 3x3 rotation matrix to a normalized wxyz quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([w, x, y, z], np.float32)
    if quaternion[0] < 0:
        quaternion *= -1
    return _unit_quaternion(quaternion)


def forward_up_quaternion(forward: ArrayLike, up_hint: ArrayLike = (0, 0, 1)) -> F32:
    """Build a gate/body frame whose +X follows ``forward`` and +Z follows the hint."""

    x_axis = np.asarray(forward, np.float64)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
    up = np.asarray(up_hint, np.float64)
    up = up - x_axis * float(up @ x_axis)
    if np.linalg.norm(up) < 1e-6:
        fallback = np.asarray([0.0, 1.0, 0.0])
        up = fallback - x_axis * float(fallback @ x_axis)
    z_axis = up / np.linalg.norm(up)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    return matrix_quaternion(np.stack([x_axis, y_axis, z_axis], axis=1))


@dataclass(frozen=True, slots=True)
class Gate:
    """A rectangular gate whose local +X axis is the forward plane normal."""

    position: F32
    quaternion_wxyz: F32
    size: F32
    name: str = "gate"
    enter_from_opposite_side: bool = False
    kind: str = "gate"
    render: bool = True
    _rotation_key: bytes | None = field(default=None, init=False, repr=False, compare=False)
    _rotation_value: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float32)
        size = np.asarray(self.size, dtype=np.float32)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("gate position must be a finite (3,) vector")
        if size.shape != (2,) or np.any(size <= 0):
            raise ValueError("gate size must be positive [width, height]")
        if self.kind not in {"gate", "flag_route", "over_route", "maneuver_route"}:
            raise ValueError(f"unsupported route checkpoint kind {self.kind!r}")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "quaternion_wxyz", _unit_quaternion(self.quaternion_wxyz))

    @property
    def rotation(self) -> F32:
        # Gate arrays are historically mutable despite frozen=True. Key by the
        # quaternion bytes, and return a copy so neither mutation path can make
        # the cached geometry stale or corrupt it. Preserve the exact arithmetic.
        if not GEOMETRY_CACHE_ENABLED:
            return quaternion_matrix(self.quaternion_wxyz)
        key = self.quaternion_wxyz.tobytes()
        if key != self._rotation_key:
            object.__setattr__(self, '_rotation_value', quaternion_matrix(self.quaternion_wxyz))
            object.__setattr__(self, '_rotation_key', key)
        return self._rotation_value.copy()

    @property
    def normal(self) -> F32:
        """Directed traversal normal; approach negative, then exit positive."""

        normal = self.rotation[:, 0]
        return -normal if self.enter_from_opposite_side else normal

    @property
    def physical_normal(self) -> F32:
        """Unmodified local +X normal of the rendered gate frame."""

        return self.rotation[:, 0]

    @property
    def lateral(self) -> F32:
        return self.rotation[:, 1]

    @property
    def up(self) -> F32:
        return self.rotation[:, 2]

    @property
    def directed_rotation(self) -> F32:
        """Right-handed gate-to-world frame whose +X is the traversal direction."""

        rotation = self.rotation.copy()
        if self.enter_from_opposite_side:
            rotation[:, 0:2] *= -1.0
        return rotation


@dataclass(frozen=True, slots=True)
class Track:
    name: str
    gates: tuple[Gate, ...]
    bounds: F32
    loop: bool = True
    metadata: dict[str, Any] | None = None
    _plan_key: Any = field(default=None, init=False, repr=False, compare=False)
    _plan_value: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        bounds = np.asarray(self.bounds, dtype=np.float32)
        if not self.gates:
            raise ValueError("a track needs at least one gate")
        if bounds.shape != (3, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("bounds must have shape (3, 2) with min < max")
        object.__setattr__(self, "bounds", bounds)

    def gate_indices(
        self, start: int, count: int, *, remaining: int | None = None,
    ) -> NDArray[np.int64]:
        if count < 1:
            raise ValueError("count must be positive")
        raw = np.arange(start, start + count, dtype=np.int64)
        indices = raw % len(self.gates) if self.loop else np.minimum(
            raw, len(self.gates) - 1
        )
        if remaining is not None:
            remaining = int(remaining)
            if remaining < 1:
                raise ValueError("remaining route gates must be positive")
            terminal_offset = min(remaining, count) - 1
            indices[terminal_offset + 1 :] = indices[terminal_offset]
        return indices

    def closest_gate_index(self, position: ArrayLike) -> int:
        point = np.asarray(position, dtype=np.float32)
        centers = np.stack([gate.position for gate in self.gates])
        return int(np.argmin(np.linalg.norm(centers - point, axis=1)))

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.name.encode("utf-8"))
        digest.update(np.asarray(self.bounds, np.float32).tobytes())
        digest.update(bytes([int(self.loop)]))
        for gate in self.gates:
            digest.update(gate.name.encode("utf-8"))
            digest.update(gate.position.tobytes())
            digest.update(gate.quaternion_wxyz.tobytes())
            digest.update(gate.size.tobytes())
            digest.update(bytes([int(gate.enter_from_opposite_side)]))
            # Preserve fingerprints of all pre-route-semantics tracks while
            # making non-physical route checkpoints unambiguous.
            if gate.kind != "gate" or not gate.render:
                digest.update(gate.kind.encode("utf-8"))
                digest.update(bytes([int(gate.render)]))
        return digest.hexdigest()

    def relative_gates(
        self,
        position_world: ArrayLike,
        quaternion_wxyz: ArrayLike,
        count: int,
        start_index: int | None = None,
        remaining: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return next-N gate references expressed in the vehicle body frame."""

        position = np.asarray(position_world, dtype=np.float32)
        if position.shape != (3,):
            raise ValueError("position_world must have shape (3,)")
        world_from_body = quaternion_matrix(quaternion_wxyz)
        body_from_world = world_from_body.T
        start = self.closest_gate_index(position) if start_index is None else int(start_index)
        indices = self.gate_indices(start, count, remaining=remaining)
        gates = [self.gates[index] for index in indices]
        relative_positions = np.stack(
            [body_from_world @ (gate.position - position) for gate in gates]
        ).astype(np.float32)
        normals = np.stack([body_from_world @ gate.normal for gate in gates]).astype(np.float32)
        up = np.stack([body_from_world @ gate.up for gate in gates]).astype(np.float32)
        return {
            "position": relative_positions,
            "normal": normals,
            "up": up,
            "size": np.stack([gate.size for gate in gates]).astype(np.float32),
            "distance": np.linalg.norm(relative_positions, axis=1).astype(np.float32),
            "index": indices,
            "mask": np.ones(count, dtype=np.bool_),
            "enter_from_opposite_side": np.asarray(
                [gate.enter_from_opposite_side for gate in gates], dtype=np.bool_
            ),
        }

    def flight_plan(
        self, start: int, count: int = 3, *, remaining: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Upcoming gates expressed in the active gate's directed frame.

        This depends only on the track and active-gate index, not vehicle pose, so
        it is safe to use as a deployable input to a learned state estimator.
        """

        if not GEOMETRY_CACHE_ENABLED:
            return self._flight_plan_uncached(start, count, remaining=remaining)
        # One-entry cache: bounded memory even with online generated courses.
        # Array mutation is supported by the historical geometry API, so include
        # all semantic inputs, not just object identity or the active index.
        key = (start, count, remaining, self.loop, tuple(
            (g.position.tobytes(), g.quaternion_wxyz.tobytes(), g.size.tobytes(),
             g.enter_from_opposite_side) for g in self.gates
        ))
        if key != self._plan_key:
            value = self._flight_plan_uncached(start, count, remaining=remaining)
            object.__setattr__(self, '_plan_value', value)
            object.__setattr__(self, '_plan_key', key)
        # Callers randomize plans in place. Never expose cache-owned arrays.
        return {name: array.copy() for name, array in self._plan_value.items()}

    def _flight_plan_uncached(
        self, start: int, count: int = 3, *, remaining: int | None = None,
    ) -> dict[str, np.ndarray]:
        indices = self.gate_indices(start, count, remaining=remaining)
        active = self.gates[int(indices[0])]
        active_from_world = active.directed_rotation.T
        gates = [self.gates[int(index)] for index in indices]
        position = np.stack(
            [active_from_world @ (gate.position - active.position) for gate in gates]
        ).astype(np.float32)
        normal = np.stack(
            [active_from_world @ gate.normal for gate in gates]
        ).astype(np.float32)
        up = np.stack([active_from_world @ gate.up for gate in gates]).astype(np.float32)
        opposite = np.asarray(
            [gate.enter_from_opposite_side for gate in gates], dtype=np.bool_
        )
        mask = np.ones(count, dtype=np.bool_)
        records = np.concatenate(
            [
                position,
                normal,
                up,
                np.stack([gate.size for gate in gates]).astype(np.float32),
                opposite[:, None].astype(np.float32),
                mask[:, None].astype(np.float32),
            ],
            axis=-1,
        )
        return {
            "records": records,
            "position": position,
            "normal": normal,
            "up": up,
            "size": np.stack([gate.size for gate in gates]).astype(np.float32),
            "enter_from_opposite_side": opposite,
            "mask": mask,
            "index": indices,
        }

    def geometry_report(self, *, minimum_alignment: float = 0.25) -> dict[str, Any]:
        """Return deterministic centerline feasibility diagnostics.

        A gate is considered directionally feasible when its traversal normal
        has a positive projection onto both the incoming and outgoing segments.
        Aperture corners must also remain inside the declared world bounds.
        """

        gates = self.gates
        records: list[dict[str, float | int | bool]] = []
        issues: list[str] = []
        for index, gate in enumerate(gates):
            previous = gates[(index - 1) % len(gates)] if self.loop or index else gate
            following = gates[(index + 1) % len(gates)] if self.loop or index + 1 < len(gates) else gate
            incoming = gate.position - previous.position
            outgoing = following.position - gate.position
            incoming_length = float(np.linalg.norm(incoming))
            outgoing_length = float(np.linalg.norm(outgoing))
            incoming_alignment = float(gate.normal @ (incoming / max(incoming_length, 1e-9)))
            outgoing_alignment = float(gate.normal @ (outgoing / max(outgoing_length, 1e-9)))
            corners = np.stack(
                [
                    gate.position + sy * 0.5 * gate.size[0] * gate.lateral + sz * 0.5 * gate.size[1] * gate.up
                    for sy, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                ]
            )
            inside_bounds = bool(
                np.all(corners >= self.bounds[:, 0]) and np.all(corners <= self.bounds[:, 1])
            )
            has_incoming = self.loop or index > 0
            has_outgoing = self.loop or index + 1 < len(gates)
            incoming_feasible = not has_incoming or (
                incoming_length > 0.5 and incoming_alignment >= minimum_alignment
            )
            outgoing_feasible = not has_outgoing or (
                outgoing_length > 0.5 and outgoing_alignment >= minimum_alignment
            )
            feasible = bool(incoming_feasible and outgoing_feasible and inside_bounds)
            if not feasible:
                issues.append(
                    f"gate {index}: incoming={incoming_alignment:.3f} "
                    f"outgoing={outgoing_alignment:.3f} bounds={inside_bounds}"
                )
            records.append(
                {
                    "index": index,
                    "incoming_length": incoming_length,
                    "outgoing_length": outgoing_length,
                    "incoming_alignment": incoming_alignment,
                    "outgoing_alignment": outgoing_alignment,
                    "inside_bounds": inside_bounds,
                    "feasible": feasible,
                }
            )
        return {"track": self.name, "feasible": not issues, "issues": issues, "gates": records}


class GateTracker:
    """Stateful ordered gate-passage tracker using oriented plane crossings."""

    def __init__(self, track: Track, start_index: int = 0, aperture_margin: float = 0.0):
        self.track = track
        self.index = int(start_index) % len(track.gates)
        self.aperture_margin = float(aperture_margin)
        self._previous_position: F32 | None = None
        self.passed_count = 0
        self.lap = 0

    def reset(self, position: ArrayLike | None = None, start_index: int = 0) -> None:
        self.index = int(start_index) % len(self.track.gates)
        self._previous_position = (
            None if position is None else np.asarray(position, dtype=np.float32).copy()
        )
        self.passed_count = 0
        self.lap = 0

    def update(self, position: ArrayLike) -> bool:
        current = np.asarray(position, dtype=np.float32)
        gate = self.track.gates[self.index]
        passed = False
        if self._previous_position is not None:
            previous_delta = self._previous_position - gate.position
            current_delta = current - gate.position
            previous_side = float(previous_delta @ gate.normal)
            current_side = float(current_delta @ gate.normal)
            intersection_alpha = previous_side / (previous_side - current_side + 1e-12)
            intersection = previous_delta + intersection_alpha * (current - self._previous_position)
            half_width = gate.size[0] * 0.5 + self.aperture_margin
            half_height = gate.size[1] * 0.5 + self.aperture_margin
            inside = (
                abs(float(intersection @ gate.lateral)) <= half_width
                and abs(float(intersection @ gate.up)) <= half_height
            )
            passed = bool(previous_side < 0.0 <= current_side and inside)
            if passed:
                previous_index = self.index
                self.index = (self.index + 1) % len(self.track.gates)
                self.passed_count += 1
                if self.track.loop and previous_index == len(self.track.gates) - 1:
                    self.lap += 1
        self._previous_position = current.copy()
        return passed

    def relative_gates(
        self, position: ArrayLike, quaternion_wxyz: ArrayLike, count: int,
        *, remaining: int | None = None,
    ):
        return self.track.relative_gates(
            position, quaternion_wxyz, count, self.index, remaining=remaining
        )


def load_track(name_or_path: str | Path) -> Track:
    path = Path(name_or_path)
    if not path.exists():
        path = TRACK_DIR / f"{path.stem}.yaml"
    if not path.exists():
        available = ", ".join(sorted(item.stem for item in TRACK_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"unknown track {name_or_path!r}; available: {available}")
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    gates = []
    default_size = data.get("gate_size", [2.5, 2.5])
    raw_gates = data["gates"]
    positions = [np.asarray(raw["position"], np.float32) for raw in raw_gates]
    auto_orient = bool(data.get("auto_orient", False))
    loop = bool(data.get("loop", True))
    for index, raw in enumerate(raw_gates):
        quaternion = raw.get("quaternion_wxyz")
        if quaternion is None:
            quaternion = euler_degrees_quaternion(*raw.get("rpy_degrees", [0, 0, 0]))
        if raw.get("orientation") == "auto" or (auto_orient and "orientation" not in raw):
            previous = positions[(index - 1) % len(positions)] if loop or index else positions[index]
            following = positions[(index + 1) % len(positions)] if loop or index + 1 < len(positions) else positions[index]
            incoming = positions[index] - previous
            outgoing = following - positions[index]
            incoming /= max(float(np.linalg.norm(incoming)), 1e-9)
            outgoing /= max(float(np.linalg.norm(outgoing)), 1e-9)
            tangent = incoming + outgoing
            if np.linalg.norm(tangent) < 1e-6:
                tangent = outgoing
            original_up = quaternion_matrix(quaternion)[:, 2]
            quaternion = forward_up_quaternion(tangent, raw.get("up_hint", original_up))
        gates.append(
            Gate(
                position=raw["position"],
                quaternion_wxyz=quaternion,
                size=raw.get("size", default_size),
                name=raw.get("name", f"gate_{index:02d}"),
                enter_from_opposite_side=bool(raw.get("enter_from_opposite_side", False)),
                kind=str(raw.get("kind", "gate")),
                render=bool(raw.get("render", True)),
            )
        )
    return Track(
        name=data.get("name", path.stem),
        gates=tuple(gates),
        bounds=np.asarray(data.get("bounds", [[-15, 15], [-15, 15], [0, 8]])),
        loop=loop,
        metadata=data.get("metadata", {}),
    )
