"""Static physical-feasibility checks for tracks and MPCC racing references."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..env.gate_audit import DEFAULT_VEHICLE_HALF_EXTENTS
from ..env.tracks import Track
from .controller import MPCCController
from .racing_line import RacingLine


@dataclass(frozen=True, slots=True)
class RacingLineFeasibility:
    track: str
    feasible: bool
    issues: tuple[str, ...]
    length: float
    minimum_ideal_body_clearance: float
    minimum_world_bounds_clearance: float
    maximum_crossing_offset: float
    maximum_curvature: float
    minimum_speed: float
    maximum_speed: float
    maximum_lateral_acceleration: float
    maximum_longitudinal_acceleration: float
    maximum_braking_acceleration: float
    maximum_required_collective: float
    maximum_tangent_rate: float
    minimum_gate_tangent_alignment: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_racing_line_feasibility(
    track: Track,
    line: RacingLine,
    controller: MPCCController,
    *,
    vehicle_half_extents: tuple[float, float, float] = DEFAULT_VEHICLE_HALF_EXTENTS,
    minimum_body_clearance: float = 0.10,
    collective_reserve_fraction: float = 0.90,
    body_rate_reserve_fraction: float = 0.80,
) -> RacingLineFeasibility:
    """Check whether the planned centreline and speed profile retain control reserve."""

    half_extents = np.asarray(vehicle_half_extents, np.float64)
    crossing_clearances: list[float] = []
    alignments: list[float] = []
    for index, gate in enumerate(track.gates):
        frame = line.evaluate(line.gate_progress[index])
        crossing = np.asarray(frame["position"], np.float64)
        rotation = np.stack(
            [frame["tangent"], frame["lateral"], frame["up"]], axis=1
        )
        delta = crossing - gate.position
        lateral = float(delta @ gate.lateral)
        vertical = float(delta @ gate.up)
        lateral_extent = float(np.abs(gate.lateral @ rotation) @ half_extents)
        vertical_extent = float(np.abs(gate.up @ rotation) @ half_extents)
        crossing_clearances.append(
            min(
                0.5 * float(gate.size[0]) - abs(lateral) - lateral_extent,
                0.5 * float(gate.size[1]) - abs(vertical) - vertical_extent,
            )
        )
        alignments.append(float(np.asarray(frame["tangent"]) @ gate.normal))

    progress = np.asarray(controller._speed_progress, np.float64)
    speed = np.asarray(controller._speed_profile, np.float64)
    frame = line.evaluate(progress)
    curvature = np.asarray(frame["curvature"], np.float64)
    segment = np.diff(np.concatenate([progress, [line.length]]))
    following_speed = np.roll(speed, -1) if track.loop else np.concatenate([speed[1:], speed[-1:]])
    longitudinal = (following_speed**2 - speed**2) / np.maximum(2.0 * segment, 1e-9)
    lateral_acceleration = speed**2 * curvature
    required_collective = np.sqrt(
        controller.vehicle.gravity**2 + lateral_acceleration**2 + longitudinal**2
    )
    tangent_rate = speed * curvature

    radius = float(np.linalg.norm(half_extents))
    positions = np.asarray(line.position, np.float64)
    bounds_clearance = float(
        np.min(
            np.concatenate(
                [
                    positions - track.bounds[:, 0],
                    track.bounds[:, 1] - positions,
                ],
                axis=1,
            )
        )
        - radius
    )
    geometry = track.geometry_report()
    issues: list[str] = list(geometry["issues"])
    minimum_clearance = min(crossing_clearances)
    maximum_collective = float(np.max(required_collective))
    maximum_rate = float(np.max(tangent_rate))
    minimum_alignment = min(alignments)
    if minimum_clearance < minimum_body_clearance:
        issues.append(
            f"ideal body clearance {minimum_clearance:.3f} m < {minimum_body_clearance:.3f} m"
        )
    if bounds_clearance < 0.0:
        issues.append(f"vehicle reference exceeds world bounds by {-bounds_clearance:.3f} m")
    collective_limit = collective_reserve_fraction * controller.config.maximum_collective_thrust
    if maximum_collective > collective_limit:
        issues.append(
            f"required collective {maximum_collective:.3f} > reserve limit {collective_limit:.3f}"
        )
    rate_limit = body_rate_reserve_fraction * min(controller.vehicle.body_rate_max)
    if maximum_rate > rate_limit:
        issues.append(f"tangent rate {maximum_rate:.3f} > reserve limit {rate_limit:.3f}")
    if minimum_alignment < 0.95:
        issues.append(f"gate tangent alignment {minimum_alignment:.3f} < 0.950")

    return RacingLineFeasibility(
        track=track.name,
        feasible=not issues,
        issues=tuple(issues),
        length=line.length,
        minimum_ideal_body_clearance=minimum_clearance,
        minimum_world_bounds_clearance=bounds_clearance,
        maximum_crossing_offset=float(np.max(np.linalg.norm(line.crossing_offsets, axis=1))),
        maximum_curvature=float(np.max(curvature)),
        minimum_speed=float(np.min(speed)),
        maximum_speed=float(np.max(speed)),
        maximum_lateral_acceleration=float(np.max(lateral_acceleration)),
        maximum_longitudinal_acceleration=float(np.max(longitudinal)),
        maximum_braking_acceleration=float(np.max(-longitudinal)),
        maximum_required_collective=maximum_collective,
        maximum_tangent_rate=maximum_rate,
        minimum_gate_tangent_alignment=minimum_alignment,
    )
