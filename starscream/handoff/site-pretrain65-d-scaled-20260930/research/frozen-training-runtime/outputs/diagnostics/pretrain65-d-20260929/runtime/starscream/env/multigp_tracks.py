"""Source-backed MultiGP UTT and Global Qualifier benchmark tracks.

The UTT field guides define physical obstacle layouts.  ``Track.gates`` is an
ordered route-checkpoint representation for the planner and policy.  Physical
flags and special over-gate manoeuvres are retained in metadata and represented
by invisible route checkpoints; they are never rendered as ordinary gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Sequence

import numpy as np

from .tracks import Gate, Track, forward_up_quaternion


STANDARD_GATE_M = 1.524
MEGA_GATE_M = 3.6576
WHOOP_GATE_M = 0.483
STANDARD_GATE_CENTER_Z_M = STANDARD_GATE_M * 0.5
FLAG_ROUTE_Z_M = 1.20
OVER_ROUTE_Z_M = 3.20
FLAG_PASS_RADIUS_M = 1.80

UTT_GUIDES = {
    1: "https://www.multigp.com/wp-content/uploads/2017/05/MultiGP-universal-time-trial-track1.pdf",
    2: "https://www.multigp.com/wp-content/uploads/2017/05/MultiGP-universal-time-trial-track-2-Tsnuami-001.pdf",
    3: "https://www.multigp.com/wp-content/uploads/2017/05/MultiGP-universal-time-trial-track-3-BesselRun-002.pdf",
    4: "https://www.multigp.com/wp-content/uploads/2017/05/MultiGP-universal-time-trial-track-4-high-voltage-001.pdf",
    5: "https://www.multigp.com/wp-content/uploads/2017/05/MultiGP-universal-time-trial-track-5-nautilus-manual-001.pdf",
    6: "https://www.multigp.com/wp-content/uploads/2017/05/MultiGP-universal-time-trial-track-6-fury.pdf",
    7: "https://www.multigp.com/wp-content/uploads/2017/05/MultiGP-universal-time-trial-track-7-manual.pdf",
    8: "https://www.multigp.com/wp-content/uploads/2019/01/MultiGP-universal-time-trial-8-manual.pdf",
    9: "https://www.multigp.com/wp-content/uploads/2019/10/MultiGP-universal-time-trial-track-9-megautt-manual001.pdf",
    10: "https://www.multigp.com/wp-content/uploads/2020/06/MultiGP-universal-time-trial-track-10-prairie-rage-manual1.pdf",
}
GQ_2026_SOURCE = "https://ereadrone.com/en/map/1928"
# Invisible flag/over-route checkpoints constrain which side of a physical
# obstacle the racing line uses; they are not 10 cm physical gates. A 1 m
# aperture keeps a five-inch vehicle safely inside the intended 1.8 m flag
# offset while avoiding an artificial +/-5 cm gate-tracker requirement that
# made otherwise correct UTT laps nearly impossible to complete.
ROUTE_CONSTRAINT_M = 1.00


@dataclass(frozen=True, slots=True)
class RoutePoint:
    name: str
    position: tuple[float, float, float]
    kind: str = "gate"
    size: tuple[float, float] = (STANDARD_GATE_M, STANDARD_GATE_M)
    render: bool = True
    normal: tuple[float, float, float] | None = None
    physical_obstacle: int | None = None


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1.0e-12)


def _route_track(
    name: str,
    route: Sequence[RoutePoint],
    *,
    metadata: dict[str, Any],
    margin_m: float = 8.0,
    maximum_z_m: float = 10.0,
) -> Track:
    points = np.asarray([item.position for item in route], np.float64)
    gates: list[Gate] = []
    for index, item in enumerate(route):
        if item.normal is None:
            incoming = _unit(points[index] - points[(index - 1) % len(points)])
            outgoing = _unit(points[(index + 1) % len(points)] - points[index])
            normal = incoming + outgoing
            if np.linalg.norm(normal) < 1.0e-6:
                normal = outgoing
        else:
            normal = np.asarray(item.normal, np.float64)
        gates.append(Gate(
            position=points[index].astype(np.float32),
            quaternion_wxyz=forward_up_quaternion(normal),
            size=np.asarray(item.size, np.float32),
            name=item.name,
            kind=item.kind,
            render=item.render,
        ))
    bounds = np.asarray([
        [float(points[:, 0].min() - margin_m), float(points[:, 0].max() + margin_m)],
        [float(points[:, 1].min() - margin_m), float(points[:, 1].max() + margin_m)],
        [0.0, max(maximum_z_m, float(points[:, 2].max() + 4.0))],
    ], np.float32)
    payload = dict(metadata)
    payload["route_checkpoint_count"] = len(route)
    payload["route_semantics"] = [
        {
            "index": index, "name": item.name, "kind": item.kind,
            "physical_obstacle": item.physical_obstacle,
        }
        for index, item in enumerate(route)
    ]
    return Track(name=name, gates=tuple(gates), bounds=bounds, loop=True, metadata=payload)


def _physical(
    labels: Iterable[tuple[str, str, tuple[float, float], float]],
) -> list[dict[str, Any]]:
    return [
        {"name": name, "kind": kind, "position_m": [x, y, z]}
        for name, kind, (x, y), z in labels
    ]


def _corner_flag_proxy(
    previous_xy: Sequence[float], flag_xy: Sequence[float], next_xy: Sequence[float],
    *, radius: float = FLAG_PASS_RADIUS_M,
) -> tuple[float, float, float]:
    """Place a route proxy on the interior angle bisector around a flag pole.

    MultiGP specifies the pole coordinate and traversal side, not one exact
    racing-line point.  This proxy preserves the obstacle coordinate while
    forcing a finite-radius pass instead of the invalid pole-center shortcut.
    """

    previous = np.asarray(previous_xy, np.float64)
    flag = np.asarray(flag_xy, np.float64)
    following = np.asarray(next_xy, np.float64)
    incoming = _unit(flag - previous)
    outgoing = _unit(following - flag)
    bisector = -incoming + outgoing
    if float(np.linalg.norm(bisector)) < 1.0e-6:
        bisector = np.asarray([-incoming[1], incoming[0]], np.float64)
    proxy = flag + float(radius) * _unit(bisector)
    return float(proxy[0]), float(proxy[1]), FLAG_ROUTE_Z_M


def _base_metadata(
    number: int, title: str, physical: list[dict[str, Any]], *,
    fidelity: str = "official_dimensioned_plan",
    five_inch: bool = True,
    comparison_eligible: bool = True,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    payload = {
        "series": "MultiGP Universal Time Trial",
        "utt_number": number,
        "title": title,
        "source": UTT_GUIDES[number],
        "source_authority": "MultiGP official field implementation guide",
        "geometry_fidelity": fidelity,
        "five_inch": five_inch,
        "dagger_eligible_5inch": five_inch and comparison_eligible,
        "leaderboard_comparison_eligible": comparison_eligible,
        "physical_obstacles": physical,
        "notes": list(notes),
    }
    encoded = json.dumps(physical, sort_keys=True, separators=(",", ":")).encode()
    payload["source_geometry_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return payload


def utt1() -> Track:
    xy = [(0, 0), (56, 0), (28, 14), (56, 28), (-14, 14)]
    normals = [(1, 0, 0), (1, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0)]
    physical = _physical((f"gate_{i+1}", "gate", p, STANDARD_GATE_CENTER_Z_M) for i, p in enumerate(xy))
    route = [RoutePoint(f"obstacle_{i+1}_gate", (*p, STANDARD_GATE_CENTER_Z_M), normal=normals[i], physical_obstacle=i + 1) for i, p in enumerate(xy)]
    return _route_track("multigp_utt01", route, metadata=_base_metadata(1, "UTT 1", physical))


def utt2() -> Track:
    xy = [(0, 0), (-28, 0), (0, 14), (0, 19), (28, 0)]
    normals = [(-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (1, 0, 0), (-1, 0, 0)]
    physical = _physical((f"gate_{i+1}", "gate", p, STANDARD_GATE_CENTER_Z_M) for i, p in enumerate(xy))
    z = STANDARD_GATE_CENTER_Z_M
    # Gates 3 and 4 are only 5 m apart and must be traversed in opposite
    # directions.  A centre-only spline has no information about which side of
    # the resulting 180-degree manoeuvre to use and produces a singular racing
    # line.  These invisible, wide checkpoints encode the legal west-side
    # teardrop shown by the official route arrows without moving or replacing
    # either physical gate.  They constrain the planner, not the benchmark
    # geometry, and therefore remain part of every augmentation.
    route = [
        RoutePoint("obstacle_1_gate", (0, 0, z), normal=normals[0], physical_obstacle=1),
        RoutePoint("obstacle_2_gate", (-28, 0, z), normal=normals[1], physical_obstacle=2),
        # Gate 3 is traversed westward although the preceding obstacle lies to
        # its west.  The official route therefore goes around its east side
        # before entering the gate; encode that approach instead of allowing a
        # direction-inconsistent centre-to-centre chord.
        RoutePoint(
            "route_gate_3_east_approach", (5.5, 7.0, 1.15),
            kind="maneuver_route", size=(4.0, 3.0), render=False,
            normal=(1.0, 0.5, 0.0), physical_obstacle=3,
        ),
        RoutePoint(
            "route_gate_3_east_apex", (8.0, 12.0, 1.35),
            kind="maneuver_route", size=(4.0, 3.0), render=False,
            normal=(0.0, 1.0, 0.0), physical_obstacle=3,
        ),
        RoutePoint(
            "route_gate_3_east_entry", (5.5, 14.0, 1.15),
            kind="maneuver_route", size=(4.0, 3.0), render=False,
            normal=(-1.0, 0.0, 0.0), physical_obstacle=3,
        ),
        RoutePoint("obstacle_3_gate", (0, 14, z), normal=normals[2], physical_obstacle=3),
        RoutePoint(
            "route_hairpin_west_entry", (-5.5, 14.0, 1.25),
            kind="maneuver_route", size=(4.0, 3.0), render=False,
            normal=(-1.0, 0.0, 0.0), physical_obstacle=3,
        ),
        RoutePoint(
            "route_hairpin_west_apex", (-7.5, 16.5, 1.45),
            kind="maneuver_route", size=(4.0, 3.0), render=False,
            normal=(0.0, 1.0, 0.0), physical_obstacle=3,
        ),
        RoutePoint(
            "route_hairpin_west_exit", (-5.5, 19.0, 1.25),
            kind="maneuver_route", size=(4.0, 3.0), render=False,
            normal=(1.0, 0.0, 0.0), physical_obstacle=4,
        ),
        RoutePoint("obstacle_4_gate", (0, 19, z), normal=normals[3], physical_obstacle=4),
        # Likewise Gate 5's westward arrow needs an east-side setup after
        # leaving Gate 4.  These points create a finite-radius turn before the
        # exact physical gate centre and direction.
        RoutePoint(
            "route_gate_5_east_high", (34.0, 8.0, 1.20),
            kind="maneuver_route", size=(5.0, 3.0), render=False,
            normal=(1.0, -0.5, 0.0), physical_obstacle=5,
        ),
        RoutePoint(
            "route_gate_5_east_apex", (36.0, 2.0, 1.35),
            kind="maneuver_route", size=(4.0, 3.0), render=False,
            normal=(0.0, -1.0, 0.0), physical_obstacle=5,
        ),
        RoutePoint(
            "route_gate_5_east_entry", (34.0, 0.0, 1.10),
            kind="maneuver_route", size=(4.0, 3.0), render=False,
            normal=(-1.0, 0.0, 0.0), physical_obstacle=5,
        ),
        RoutePoint("obstacle_5_gate", (28, 0, z), normal=normals[4], physical_obstacle=5),
    ]
    return _route_track("multigp_utt02_tsunami", route, metadata=_base_metadata(
        2, "Tsunami", physical,
        notes=(
            "Official direction-consistent approaches to Gates 3 and 5 and the "
            "west-side reversal between Gates 3 and 4 are encoded with invisible "
            "finite-radius planning checkpoints; physical gate coordinates and "
            "traversal directions remain unchanged.",
        ),
    ))


def utt3() -> Track:
    xy = [(28, 14), (56, 0), (28, 0), (7, 0), (0, 0)]
    physical = _physical((f"gate_{i+1}", "gate", p, STANDARD_GATE_CENTER_Z_M) for i, p in enumerate(xy))
    route = [RoutePoint(f"obstacle_{i+1}_gate", (*p, STANDARD_GATE_CENTER_Z_M), physical_obstacle=i + 1) for i, p in enumerate(xy)]
    return _route_track("multigp_utt03_bessel_run", route, metadata=_base_metadata(
        3, "Bessel Run", physical,
        notes=("Obstacle centers and field dimensions are official; planner normals follow the ordered route because the guide diagram does not numerically encode yaw.",),
    ))


def utt4() -> Track:
    xy = [(0, 0), (14, 14), (28, 28), (-14, 28), (42, 0)]
    physical = _physical((f"gate_{i+1}", "gate", p, STANDARD_GATE_CENTER_Z_M) for i, p in enumerate(xy))
    route = [
        RoutePoint("obstacle_1_gate", (0, 0, STANDARD_GATE_CENTER_Z_M), physical_obstacle=1),
        RoutePoint("obstacle_2_gate", (14, 14, STANDARD_GATE_CENTER_Z_M), physical_obstacle=2),
        RoutePoint("obstacle_3_gate", (28, 28, STANDARD_GATE_CENTER_Z_M), physical_obstacle=3),
        RoutePoint("obstacle_4_gate", (-14, 28, STANDARD_GATE_CENTER_Z_M), physical_obstacle=4),
        RoutePoint("route_over_gate_2", (14, 14, OVER_ROUTE_Z_M), kind="over_route", size=(ROUTE_CONSTRAINT_M, ROUTE_CONSTRAINT_M), render=False, physical_obstacle=2),
        RoutePoint("obstacle_5_gate", (42, 0, STANDARD_GATE_CENTER_Z_M), physical_obstacle=5),
    ]
    return _route_track("multigp_utt04_high_voltage", route, metadata=_base_metadata(
        4, "High Voltage", physical,
        notes=("The official required overflight of Gate 2 between obstacles 4 and 5 is an explicit route checkpoint.",),
    ))


def utt5() -> Track:
    xy = [(0, 0), (28, -28), (45.5, -10.5), (38.5, 0), (28, -7)]
    physical = _physical((f"gate_{i+1}", "gate", p, STANDARD_GATE_CENTER_Z_M) for i, p in enumerate(xy))
    route = [RoutePoint(f"obstacle_{i+1}_gate", (*p, STANDARD_GATE_CENTER_Z_M), physical_obstacle=i + 1) for i, p in enumerate(xy)]
    return _route_track("multigp_utt05_nautilus", route, metadata=_base_metadata(
        5, "Nautilus", physical,
        notes=("Metric centers are reconstructed from the official chained 28 m, 17.5 m, 10.5 m, and 7 m setup dimensions.",),
    ))


def utt6() -> Track:
    z = STANDARD_GATE_CENTER_Z_M
    physical = _physical([
        ("gate_1", "gate", (0, -14), z), ("gate_2", "gate", (0, 14), z),
        ("gate_3", "gate", (0, 0), z), ("gate_4", "gate", (-28, 0), z),
        ("gate_5", "gate", (0, 0), z), ("flag_left", "flag", (-1.5, 0), 0.0),
        ("flag_right", "flag", (1.5, 0), 0.0),
    ])
    route = [
        RoutePoint("obstacle_1_gate", (0, -14, z), normal=(0, -1, 0), physical_obstacle=1),
        RoutePoint("route_gate_1_reversal_south", (0, -18, 1.1), kind="maneuver_route", size=(3.0, 3.0), render=False, physical_obstacle=1),
        RoutePoint("route_gate_1_reversal_west", (-3.5, -18, 1.3), kind="maneuver_route", size=(3.0, 3.0), render=False, physical_obstacle=1),
        RoutePoint("route_gate_1_reversal_north", (-3.5, -14, 1.1), kind="maneuver_route", size=(3.0, 3.0), render=False, physical_obstacle=1),
        RoutePoint("route_between_flags_to_gate_2", (0, 0, 1.2), kind="maneuver_route", size=(1.0, 2.0), render=False, physical_obstacle=2),
        RoutePoint("obstacle_2_gate", (0, 14, z), normal=(0, 1, 0), physical_obstacle=2),
        RoutePoint("route_gate_2_reversal_north", (0, 18, 1.1), kind="maneuver_route", size=(3.0, 3.0), render=False, physical_obstacle=2),
        RoutePoint("route_gate_2_reversal_east", (3.5, 18, 1.3), kind="maneuver_route", size=(3.0, 3.0), render=False, physical_obstacle=2),
        RoutePoint("route_gate_2_reversal_south", (3.5, 14, 1.1), kind="maneuver_route", size=(3.0, 3.0), render=False, physical_obstacle=2),
        RoutePoint("obstacle_3_gate", (0, 0, z), normal=(0, -1, 0), physical_obstacle=3),
        # Finite-radius vertical half-loop: pass south through Gate 3, carry
        # momentum beyond it, climb, and return north above the gate/flags.
        RoutePoint("route_gate_3_loop_south_low", (0, -3.5, 1.2), kind="maneuver_route", size=(3.0, 3.0), render=False, physical_obstacle=3),
        RoutePoint("route_gate_3_loop_south_high", (0, -3.5, 3.5), kind="maneuver_route", size=(3.0, 3.0), render=False, physical_obstacle=3),
        RoutePoint("route_after_gate_3_between_flags", (0, 0, 3.6), kind="over_route", size=(2.0, 1.0), render=False, physical_obstacle=3),
        RoutePoint("obstacle_4_gate", (-28, 0, z), normal=(-1, 0, 0), physical_obstacle=4),
        # The return obstacle is an over-then-through split-S, not a vertical
        # line at one x/y coordinate.  The east-side apex gives the dive a
        # legal radius while retaining the exact overflight and gate centers.
        RoutePoint("route_gate_5_climb_west", (-4.0, 0, 2.0), kind="maneuver_route", size=(3.0, 3.0), render=False, physical_obstacle=5),
        RoutePoint("obstacle_5_over_between_flags", (0, 0, 3.6), kind="over_route", size=(2.0, 1.0), render=False, physical_obstacle=5),
        RoutePoint("route_gate_5_dive_east", (4.0, 0, 2.0), kind="maneuver_route", size=(3.0, 3.0), render=False, physical_obstacle=5),
        RoutePoint("obstacle_5_through_gate", (0, 0, z), physical_obstacle=5),
        RoutePoint("route_clear_flags_after_gate_5", (0, -4.0, 1.0), kind="maneuver_route", size=(1.5, 2.0), render=False, physical_obstacle=5),
    ]
    return _route_track("multigp_utt06_fury", route, metadata=_base_metadata(
        6, "Fury", physical,
        notes=("Both official over-gate/between-flags requirements use finite-radius invisible power-loop/split-S planning checkpoints; all physical coordinates are unchanged.",),
    ))


def utt7() -> Track:
    # The guide defines a whoop-scale layout and a chair obstacle.  It is kept
    # out of the five-inch plant and leaderboard aggregate by construction.
    xy = [(1.702, 0), (1.219, 0), (0, 0), (3.353, 0), (1.219, -0.9144)]
    yaws = [(0, -1, 0), (0, -1, 0), (1, 1, 0), (-1, 1, 0), (1, 1, 0)]
    physical = _physical([
        ("gate_1", "gate", xy[0], WHOOP_GATE_M * 0.5),
        ("gate_2", "gate", xy[1], WHOOP_GATE_M * 0.5),
        ("gate_3", "gate", xy[2], WHOOP_GATE_M * 0.5),
        ("gate_4", "gate", xy[3], WHOOP_GATE_M * 0.5),
        ("chair_gate_5", "chair", xy[4], 0.46),
    ])
    route = [RoutePoint(
        f"obstacle_{i+1}_{'chair' if i == 4 else 'gate'}", (*p, 0.46),
        size=(WHOOP_GATE_M, WHOOP_GATE_M), normal=yaws[i], physical_obstacle=i + 1,
    ) for i, p in enumerate(xy)]
    return _route_track("multigp_utt07_tiny_whutt", route, margin_m=1.5, maximum_z_m=2.5, metadata=_base_metadata(
        7, "Tiny WhUTT", physical, five_inch=False, comparison_eligible=False,
        notes=("Requires a separately identified 65 mm whoop plant; never scale this geometry into the five-inch suite.",),
    ))


def utt8() -> Track:
    z = STANDARD_GATE_CENTER_Z_M
    physical = _physical([
        ("gate_1", "gate", (0, 0), z), ("gate_2", "gate", (40, 8), z),
        ("flag_3", "flag", (40, 8), 0), ("flag_4", "flag", (33, 22), 0),
        ("flag_5", "flag", (33, 36), 0), ("gate_6", "gate", (21, 41), z),
        ("flag_7", "flag", (21, 41), 0), ("gate_8", "gate", (21, 22), z),
        ("gate_9", "gate", (16, 17), z), ("flag_10", "flag", (0, 22), 0),
    ])
    # The field guide co-mounts these flags with Gates 2 and 6 but does not
    # dimension the sub-meter pole-to-gate offset. Preserve that fact without
    # inventing a collision-cylinder coordinate.
    physical[2]["collision_audit"] = False
    physical[6]["collision_audit"] = False
    flag4_proxy = _corner_flag_proxy((40, 8), (33, 22), (33, 36))
    flag5_proxy = _corner_flag_proxy((33, 22), (33, 36), (21, 41))
    flag10_proxy = _corner_flag_proxy((16, 17), (0, 22), (0, 0))
    raw = [
        ("obstacle_1_gate", (0, 0, z), "gate", True, 1),
        ("obstacle_2_gate", (40, 8, z), "gate", True, 2),
        ("route_gate_2_power_loop_east_low", (43.5, 8, 1.2), "maneuver_route", False, 2),
        ("route_gate_2_power_loop_east_high", (43.5, 8, 3.2), "maneuver_route", False, 2),
        ("obstacle_3_flag_power_loop", (40, 8, 3.8), "flag_route", False, 3),
        ("obstacle_4_flag", flag4_proxy, "flag_route", False, 4),
        ("obstacle_5_flag", flag5_proxy, "flag_route", False, 5),
        ("obstacle_6_gate", (21, 41, z), "gate", True, 6),
        ("route_gate_6_corkscrew_south_low", (21, 44.5, 1.2), "maneuver_route", False, 6),
        ("route_gate_6_corkscrew_south_high", (21, 44.5, 3.2), "maneuver_route", False, 6),
        ("obstacle_7_flag_corkscrew", (21, 41, 3.8), "flag_route", False, 7),
        ("obstacle_8_gate", (21, 22, z), "gate", True, 8),
        ("obstacle_9_gate", (16, 17, z), "gate", True, 9),
        ("obstacle_10_flag", flag10_proxy, "flag_route", False, 10),
    ]
    route = [RoutePoint(
        n, p, kind=k, render=r,
        size=((3.0, 3.0) if k == "maneuver_route" else ((ROUTE_CONSTRAINT_M, ROUTE_CONSTRAINT_M) if k != "gate" else (STANDARD_GATE_M, STANDARD_GATE_M))),
        physical_obstacle=o,
    ) for n, p, k, r, o in raw]
    return _route_track("multigp_utt08_revenge", route, metadata=_base_metadata(
        8, "Revenge", physical,
        notes=("The official metric obstacle layout is exact; finite-radius power-loop/corkscrew checkpoints, flag flight height, and 1.8 m corner-bisector flag passes are documented controller route contracts because the 2D guide does not specify a unique racing line.",),
    ))


def utt9() -> Track:
    # The official setup sketch supplies string lengths but not enough angles
    # to determine a unique coordinate embedding.  Keep this useful topology
    # reconstruction out of exact leaderboard comparisons.
    ft = 0.3048
    entries = [
        ("mega_gate_1", "gate", (0, 0), MEGA_GATE_M * 0.5),
        ("flag_2", "flag", (-46 * ft, 46 * ft), 0),
        ("flag_3", "flag", (46 * ft, 69 * ft), 0),
        ("flag_4", "flag", (-138 * ft, 92 * ft), 0),
        ("mega_gate_5", "gate", (92 * ft, 129.5 * ft), MEGA_GATE_M * 0.5),
        ("flag_6", "flag", (138 * ft, 69 * ft), 0),
    ]
    physical = _physical(entries)
    route: list[RoutePoint] = []
    for i, (_, kind, xy, z) in enumerate(entries):
        position = (
            (xy[0], xy[1], z) if kind == "gate" else
            _corner_flag_proxy(entries[i - 1][2], xy, entries[(i + 1) % len(entries)][2])
        )
        route.append(RoutePoint(
            f"obstacle_{i+1}_{kind}", position,
            kind="gate" if kind == "gate" else "flag_route", render=kind == "gate",
            size=(MEGA_GATE_M, MEGA_GATE_M) if kind == "gate" else (ROUTE_CONSTRAINT_M, ROUTE_CONSTRAINT_M),
            physical_obstacle=i + 1,
        ))
    return _route_track("multigp_utt09_mega_reconstruction", route, margin_m=15, metadata=_base_metadata(
        9, "Mega UTT", physical, fidelity="official_dimensioned_topology_reconstruction",
        comparison_eligible=False,
        notes=("Excluded from exact timing claims: the released perspective/string-length diagram is underdetermined without setup angles or coordinates.",),
    ))


def utt10() -> Track:
    z = STANDARD_GATE_CENTER_Z_M
    physical = _physical([
        ("start_gate", "gate", (0, 0), z), ("flag_1", "flag", (24, 0), 0),
        ("flag_2", "flag", (14, 18), 0), ("flag_3", "flag", (24, 32), 0),
        ("gate_2", "gate", (0, 36), z), ("gate_3", "gate", (0, 28), z),
        ("flag_4", "flag", (0, 27), 0), ("flag_5", "flag", (3, 14), 0),
        ("gate_4", "gate", (6, 14), z), ("gate_5", "gate", (-8, 14), z),
    ])
    flag1_proxy = _corner_flag_proxy((0, 0), (24, 0), (14, 18))
    flag2_proxy = _corner_flag_proxy((24, 0), (14, 18), (24, 32))
    flag3_proxy = _corner_flag_proxy((14, 18), (24, 32), (0, 36))
    flag5_outbound_proxy = _corner_flag_proxy((0, 27), (3, 14), (6, 14))
    flag5_return_proxy = _corner_flag_proxy((6, 14), (3, 14), (-8, 14))
    raw = [
        ("obstacle_1_start_gate", (0, 0, z), "gate", True, 1),
        ("obstacle_2_flag_1", flag1_proxy, "flag_route", False, 2),
        ("obstacle_3_flag_2", flag2_proxy, "flag_route", False, 3),
        ("obstacle_4_flag_3", flag3_proxy, "flag_route", False, 4),
        ("obstacle_5_gate_2", (0, 36, z), "gate", True, 5),
        # Begin the westward avoidance before Gate 3.  Flag 4 is physically
        # only 1 m beyond this gate, so waiting until after a centre crossing
        # leaves no dynamically feasible lateral runout for a five-inch quad.
        ("route_gate_3_west_approach", (-1.5, 31.0, FLAG_ROUTE_Z_M), "flag_route", False, 6),
        ("obstacle_6_gate_3", (0, 28, z), "gate", True, 6),
        # The physical Flag 4 pole is only 1 m beyond Gate 3 on the centerline.
        # It is obstacle 8 later in the lap, but must already be avoided while
        # travelling from obstacle 6 to the timing gate at obstacle 7.
        ("route_avoid_flag_4_after_gate_3", (-2.5, 26.5, FLAG_ROUTE_Z_M), "flag_route", False, 7),
        ("obstacle_7_start_gate_reverse", (0, 0, z), "gate", False, 1),
        # Obstacle 7 crosses the timing gate southbound, after which the
        # published route returns north toward Flag 4.  The crossing point is
        # not itself a 180-degree turn: give the racing line a finite-radius
        # runout south/east of the timing gate before it heads north again.
        # These planner knots do not move or add a physical obstacle.
        ("route_7_runout_south", (0.0, -4.0, FLAG_ROUTE_Z_M), "flag_route", False, 1),
        ("route_7_turn_southeast", (2.828, -2.828, FLAG_ROUTE_Z_M), "flag_route", False, 1),
        ("route_7_turn_east", (4.0, 0.0, FLAG_ROUTE_Z_M), "flag_route", False, 1),
        ("route_7_turn_northeast", (2.828, 2.828, FLAG_ROUTE_Z_M), "flag_route", False, 1),
        # Flag 4 is a near-180-degree reversal.  A single offset waypoint lets
        # a cubic racing line cut back through the pole, so a sampled 2 m
        # semicircle encodes the legal east-side wrap without inventing an
        # obstacle or relying on an unconstrained spline between sparse knots.
        ("obstacle_8_flag_4_south", (0.0, 25.0, FLAG_ROUTE_Z_M), "flag_route", False, 7),
        ("obstacle_8_flag_4_southeast", (1.414, 25.586, FLAG_ROUTE_Z_M), "flag_route", False, 7),
        ("obstacle_8_flag_4_east", (2.0, 27.0, FLAG_ROUTE_Z_M), "flag_route", False, 7),
        ("obstacle_8_flag_4_northeast", (1.414, 28.414, FLAG_ROUTE_Z_M), "flag_route", False, 7),
        ("obstacle_8_flag_4_north", (0.0, 29.0, FLAG_ROUTE_Z_M), "flag_route", False, 7),
        ("obstacle_9_flag_5_outbound", flag5_outbound_proxy, "flag_route", False, 8),
        ("obstacle_10_gate_4", (6, 14, z), "gate", True, 9),
        ("obstacle_11_flag_5_return", flag5_return_proxy, "flag_route", False, 8),
        ("obstacle_12_gate_5", (-8, 14, z), "gate", True, 10),
    ]
    route = [RoutePoint(n, p, kind=k, render=r, size=((ROUTE_CONSTRAINT_M, ROUTE_CONSTRAINT_M) if k != "gate" else (STANDARD_GATE_M, STANDARD_GATE_M)), physical_obstacle=o) for n, p, k, r, o in raw]
    return _route_track("multigp_utt10_prairie_rage", route, metadata=_base_metadata(
        10, "Prairie Rage", physical,
        notes=("Coordinates and the numbered 12-obstacle route are copied from the official 1 m-grid diagram; the timing gate is repeated at obstacle 7, Flag 5 is repeated around Gate 4 at obstacles 9 and 11, and explicit invisible routing knots provide a finite-radius post-timing-gate runout and avoid the Flag-4 pole both after Gate 3 and during its near-180-degree obstacle-8 pass.",),
    ))


# Current downloadable EreaDrone checkpoint export, map 1928.  The container
# identifies itself as file Version 4 and TrackDataCurrentVersion 5.  Entries
# are (element id, model id, Unity x/y/z, Unity yaw quaternion y/w, direction).
_GQ2026 = (
    (22, 22, 29.505, 0.000, -40.233, 0.000, 1.000, "forward"),
    (39, 0, 38.032, 4.398, -24.083, 0.000, -1.000, "forward"),
    (38, 25, 29.504, 1.772, -24.083, 0.000, 1.000, "backward"),
    (25, 22, 29.504, 0.000, -24.083, 0.000, 1.000, "forward"),
    (37, 22, 29.501, 1.769, -7.323, 0.000, 1.000, "forward"),
    (40, 0, 35.200, 4.855, -7.323, 1.000, 0.000, "forward"),
    (35, 22, 29.501, 0.000, -7.323, 0.000, 1.000, "forward"),
    (41, 0, 22.952, 4.799, -7.323, 1.000, 0.000, "forward"),
    (36, 22, 26.756, 0.000, -7.323, 0.000, 1.000, "forward"),
    (42, 0, 1.724, 5.489, 4.807, 0.943, -0.333, "forward"),
    (33, 22, 17.626, 0.000, -9.605, 0.383, -0.924, "backward"),
    (43, 0, 15.677, 4.927, -11.571, 0.383, -0.924, "forward"),
    (32, 22, 13.957, 0.000, -13.264, 0.383, -0.924, "backward"),
    (48, 0, 25.858, 4.923, -23.998, 1.000, 0.004, "forward"),
    (44, 0, 18.383, 4.911, -31.598, 0.707, -0.707, "forward"),
    (45, 0, 9.231, 6.320, -17.361, 0.316, 0.949, "forward"),
    (27, 22, 20.051, 0.000, -24.086, 0.000, 1.000, "backward"),
    (24, 22, 18.383, 0.000, -29.413, 0.707, -0.707, "forward"),
    (46, 0, 0.320, 5.459, -40.233, 1.000, 0.000, "forward"),
    (47, 0, 24.383, 4.936, -54.385, 0.713, 0.701, "forward"),
    (21, 25, 24.473, 1.768, -45.863, 0.707, 0.707, "backward"),
    (20, 22, 24.473, 0.000, -45.878, 0.707, -0.707, "backward"),
)


def global_qualifier_2026() -> Track:
    physical: list[dict[str, Any]] = []
    route: list[RoutePoint] = []
    for index, (element, model, ux, uy, uz, qy, qw, direction) in enumerate(_GQ2026):
        kind = "flag_route" if model == 22 else "gate"
        altitude = FLAG_ROUTE_Z_M if kind == "flag_route" and uy < 0.1 else uy
        physical.append({
            "name": f"checkpoint_{index + 1:02d}_element_{element}",
            "kind": "route_proxy" if model == 22 else "gate",
            "position_m": [ux, -uz, uy],
            "ereadrone_model_id": model,
            "ereadrone_rotation_yw": [qy, qw],
            "direction": direction,
        })
        route.append(RoutePoint(
            f"checkpoint_{index + 1:02d}_element_{element}", (ux, -uz, altitude),
            kind=kind, render=kind == "gate",
            size=(2.2, 2.2) if kind == "gate" else (ROUTE_CONSTRAINT_M, ROUTE_CONSTRAINT_M),
            physical_obstacle=index + 1,
        ))
    metadata = {
        "series": "MultiGP Global Qualifier",
        "year": 2026,
        "source": GQ_2026_SOURCE,
        "source_authority": "MultiGP-published EreaDrone map",
        "source_map_id": 1928,
        "source_map_file_version": 4,
        "source_map_current_version": 5,
        "source_reported_track_length_m": 211.214691,
        "geometry_fidelity": "official_simulator_checkpoint_export_with_route_height_adaptation",
        "five_inch": True,
        "dagger_eligible_5inch": True,
        # The ordered checkpoint transforms are authoritative, but EreaDrone's
        # reported 211.2 m length disagrees with both their checkpoint chord
        # sum and Starscream's closed spline.  Keep this useful course in
        # training, but do not publish timing comparisons until the simulator
        # route-surface semantics are independently calibrated.
        "leaderboard_comparison_eligible": False,
        "physical_obstacles": physical,
        "notes": [
            "Checkpoint centers/order/direction metadata are direct from the official downloadable map.",
            "EreaDrone model-22 route proxies at ground height use a 1.2 m Starscream flight altitude.",
            "Timing comparison is disabled pending resolution of the source length/checkpoint discrepancy.",
        ],
    }
    encoded = json.dumps(physical, sort_keys=True, separators=(",", ":")).encode()
    metadata["source_geometry_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return _route_track("multigp_global_qualifier_2026", route, metadata=metadata, margin_m=10, maximum_z_m=12)


def canonical_multigp_tracks() -> tuple[Track, ...]:
    return (utt1(), utt2(), utt3(), utt4(), utt5(), utt6(), utt7(), utt8(), utt9(), utt10(), global_qualifier_2026())


def five_inch_dagger_sources() -> tuple[Track, ...]:
    return tuple(track for track in canonical_multigp_tracks() if bool((track.metadata or {}).get("dagger_eligible_5inch")))


def exact_timing_suite() -> tuple[Track, ...]:
    return tuple(track for track in canonical_multigp_tracks() if bool((track.metadata or {}).get("leaderboard_comparison_eligible")))


def augment_multigp_track(track: Track, *, seed: int, name: str) -> Track:
    """Apply survey/setup tolerance without changing route topology."""

    rng = np.random.default_rng(int(seed))
    yaw = float(rng.uniform(-np.pi, np.pi))
    rotation = np.asarray([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    rotation_3d = np.asarray(
        [[rotation[0, 0], rotation[0, 1], 0.0],
         [rotation[1, 0], rotation[1, 1], 0.0],
         [0.0, 0.0, 1.0]],
        np.float64,
    )
    scale = float(rng.uniform(0.985, 1.015))
    translation = rng.uniform(-1.0, 1.0, 2)
    points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
    center = points[:, :2].mean(0)
    points[:, :2] = (scale * (points[:, :2] - center)) @ rotation.T + center + translation
    source_metadata = dict(track.metadata or {})
    physical = [dict(item) for item in source_metadata.get("physical_obstacles", ())]
    route_semantics = list(source_metadata.get("route_semantics", ()))
    obstacle_xy_jitter = np.clip(
        rng.normal(0, 0.035, (len(physical), 2)), -0.08, 0.08,
    )
    obstacle_z_jitter = np.clip(
        rng.normal(0, 0.025, len(physical)), -0.05, 0.05,
    )
    for index, semantic in enumerate(route_semantics):
        obstacle = semantic.get("physical_obstacle")
        if obstacle is None:
            continue
        physical_index = int(obstacle) - 1
        points[index, :2] += obstacle_xy_jitter[physical_index]
        points[index, 2] += obstacle_z_jitter[physical_index]

    transformed_physical: list[dict[str, Any]] = []
    for index, item in enumerate(physical):
        transformed = dict(item)
        position = np.asarray(item["position_m"], np.float64)
        position[:2] = (
            scale * (position[:2] - center)
        ) @ rotation.T + center + translation + obstacle_xy_jitter[index]
        if str(item.get("kind")) != "flag":
            position[2] += obstacle_z_jitter[index]
        transformed["position_m"] = position.tolist()
        transformed_physical.append(transformed)
    gates: list[Gate] = []
    for index, gate in enumerate(track.gates):
        # A rigid course augmentation must preserve the source route arrows.
        # Re-inferring normals from neighbouring centres destroys deliberately
        # opposed gates such as Tsunami's 5 m reversal and creates a training /
        # exact-evaluation contract mismatch.
        normal = rotation_3d @ gate.normal
        up = rotation_3d @ gate.up
        aperture_scale = float(rng.uniform(0.98, 1.02)) if gate.kind == "gate" else 1.0
        gates.append(Gate(
            position=points[index].astype(np.float32),
            quaternion_wxyz=forward_up_quaternion(normal, up),
            size=(gate.size * aperture_scale).astype(np.float32), name=gate.name,
            kind=gate.kind, render=gate.render,
        ))
    bounds = track.bounds.copy()
    bounds[:2, 0] = np.minimum(bounds[:2, 0], points[:, :2].min(0) - 8)
    bounds[:2, 1] = np.maximum(bounds[:2, 1], points[:, :2].max(0) + 8)
    metadata = source_metadata
    metadata.update({
        "reference_name": track.name,
        "reference_track_fingerprint": track.fingerprint,
        "augmentation_seed": int(seed),
        "augmentation_contract": {
            "horizontal_scale": [0.985, 1.015], "survey_jitter_limit_m": 0.08,
            "vertical_jitter_limit_m": 0.05, "aperture_scale": [0.98, 1.02],
        },
        "scientific_scope": "reference-centered train augmentation; not zero-shot evidence",
        "physical_obstacles": transformed_physical,
    })
    return Track(name=name, gates=tuple(gates), bounds=bounds, loop=True, metadata=metadata)
