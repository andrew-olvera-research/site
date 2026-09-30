"""Physical track realizers for spline and maneuver-grammar ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline, splev, splprep

from ..tracks import Gate, Track, forward_up_quaternion
from .grammar import ManeuverGrammar
from .schema import (
    CourseProgram, GeneratorBackend, ManeuverKind, ManeuverSpec,
    RacingDistributionConfig,
)


K = ManeuverKind


@dataclass(frozen=True, slots=True)
class _Step:
    forward_m: float
    lateral_m: float
    vertical_m: float
    yaw_degrees: float
    roll_degrees: float
    label: str
    macro_index: int


def _severity(spec: ManeuverSpec, low: float, high: float) -> float:
    return float(low + spec.severity * (high - low))


def _motif(spec: ManeuverSpec, macro_index: int, rng: np.random.Generator) -> list[_Step]:
    """Expand one semantic macro into local route segments."""

    length = float(spec.length_scale)
    vertical = float(spec.vertical_scale)
    sign = -1.0 if spec.kind.value.endswith("right") else 1.0
    jitter = lambda scale: float(rng.uniform(-scale, scale))
    make: Callable[[float, float, float, float, float, str], _Step] = (
        lambda f, y, z, yaw, roll, label: _Step(
            f * length, y * length, z * vertical, yaw, roll,
            f"{spec.kind.value}:{label}", macro_index,
        )
    )
    if spec.kind == K.STRAIGHT:
        return [make(_severity(spec, 7.0, 13.0), jitter(0.25), jitter(0.15), jitter(5), 0, "run")]
    if spec.kind == K.ACCELERATION:
        base = _severity(spec, 7.0, 11.0)
        return [
            make(base, jitter(0.18), jitter(0.10), jitter(4), 0, "entry"),
            make(base * 1.05, jitter(0.18), jitter(0.10), jitter(4), 0, "exit"),
        ]
    if spec.kind == K.BRAKING:
        return [
            make(_severity(spec, 7.0, 11.0), 0, jitter(0.15), sign * _severity(spec, 8, 20), 0, "approach"),
            make(_severity(spec, 3.0, 5.5), sign * _severity(spec, 0.4, 1.3), jitter(0.10), sign * _severity(spec, 20, 45), 0, "compression"),
        ]
    if spec.kind in {K.TURN_LEFT, K.TURN_RIGHT}:
        return [make(
            _severity(spec, 4.0, 7.5), sign * _severity(spec, 0.4, 1.5),
            jitter(0.25), sign * _severity(spec, 28, 78), sign * _severity(spec, 0, 18), "arc",
        )]
    if spec.kind in {K.HAIRPIN_LEFT, K.HAIRPIN_RIGHT}:
        yaw = sign * _severity(spec, 55, 82)
        side = sign * _severity(spec, 0.8, 1.8)
        return [
            make(_severity(spec, 3.2, 5.0), side, jitter(0.20), yaw, sign * 18, "entry"),
            make(_severity(spec, 3.2, 5.0), side, jitter(0.20), yaw, sign * 28, "exit"),
        ]
    if spec.kind in {K.SLALOM_LEFT, K.SLALOM_RIGHT}:
        yaw = _severity(spec, 24, 48)
        side = _severity(spec, 0.7, 1.5)
        return [
            make(_severity(spec, 3.5, 5.8), sign * side, jitter(0.15), sign * yaw, sign * 15, "one"),
            make(_severity(spec, 3.5, 5.8), -sign * side, jitter(0.15), -2 * sign * yaw, -sign * 20, "two"),
            make(_severity(spec, 3.5, 5.8), sign * side, jitter(0.15), sign * yaw, sign * 15, "three"),
        ]
    if spec.kind == K.CLIMB:
        rise = _severity(spec, 1.1, 2.3)
        return [
            make(_severity(spec, 4.0, 6.5), jitter(0.4), rise, jitter(12), 0, "entry"),
            make(_severity(spec, 4.0, 6.5), jitter(0.4), rise, jitter(12), 0, "crest"),
        ]
    if spec.kind == K.DIVE:
        drop = _severity(spec, 1.0, 2.2)
        return [
            make(_severity(spec, 4.0, 6.5), jitter(0.4), -drop, jitter(12), 0, "entry"),
            make(_severity(spec, 4.0, 6.5), jitter(0.4), -drop, jitter(12), 0, "exit"),
        ]
    if spec.kind in {K.SPLIT_S_LEFT, K.SPLIT_S_RIGHT}:
        yaw = sign * _severity(spec, 38, 52)
        descent = _severity(spec, 0.9, 1.45)
        side = sign * _severity(spec, 0.7, 1.5)
        return [
            make(_severity(spec, 3.5, 5.0), side, 0.7 * descent, yaw, sign * 35, "entry"),
            make(_severity(spec, 2.8, 4.2), side, 0.15 * descent, yaw, sign * 95, "apex"),
            make(_severity(spec, 2.8, 4.2), side, -1.6 * descent, yaw, sign * 175, "descent"),
            make(_severity(spec, 3.5, 5.0), side, -0.8 * descent, yaw, sign * 180, "exit"),
        ]
    if spec.kind in {K.CORKSCREW_LEFT, K.CORKSCREW_RIGHT}:
        yaw = sign * _severity(spec, 35, 58)
        rise = _severity(spec, 0.8, 1.5)
        return [
            make(_severity(spec, 3.3, 5.0), sign * 0.8, rise, yaw, sign * 55, "entry"),
            make(_severity(spec, 3.0, 4.7), sign * 1.0, 0.25 * rise, yaw, sign * 125, "apex"),
            make(_severity(spec, 3.3, 5.0), sign * 0.8, -1.25 * rise, yaw, sign * 190, "exit"),
        ]
    if spec.kind == K.STACKED_REVERSAL:
        drop = _severity(spec, 2.6, 4.3)
        direction = -1.0 if rng.random() < 0.5 else 1.0
        return [
            make(_severity(spec, 4.0, 6.0), 0, 0.55 * drop, direction * 35, 0, "setup"),
            make(_severity(spec, 1.9, 2.8), direction * 0.25, -drop, direction * 145, direction * 170, "stack"),
            make(_severity(spec, 4.0, 6.0), direction * 0.8, 0.45 * drop, direction * 25, direction * 180, "exit"),
        ]
    raise ValueError(f"unsupported maneuver {spec.kind}")


def _rotate_xy(vector: np.ndarray, yaw_radians: float) -> np.ndarray:
    c, s = np.cos(yaw_radians), np.sin(yaw_radians)
    result = np.asarray(vector, np.float64).copy()
    result[:2] = np.asarray([[c, -s], [s, c]]) @ result[:2]
    return result


def _gate_frame(
    points: np.ndarray, index: int, roll_degrees: float,
    normal_yaw_degrees: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    incoming = points[index] - points[(index - 1) % len(points)]
    outgoing = points[(index + 1) % len(points)] - points[index]
    incoming /= max(float(np.linalg.norm(incoming)), 1.0e-12)
    outgoing /= max(float(np.linalg.norm(outgoing)), 1.0e-12)
    tangent = incoming + outgoing
    if np.linalg.norm(tangent) < 0.12:
        tangent = outgoing
    tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
    tangent = _rotate_xy(tangent, np.radians(normal_yaw_degrees))
    tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
    up = np.asarray([0.0, 0.0, 1.0], np.float64)
    up -= tangent * float(up @ tangent)
    if np.linalg.norm(up) < 1.0e-5:
        up = np.asarray([0.0, 1.0, 0.0])
        up -= tangent * float(up @ tangent)
    up /= np.linalg.norm(up)
    angle = np.radians(roll_degrees)
    up = (
        up * np.cos(angle) + np.cross(tangent, up) * np.sin(angle)
        + tangent * float(tangent @ up) * (1.0 - np.cos(angle))
    )
    return tangent, up


def _bounds(points: np.ndarray, cfg: RacingDistributionConfig) -> np.ndarray:
    center_xy = 0.5 * (points[:, :2].min(axis=0) + points[:, :2].max(axis=0))
    required_half = 0.5 * np.ptp(points[:, :2], axis=0) + cfg.bounds_margin_m
    if float(required_half.max()) > cfg.bounds_xy_m[1]:
        raise ValueError(
            f"course needs XY half-bound {required_half.max():.2f} m above contract"
        )
    half_xy = np.maximum(required_half, cfg.bounds_xy_m[0])
    lower = np.asarray([
        center_xy[0] - half_xy[0], center_xy[1] - half_xy[1], 0.0,
    ])
    upper = np.asarray([
        center_xy[0] + half_xy[0], center_xy[1] + half_xy[1],
        max(float(points[:, 2].max() + cfg.bounds_margin_m), cfg.maximum_altitude_m + 1.0),
    ])
    return np.stack([lower, upper], axis=1).astype(np.float32)


def _build_track(
    *, name: str, split: str, seed: int, backend: GeneratorBackend,
    points: np.ndarray, labels: list[str], rolls: np.ndarray,
    config: RacingDistributionConfig, metadata: dict[str, object],
    tangent_override: np.ndarray | None = None,
    upright_override: bool = False,
) -> Track:
    rng = np.random.default_rng(int(seed) ^ 0x51A7C0DE)
    width = float(rng.uniform(*config.gate_width_range_m))
    height = float(rng.uniform(*config.gate_height_range_m))
    gates: list[Gate] = []
    for index, point in enumerate(points):
        if tangent_override is None:
            normal, up = _gate_frame(
                points, index, float(rolls[index]),
                float(rng.uniform(
                    -config.gate_normal_jitter_degrees,
                    config.gate_normal_jitter_degrees,
                )),
            )
        else:
            normal = np.asarray(tangent_override[index], np.float64)
            normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
            up = (
                np.asarray([0.0, 0.0, 1.0], np.float64)
                if upright_override
                else _gate_frame(points, index, float(rolls[index]))[1]
            )
        size = np.asarray([width, height], np.float64) * rng.uniform(0.94, 1.06, 2)
        gates.append(Gate(
            position=point.astype(np.float32),
            quaternion_wxyz=forward_up_quaternion(normal, up),
            size=size.astype(np.float32), name=f"gate_{index:02d}_{labels[index]}",
        ))
    return Track(
        name=name, gates=tuple(gates), bounds=_bounds(points, config), loop=True,
        metadata={
            "source": "zero-shot-racing-distribution-v1",
            "scientific_scope": "generator-only-no-real-course-input",
            "backend": backend.value, "split": split, "seed": int(seed),
            "primitive_labels": list(labels), **metadata,
        },
    )


class RacingTaskGenerator:
    """Generate tracks without consulting benchmark or held-out geometry."""

    def __init__(self, config: RacingDistributionConfig | None = None) -> None:
        self.config = config or RacingDistributionConfig()
        self.grammar = ManeuverGrammar(self.config)

    def generate(
        self, *, backend: GeneratorBackend | str, seed: int, split: str,
        name: str, program: CourseProgram | None = None,
    ) -> Track:
        selected = GeneratorBackend(backend)
        if selected == GeneratorBackend.INFORMED_SPLINE:
            return self._generate_spline(seed=seed, split=split, name=name)
        selected_program = program or self.grammar.sample_program(seed=seed, split=split)
        if selected_program.backend != GeneratorBackend.MANEUVER_GRAMMAR:
            raise ValueError("grammar realization requires a grammar course program")
        return self._generate_grammar(selected_program, name=name)

    def _generate_grammar(self, program: CourseProgram, *, name: str) -> Track:
        cfg = self.config
        rng = np.random.default_rng(int(program.seed))
        steps = [
            step
            for macro_index, spec in enumerate(program.maneuvers)
            for step in _motif(spec, macro_index, rng)
        ]
        if not cfg.minimum_gates <= len(steps) <= cfg.maximum_gates:
            raise ValueError(
                f"expanded grammar program has {len(steps)} gates outside "
                f"[{cfg.minimum_gates}, {cfg.maximum_gates}]"
            )
        heading = float(rng.uniform(-np.pi, np.pi))
        raw_vectors: list[np.ndarray] = []
        for step in steps:
            half_yaw = np.radians(step.yaw_degrees) * 0.5
            heading += half_yaw
            vector = _rotate_xy(
                np.asarray([step.forward_m, step.lateral_m, step.vertical_m]),
                heading,
            )
            heading += half_yaw
            raw_vectors.append(vector)
        vectors = np.stack(raw_vectors)
        closure = vectors.mean(axis=0)
        correction_fraction = float(
            np.linalg.norm(closure) / max(float(np.median(np.linalg.norm(vectors, axis=1))), 1.0e-9)
        )
        if correction_fraction > cfg.maximum_closure_correction_fraction:
            raise ValueError(
                f"grammar closure correction {correction_fraction:.3f} exceeds contract"
            )
        vectors -= closure
        points = np.concatenate([
            np.zeros((1, 3), np.float64), np.cumsum(vectors[:-1], axis=0)
        ])
        # Preserve vertical shape while placing the lowest gate inside the
        # flight volume.  Horizontal scaling keeps the course in the target
        # length band without changing primitive order.
        raw_length = float(np.sum(np.linalg.norm(vectors, axis=1)))
        target_length = float(np.clip(
            raw_length * rng.uniform(0.88, 1.08),
            cfg.minimum_length_m + 2.0,
            cfg.maximum_length_m - 2.0,
        ))
        xy_scale = target_length / max(raw_length, 1.0e-9)
        points[:, :2] *= xy_scale
        points[:, 2] -= float(points[:, 2].min())
        excursion = float(np.ptp(points[:, 2]))
        allowed = cfg.maximum_altitude_m - cfg.minimum_altitude_m - 0.5
        if excursion > allowed:
            points[:, 2] *= allowed / excursion
        points[:, 2] += cfg.minimum_altitude_m + float(rng.uniform(0.15, 0.55))
        yaw = float(rng.uniform(-np.pi, np.pi))
        points = np.stack([_rotate_xy(point, yaw) for point in points])
        points[:, :2] += rng.uniform(-2.0, 2.0, 2)
        labels = [step.label for step in steps]
        rolls = np.asarray([step.roll_degrees for step in steps], np.float64)
        return _build_track(
            name=name, split=program.split, seed=program.seed,
            backend=GeneratorBackend.MANEUVER_GRAMMAR, points=points,
            labels=labels, rolls=rolls, config=cfg,
            metadata={
                "program": program.to_mapping(),
                "macro_labels": [item.kind.value for item in program.maneuvers],
                "expanded_macro_indices": [step.macro_index for step in steps],
                "closure_correction_fraction": correction_fraction,
                "global_yaw_radians": yaw,
            },
        )

    def _generate_spline(self, *, seed: int, split: str, name: str) -> Track:
        cfg = self.config
        rng = np.random.default_rng(int(seed))
        control_count = int(rng.integers(
            cfg.spline_control_points[0], cfg.spline_control_points[1] + 1
        ))
        sampled_half_bound: float | None = None
        if cfg.spline_control_sampling_mode == "radial_sorted":
            angles = np.linspace(0.0, 2.0 * np.pi, control_count, endpoint=False)
            angles += rng.uniform(-0.28, 0.28, control_count) * (
                2.0 * np.pi / control_count
            )
            angles = np.sort(np.mod(angles, 2.0 * np.pi))
            radius = rng.uniform(*cfg.spline_radial_range_m, control_count)
            # Preserve the historical informed generator exactly by default.
            radius = (
                0.25 * np.roll(radius, 1)
                + 0.5 * radius
                + 0.25 * np.roll(radius, -1)
            )
            amplitude = float(rng.uniform(*cfg.spline_vertical_amplitude_m))
            phase = float(rng.uniform(-np.pi, np.pi))
            harmonic = int(rng.integers(1, 4))
            z = (
                cfg.minimum_altitude_m + 0.8 + amplitude
                + amplitude * (
                    0.60 * np.sin(harmonic * angles + phase)
                    + 0.25 * np.sin((harmonic + 1) * angles - 0.7 * phase)
                )
            )
            z -= min(float(z.min() - cfg.minimum_altitude_m - 0.2), 0.0)
            z = np.clip(
                z,
                cfg.minimum_altitude_m + 0.15,
                cfg.maximum_altitude_m - 0.35,
            )
            controls = np.column_stack([
                radius * np.cos(angles), radius * np.sin(angles), z,
            ])
        else:
            if cfg.spline_fit_mode != "periodic_splprep":
                raise ValueError(
                    "bounded-uniform control points require periodic_splprep"
                )
            # Green et al. sample control points uniformly in a padded bounded
            # world, reject horizontal pair collisions, then recenter the set.
            # Rejection after recentering preserves the explicit boundary
            # contract rather than clipping and distorting the distribution.
            sampled_half_bound = float(rng.uniform(*cfg.bounds_xy_m))
            low = -sampled_half_bound + cfg.bounds_margin_m
            high = sampled_half_bound - cfg.bounds_margin_m
            if low >= high:
                raise ValueError("spline bounds leave no padded control-point area")
            minimum = float(cfg.spline_minimum_control_point_spacing_m)
            controls = np.empty((0, 3), np.float64)
            altitude_margin = 0.5 * float(cfg.gate_height_range_m[1])
            for _ in range(256):
                xy = rng.uniform(low, high, (control_count, 2))
                pairwise = np.linalg.norm(
                    xy[:, None, :] - xy[None, :, :], axis=-1
                )
                pairwise += np.eye(control_count) * (minimum + 1.0)
                if float(pairwise.min()) < minimum:
                    continue
                xy -= xy.mean(axis=0, keepdims=True)
                if np.any(xy < low) or np.any(xy > high):
                    continue
                z = rng.uniform(
                    cfg.minimum_altitude_m + altitude_margin,
                    cfg.maximum_altitude_m - altitude_margin,
                    control_count,
                )
                controls = np.column_stack([xy, z])
                if cfg.spline_control_order_mode == "azimuth_sorted":
                    # The paper defines a bounded point set but does not state
                    # how that set is ordered before splprep.  A cyclic angular
                    # order is the minimum deterministic completion of that
                    # underspecified contract: it preserves the uniformly
                    # sampled controls while avoiding a random Hamiltonian path
                    # whose self-crossings dominate rejection cost.
                    control_angle = np.arctan2(controls[:, 1], controls[:, 0])
                    controls = controls[np.argsort(control_angle, kind="stable")]
                break
            if not len(controls):
                raise ValueError(
                    "could not sample bounded, horizontally separated spline controls"
                )
            # splprep parameterizes points in their sampled order.
            angles = np.linspace(0.0, 2.0 * np.pi, control_count, endpoint=False)
        gate_count = int(rng.integers(cfg.minimum_gates, cfg.maximum_gates + 1))
        if cfg.spline_fit_mode == "periodic_cubic":
            parameter = np.concatenate([angles, [angles[0] + 2.0 * np.pi]])
            closed = np.concatenate([controls, controls[:1]], axis=0)
            spline = CubicSpline(parameter, closed, axis=0, bc_type="periodic")
            domain = (float(parameter[0]), float(parameter[-1]))
            evaluate = lambda values, derivative=0: np.asarray(
                spline(values, derivative), np.float64
            )
        else:
            closed_controls = np.concatenate([controls, controls[:1]], axis=0)
            knots, _ = splprep(
                closed_controls.T.copy(),
                s=float(cfg.spline_smoothing),
                per=True,
                k=min(3, len(closed_controls) - 1),
            )
            domain = (0.0, 1.0)
            evaluate = lambda values, derivative=0: np.stack(
                splev(values, knots, der=derivative), axis=-1
            ).astype(np.float64)

        if (
            cfg.spline_fit_mode == "periodic_cubic"
            and cfg.spline_arc_spacing_mode == "bounded_lognormal"
        ):
            # Preserve the historical generator bit-for-bit by default.
            allocation = np.clip(
                rng.lognormal(0.0, 0.35, gate_count), 0.58, 1.75
            )
            phases = np.concatenate([[0.0], np.cumsum(allocation[:-1])])
            phases = (
                (domain[1] - domain[0]) * phases / float(allocation.sum())
                + domain[0]
            )
        else:
            # Green et al. place gates at equal arc-length intervals.  Dense
            # inversion avoids assuming the B-spline parameter is arc length.
            dense_parameter = np.linspace(*domain, 8193)
            dense_points = evaluate(dense_parameter)
            cumulative = np.concatenate([[0.0], np.cumsum(
                np.linalg.norm(np.diff(dense_points, axis=0), axis=1)
            )])
            if cumulative[-1] <= 1.0e-9:
                raise ValueError("spline has zero arc length")
            allocation = (
                np.ones(gate_count, np.float64)
                if cfg.spline_arc_spacing_mode == "equal_arc"
                else np.clip(
                    rng.lognormal(0.0, 0.35, gate_count), 0.58, 1.75
                )
            )
            targets = np.concatenate([[0.0], np.cumsum(allocation[:-1])])
            targets *= cumulative[-1] / float(allocation.sum())
            phases = np.interp(targets, cumulative, dense_parameter)
        points = evaluate(phases)
        tangents = evaluate(phases, 1)
        if cfg.spline_gate_orientation_mode == "yaw_tangent":
            tangents[:, 2] = 0.0
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1.0e-12)
        yaw = float(rng.uniform(-np.pi, np.pi))
        points = np.stack([_rotate_xy(point, yaw) for point in points])
        tangents = np.stack([_rotate_xy(item, yaw) for item in tangents])
        local = np.roll(points, -1, axis=0) - points
        unit = local / np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1.0e-12)
        turn = np.degrees(np.arccos(np.clip(
            np.sum(unit * np.roll(unit, 1, axis=0), axis=1), -1.0, 1.0,
        )))
        incoming = np.roll(unit, 1, axis=0)
        signed_turn = np.degrees(np.arctan2(
            incoming[:, 0] * unit[:, 1] - incoming[:, 1] * unit[:, 0],
            np.sum(incoming[:, :2] * unit[:, :2], axis=1),
        ))
        dz = local[:, 2]
        labels: list[str] = []
        for angle, signed, elevation, length in zip(
            turn, signed_turn, dz, np.linalg.norm(local, axis=1)
        ):
            if elevation > 1.0:
                labels.append(K.CLIMB.value)
            elif elevation < -1.0:
                labels.append(K.DIVE.value)
            elif angle >= 80.0:
                labels.append(
                    K.HAIRPIN_LEFT.value if signed >= 0.0 else K.HAIRPIN_RIGHT.value
                )
            elif angle >= 48.0:
                labels.append(K.TURN_LEFT.value if signed >= 0.0 else K.TURN_RIGHT.value)
            elif length >= 10.0:
                labels.append(K.ACCELERATION.value)
            else:
                labels.append(K.STRAIGHT.value)
        rolls = rng.uniform(*cfg.gate_roll_range_degrees, gate_count)
        return _build_track(
            name=name, split=split, seed=seed,
            backend=GeneratorBackend.INFORMED_SPLINE, points=points,
            labels=labels, rolls=rolls, config=cfg,
            tangent_override=tangents,
            upright_override=cfg.spline_gate_orientation_mode == "yaw_tangent",
            metadata={
                "control_point_count": control_count,
                "spline_control_points": controls.tolist(),
                "spline_control_sampling_mode": cfg.spline_control_sampling_mode,
                "spline_control_order_mode": cfg.spline_control_order_mode,
                "spline_sampled_half_bound_m": sampled_half_bound,
                "spline_parameterization": (
                    f"{cfg.spline_fit_mode}-{cfg.spline_arc_spacing_mode}-v2"
                ),
                "spline_gate_orientation_mode": cfg.spline_gate_orientation_mode,
                "spline_smoothing": float(cfg.spline_smoothing),
                "global_yaw_radians": yaw,
            },
        )
