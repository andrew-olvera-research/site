"""Periodic arc-length racing lines and gate-aperture optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid
import zipfile

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator
from scipy.optimize import minimize

from ..env.tracks import Track


F64 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RacingLinePlannerConfig:
    implementation_version: int = 4
    sample_count: int = 1600
    dense_samples_per_gate: int = 160
    # The optimizer controls the vehicle centre. Keep at least half a metre of
    # nominal body/tracking room in the standard 2.5 m apertures.
    aperture_fraction: float = 0.20
    aperture_margin: float = 0.22
    corridor_expansion_rate: float = 1.8
    corridor_lateral_max: float = 3.0
    corridor_vertical_max: float = 2.0
    world_bounds_margin: float = 0.35
    offset_iterations: int = 30
    curvature_weight: float = 1.0
    curvature_peak_weight: float = 0.12
    offset_weight: float = 0.025
    offset_smoothness_weight: float = 0.08
    cache_directory: str | None = None

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class Projection:
    progress: float
    position: F64
    tangent: F64
    lateral: F64
    up: F64
    contour_error: F64
    lag_error: float
    distance: float
    gate_index: int


class RacingLine:
    """Cubic spatial racing line sampled on true arc length."""

    def __init__(
        self,
        *,
        track_name: str,
        track_fingerprint: str,
        progress: ArrayLike,
        position: ArrayLike,
        gate_progress: ArrayLike,
        corridor_lateral: ArrayLike,
        corridor_vertical: ArrayLike,
        crossing_offsets: ArrayLike,
        planner_fingerprint: str,
        loop: bool = True,
    ) -> None:
        self.track_name = str(track_name)
        self.track_fingerprint = str(track_fingerprint)
        self.progress = np.asarray(progress, dtype=np.float64)
        self.position = np.asarray(position, dtype=np.float64)
        self.gate_progress = np.asarray(gate_progress, dtype=np.float64)
        self.corridor_lateral = np.asarray(corridor_lateral, dtype=np.float64)
        self.corridor_vertical = np.asarray(corridor_vertical, dtype=np.float64)
        self.crossing_offsets = np.asarray(crossing_offsets, dtype=np.float64)
        self.planner_fingerprint = str(planner_fingerprint)
        self.loop = bool(loop)
        if self.progress.ndim != 1 or len(self.progress) < 8:
            raise ValueError("racing line needs at least eight samples")
        if self.position.shape != (len(self.progress), 3):
            raise ValueError("position must have shape (samples, 3)")
        if not np.isclose(self.progress[0], 0.0) or np.any(np.diff(self.progress) <= 0):
            raise ValueError("progress samples must increase from zero")
        if self.loop and not np.allclose(self.position[0], self.position[-1], atol=1e-7):
            raise ValueError("periodic racing line must repeat its first position")
        if self.corridor_lateral.shape != self.progress.shape or (
            self.corridor_vertical.shape != self.progress.shape
        ):
            raise ValueError("corridor arrays must match progress")
        self.length = float(self.progress[-1])
        boundary = "periodic" if self.loop else "not-a-knot"
        self._position_spline = CubicSpline(
            self.progress, self.position, axis=0, bc_type=boundary
        )
        self._lateral_spline = CubicSpline(
            self.progress, self.corridor_lateral, bc_type=boundary
        )
        self._vertical_spline = CubicSpline(
            self.progress, self.corridor_vertical, bc_type=boundary
        )
        self._native_triplet = None
        self._native_projection = None
        digest = hashlib.sha256()
        digest.update(self.track_fingerprint.encode())
        digest.update(self.planner_fingerprint.encode())
        digest.update(self.position.tobytes())
        digest.update(self.crossing_offsets.tobytes())
        self.fingerprint = digest.hexdigest()

    def wrap(self, progress: ArrayLike) -> F64:
        value = np.asarray(progress, dtype=np.float64)
        if self.loop:
            return np.mod(value, self.length)
        return np.clip(value, 0.0, self.length)

    def evaluate(self, progress: ArrayLike) -> dict[str, F64]:
        query = self.wrap(progress)
        if self._native_triplet is not None:
            position, derivative, second = self._native_triplet(query)
        else:
            position = np.asarray(self._position_spline(query), dtype=np.float64)
            derivative = np.asarray(self._position_spline(query, 1), dtype=np.float64)
            second = np.asarray(self._position_spline(query, 2), dtype=np.float64)
        speed = np.linalg.norm(derivative, axis=-1, keepdims=True)
        tangent = derivative / np.maximum(speed, 1e-9)
        if tangent.ndim == 1:
            curvature_vector = (second - tangent * float(tangent @ second)) / max(
                float(speed.squeeze()) ** 2, 1e-9
            )
        else:
            projection = np.sum(tangent * second, axis=-1, keepdims=True)
            curvature_vector = (second - tangent * projection) / np.maximum(speed**2, 1e-9)
        curvature = np.linalg.norm(curvature_vector, axis=-1)
        world_up = np.zeros_like(tangent)
        world_up[..., 2] = 1.0
        lateral = np.cross(world_up, tangent)
        lateral_norm = np.linalg.norm(lateral, axis=-1, keepdims=True)
        fallback = np.zeros_like(lateral)
        fallback[..., 1] = 1.0
        lateral = np.where(lateral_norm > 1e-5, lateral / np.maximum(lateral_norm, 1e-9), fallback)
        up = np.cross(tangent, lateral)
        return {
            "position": position,
            "tangent": tangent,
            "lateral": lateral,
            "up": up,
            "curvature": curvature,
            "corridor_lateral": np.maximum(self._lateral_spline(query), 0.05),
            "corridor_vertical": np.maximum(self._vertical_spline(query), 0.05),
        }

    def gate_index_at(self, progress: float) -> int:
        query = float(self.wrap(progress))
        index = int(np.searchsorted(self.gate_progress, query, side="right") - 1)
        if self.loop:
            return index % len(self.gate_progress)
        return int(np.clip(index, 0, len(self.gate_progress) - 1))

    def project(
        self,
        position: ArrayLike,
        *,
        hint_progress: float | None = None,
        search_radius: float = 8.0,
    ) -> Projection:
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError("projection point must have shape (3,)")
        if self._native_projection is not None:
            progress = self._native_projection(point, hint_progress, search_radius)
        else:
            progress = self._project_progress_numpy(point, hint_progress, search_radius)
        frame = self.evaluate(progress)
        center = np.asarray(frame["position"])
        tangent = np.asarray(frame["tangent"])
        residual = point - center
        lag = float(residual @ tangent)
        contour = residual - lag * tangent
        return Projection(
            progress=progress,
            position=center,
            tangent=tangent,
            lateral=np.asarray(frame["lateral"]),
            up=np.asarray(frame["up"]),
            contour_error=contour,
            lag_error=lag,
            distance=float(np.linalg.norm(contour)),
            gate_index=self.gate_index_at(progress),
        )

    def _project_progress_numpy(self, point, hint_progress, search_radius):
        """Reference implementation retained for exact native regression tests."""
        sample_progress = self.progress[:-1]
        sample_position = self.position[:-1]
        if hint_progress is not None:
            hint = float(self.wrap(hint_progress))
            delta = np.abs(sample_progress - hint)
            if self.loop:
                delta = np.minimum(delta, self.length - delta)
            candidates = np.flatnonzero(delta <= max(search_radius, self.length / len(sample_progress)))
            if not len(candidates):
                candidates = np.arange(len(sample_progress))
        else:
            candidates = np.arange(len(sample_progress))
        distances = np.linalg.norm(sample_position[candidates] - point, axis=1)
        progress = float(sample_progress[candidates[int(np.argmin(distances))]])
        # Orthogonal Newton projection on the cubic spline, bounded in step size
        # to avoid jumping to another branch at crossings.
        spacing = self.length / max(len(sample_progress), 1)
        for _ in range(8):
            if self._native_triplet is not None:
                current, first, second = self._native_triplet(self.wrap(progress))
            else:
                current = np.asarray(self._position_spline(self.wrap(progress)), dtype=np.float64)
                first = np.asarray(self._position_spline(self.wrap(progress), 1), dtype=np.float64)
                second = np.asarray(self._position_spline(self.wrap(progress), 2), dtype=np.float64)
            residual = current - point
            denominator = float(first @ first + residual @ second)
            if abs(denominator) < 1e-9:
                break
            step = float(residual @ first / denominator)
            progress = float(self.wrap(progress - np.clip(step, -4.0 * spacing, 4.0 * spacing)))
            if abs(step) < 1e-6:
                break
        return progress

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # DAgger workers deliberately share this cache.  np.savez_compressed
        # writes the zip archive incrementally, so writing directly to the
        # destination allows another worker to observe a valid-looking path
        # before the central directory has been flushed.  Build beside the
        # destination and publish with an atomic rename instead.
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as handle:
                np.savez_compressed(
                    handle,
                    track_name=np.asarray(self.track_name),
                    track_fingerprint=np.asarray(self.track_fingerprint),
                    progress=self.progress,
                    position=self.position,
                    gate_progress=self.gate_progress,
                    corridor_lateral=self.corridor_lateral,
                    corridor_vertical=self.corridor_vertical,
                    crossing_offsets=self.crossing_offsets,
                    planner_fingerprint=np.asarray(self.planner_fingerprint),
                    loop=np.asarray(self.loop),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "RacingLine":
        with np.load(path, allow_pickle=False) as archive:
            return cls(**{key: archive[key].item() if archive[key].ndim == 0 else archive[key] for key in archive.files})


class RacingLinePlanner:
    def __init__(self, config: RacingLinePlannerConfig | None = None) -> None:
        self.config = config or RacingLinePlannerConfig()

    def _crossings(self, track: Track, offsets: F64) -> F64:
        return np.stack(
            [
                gate.position.astype(np.float64)
                + offsets[index, 0] * gate.lateral.astype(np.float64)
                + offsets[index, 1] * gate.up.astype(np.float64)
                for index, gate in enumerate(track.gates)
            ]
        )

    @staticmethod
    def _parameterized_spline(
        points: F64, track: Track
    ) -> tuple[F64, CubicHermiteSpline]:
        gate_derivatives = np.stack(
            [gate.normal.astype(np.float64) for gate in track.gates]
        )
        if track.loop:
            spline_points = np.vstack([points, points[0]])
            derivatives = np.vstack([gate_derivatives, gate_derivatives[0]])
        else:
            extension = 4.0
            spline_points = np.vstack(
                [
                    points[0] - extension * gate_derivatives[0],
                    points,
                    points[-1] + extension * gate_derivatives[-1],
                ]
            )
            derivatives = np.vstack(
                [gate_derivatives[0], gate_derivatives, gate_derivatives[-1]]
            )
        chord = np.linalg.norm(np.diff(spline_points, axis=0), axis=1)
        if np.any(chord < 0.2):
            raise ValueError("gate crossings must be separated by at least 0.2 m")
        parameter = np.concatenate([[0.0], np.cumsum(chord)])
        return parameter, CubicHermiteSpline(
            parameter, spline_points, derivatives, axis=0
        )

    def _objective(self, flat_offsets: F64, track: Track) -> float:
        offsets = flat_offsets.reshape(len(track.gates), 2)
        parameter, spline = self._parameterized_spline(
            self._crossings(track, offsets), track
        )
        query = np.linspace(0.0, parameter[-1], max(256, 32 * len(track.gates)), endpoint=False)
        first = spline(query, 1)
        second = spline(query, 2)
        speed = np.linalg.norm(first, axis=1)
        curvature = np.linalg.norm(np.cross(first, second), axis=1) / np.maximum(speed**3, 1e-9)
        smooth = np.roll(offsets, -1, axis=0) - 2.0 * offsets + np.roll(offsets, 1, axis=0)
        cfg = self.config
        return float(
            cfg.curvature_weight * np.mean(curvature**2)
            + cfg.curvature_peak_weight * np.quantile(curvature, 0.95) ** 2
            + cfg.offset_weight * np.mean(offsets**2)
            + cfg.offset_smoothness_weight * np.mean(smooth**2)
        )

    def _optimize_offsets(self, track: Track) -> F64:
        count = len(track.gates)
        if self.config.offset_iterations <= 0:
            return np.zeros((count, 2), dtype=np.float64)
        limits = np.stack(
            [self.config.aperture_fraction * gate.size.astype(np.float64) for gate in track.gates]
        )
        bounds = [(-float(limit), float(limit)) for pair in limits for limit in pair]
        result = minimize(
            self._objective,
            np.zeros(count * 2, dtype=np.float64),
            args=(track,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.config.offset_iterations, "ftol": 1e-10},
        )
        if not np.all(np.isfinite(result.x)):
            return np.zeros((count, 2), dtype=np.float64)
        return result.x.reshape(count, 2)

    def _corridor_envelope(
        self,
        track: Track,
        progress: F64,
        position: F64,
        gate_progress: F64,
        offsets: F64,
        length: float,
    ) -> tuple[F64, F64]:
        """Build a periodic free-space envelope around the racing line.

        Gate dimensions are only physical constraints close to a gate plane.
        Between gates the envelope expands linearly, while world-bound clipping
        keeps the resulting tube inside the simulated flight volume.  Taking
        the minimum cone from every gate makes the envelope continuous when
        adjacent gates have different aperture sizes.
        """

        cfg = self.config
        gate_lateral = np.asarray(
            [
                max(
                    0.05,
                    0.5 * float(gate.size[0])
                    - cfg.aperture_margin
                    - abs(float(offsets[index, 0])),
                )
                for index, gate in enumerate(track.gates)
            ],
            dtype=np.float64,
        )
        gate_vertical = np.asarray(
            [
                max(
                    0.05,
                    0.5 * float(gate.size[1])
                    - cfg.aperture_margin
                    - abs(float(offsets[index, 1])),
                )
                for index, gate in enumerate(track.gates)
            ],
            dtype=np.float64,
        )
        delta = np.abs(progress[:, None] - gate_progress[None, :])
        cyclic_distance = np.minimum(delta, length - delta) if track.loop else delta
        lateral = np.minimum(
            cfg.corridor_lateral_max,
            np.min(gate_lateral[None, :] + cfg.corridor_expansion_rate * cyclic_distance, axis=1),
        )
        vertical = np.minimum(
            cfg.corridor_vertical_max,
            np.min(gate_vertical[None, :] + cfg.corridor_expansion_rate * cyclic_distance, axis=1),
        )

        # Split the available axis-aligned world-bound clearance evenly between
        # lateral and vertical motion.  Then any corner of the rectangular tube
        # remains within the bounds by the triangle inequality.
        derivative = np.gradient(position, progress, axis=0, edge_order=2)
        tangent = derivative / np.maximum(np.linalg.norm(derivative, axis=1, keepdims=True), 1e-9)
        world_up = np.zeros_like(tangent)
        world_up[:, 2] = 1.0
        lateral_axis = np.cross(world_up, tangent)
        lateral_norm = np.linalg.norm(lateral_axis, axis=1, keepdims=True)
        fallback = np.zeros_like(lateral_axis)
        fallback[:, 1] = 1.0
        lateral_axis = np.where(
            lateral_norm > 1e-5,
            lateral_axis / np.maximum(lateral_norm, 1e-9),
            fallback,
        )
        vertical_axis = np.cross(tangent, lateral_axis)
        lower_room = position - np.asarray(track.bounds[:, 0], dtype=np.float64)
        upper_room = np.asarray(track.bounds[:, 1], dtype=np.float64) - position
        symmetric_room = np.maximum(np.minimum(lower_room, upper_room) - cfg.world_bounds_margin, 0.05)

        def direction_limit(direction: F64) -> F64:
            ratios = np.where(
                np.abs(direction) > 1e-6,
                0.5 * symmetric_room / np.maximum(np.abs(direction), 1e-9),
                np.inf,
            )
            return np.min(ratios, axis=1)

        lateral = np.maximum(np.minimum(lateral, direction_limit(lateral_axis)), 0.05)
        vertical = np.maximum(np.minimum(vertical, direction_limit(vertical_axis)), 0.05)
        if track.loop:
            lateral[-1] = lateral[0]
            vertical[-1] = vertical[0]
        return lateral, vertical

    def plan(self, track: Track, *, force_rebuild: bool = False) -> RacingLine:
        cache_path: Path | None = None
        if self.config.cache_directory:
            key = f"{track.fingerprint[:16]}-{self.config.fingerprint[:16]}.npz"
            cache_path = Path(self.config.cache_directory) / key
            if cache_path.exists() and not force_rebuild:
                try:
                    return RacingLine.load(cache_path)
                except (EOFError, OSError, ValueError, zipfile.BadZipFile):
                    # A cache produced by an interrupted/older non-atomic
                    # writer is disposable.  Do not unlink here: a concurrent
                    # worker may already have replaced it with a valid archive.
                    # Rebuilding below safely overwrites it atomically.
                    pass
        offsets = self._optimize_offsets(track)
        crossings = self._crossings(track, offsets)
        parameter, spline = self._parameterized_spline(crossings, track)
        dense_count = max(
            self.config.sample_count * 3,
            self.config.dense_samples_per_gate * len(track.gates),
        )
        dense_parameter = np.linspace(0.0, parameter[-1], dense_count + 1)
        dense_position = spline(dense_parameter)
        # Preserve the ordered route's altitude envelope.  Gate-normal Hermite
        # derivatives remain valuable in x/y, but are not safe in z around a
        # sparse over/under manoeuvre: they can create an uncommanded ground
        # intersection between two valid checkpoints.  A monotone piecewise
        # cubic keeps every segment inside its endpoint altitude range.
        lower = float(track.bounds[2, 0] + self.config.world_bounds_margin)
        upper = float(track.bounds[2, 1] - self.config.world_bounds_margin)
        if (
            float(np.min(dense_position[:, 2])) < lower
            or float(np.max(dense_position[:, 2])) > upper
        ):
            altitude_points = (
                np.vstack([crossings, crossings[0]])
                if track.loop
                else np.vstack([crossings[0], crossings, crossings[-1]])
            )
            dense_position[:, 2] = PchipInterpolator(
                parameter, altitude_points[:, 2], extrapolate=False
            )(dense_parameter)
            dense_position[:, 2] = np.clip(dense_position[:, 2], lower, upper)
        dense_distance = np.linalg.norm(np.diff(dense_position, axis=0), axis=1)
        dense_progress = np.concatenate([[0.0], np.cumsum(dense_distance)])
        length = float(dense_progress[-1])
        progress = np.linspace(0.0, length, self.config.sample_count + 1)
        position = np.stack(
            [np.interp(progress, dense_progress, dense_position[:, axis]) for axis in range(3)],
            axis=1,
        )
        if track.loop:
            position[-1] = position[0]
            gate_parameter = parameter[:-1]
        else:
            gate_parameter = parameter[1:-1]
        gate_progress = np.interp(gate_parameter, dense_parameter, dense_progress)
        corridor_lateral, corridor_vertical = self._corridor_envelope(
            track,
            progress,
            position,
            gate_progress,
            offsets,
            length,
        )
        line = RacingLine(
            track_name=track.name,
            track_fingerprint=track.fingerprint,
            progress=progress,
            position=position,
            gate_progress=gate_progress,
            corridor_lateral=corridor_lateral,
            corridor_vertical=corridor_vertical,
            crossing_offsets=offsets,
            planner_fingerprint=self.config.fingerprint,
            loop=track.loop,
        )
        if cache_path is not None:
            line.save(cache_path)
        return line
