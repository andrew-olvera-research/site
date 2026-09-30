"""Constrained inverse design of course geometry through periodic splines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splev, splprep

from ..tracks import Gate, Track, matrix_quaternion
from .atlas import ManifoldAtlas
from .descriptors import TrackGeometryProfile, analyze_track_geometry
from .route_grammar import (
    RouteDirectedPosition,
    RouteGrammarProfile,
    route_directed_position,
    route_grammar_profile,
)


F64 = NDArray[np.float64]

# These coordinates have useful diagnostic meaning but no stable local
# derivative (gate count, thresholded fractions, sign changes, minima).  The
# first inverse-design prototype therefore optimizes only this smooth-enough
# subset and reports changes in every feature afterward.
CONTROLLABLE_FEATURE_NAMES = frozenset({
    "length_m",
    "gate_density_per_100m",
    "maximum_gate_spacing_m",
    "spacing_coefficient_of_variation",
    "vertical_excursion_m",
    "vertical_travel_fraction",
    "p95_absolute_slope",
    "p95_turn_degrees",
    "maximum_turn_degrees",
    "p99_curvature_m_inv",
    "maximum_curvature_m_inv",
    "curve_p95_curvature_m_inv",
    "curve_p95_absolute_torsion_m_inv",
    "p95_absolute_torsion_proxy",
    "mean_gate_normal_mismatch_degrees",
    "maximum_gate_up_change_degrees",
})


@dataclass(frozen=True, slots=True)
class ManifoldExtensionProposal:
    track: Track
    profile: TrackGeometryProfile
    parameters: F64
    requested_standardized_delta: F64
    achieved_standardized_delta: F64
    alignment: float
    residual: float
    line_search_scale: float
    valid: bool
    reasons: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "track": self.track.name,
            "fingerprint": self.track.fingerprint,
            "requested_standardized_delta": self.requested_standardized_delta.tolist(),
            "achieved_standardized_delta": self.achieved_standardized_delta.tolist(),
            "alignment": self.alignment,
            "residual": self.residual,
            "line_search_scale": self.line_search_scale,
            "valid": self.valid,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class RouteExtensionProposal:
    """Local spline step optimized in the actor's rolling-route token space."""

    track: Track
    profile: RouteGrammarProfile
    position: RouteDirectedPosition
    parameters: F64
    requested_target_fraction: float
    achieved_local_target_gain_fraction: float
    residual: float
    line_search_scale: float
    ridge: float
    tangent_reachable_fraction: float
    tangent_residual_fraction: float
    valid: bool
    reasons: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "track": self.track.name,
            "fingerprint": self.track.fingerprint,
            "position": self.position.to_mapping(),
            "requested_target_fraction": self.requested_target_fraction,
            "achieved_local_target_gain_fraction": (
                self.achieved_local_target_gain_fraction
            ),
            "residual": self.residual,
            "line_search_scale": self.line_search_scale,
            "ridge": self.ridge,
            "tangent_reachable_fraction": self.tangent_reachable_fraction,
            "tangent_residual_fraction": self.tangent_residual_fraction,
            "valid": self.valid,
            "reasons": list(self.reasons),
        }


class SplineManifoldGenerator:
    """Locally deform a qualified course toward a requested manifold ray.

    Gate centres are spline interpolation knots.  Optimization variables are
    smooth local tangent/lateral/up displacements at those knots.  A numerical
    descriptor Jacobian supplies the inverse map, and a ridge solve plus line
    search keeps each proposal local and auditable.
    """

    def __init__(
        self,
        base_track: Track,
        *,
        smooth_passes: int = 1,
        minimum_gate_spacing_m: float = 1.5,
        minimum_nonadjacent_spacing_m: float = 0.9,
        maximum_gate_displacement_m: float = 2.5,
        bounds_margin_m: float = 4.0,
    ) -> None:
        if not base_track.loop:
            raise ValueError("the initial manifold generator supports loop tracks only")
        if len(base_track.gates) < 4:
            raise ValueError("periodic spline deformation needs at least four gates")
        self.base_track = base_track
        self.smooth_passes = int(smooth_passes)
        self.minimum_gate_spacing_m = float(minimum_gate_spacing_m)
        self.minimum_nonadjacent_spacing_m = float(minimum_nonadjacent_spacing_m)
        self.maximum_gate_displacement_m = float(maximum_gate_displacement_m)
        self.bounds_margin_m = float(bounds_margin_m)
        self._base_points = np.stack(
            [gate.position for gate in base_track.gates]
        ).astype(np.float64)
        self._frames = self._local_frames(self._base_points)
        self._base_spline_tangents = self._spline_tangents(self._base_points)
        self._jacobian_cache: dict[
            tuple[float, int], tuple[TrackGeometryProfile, F64]
        ] = {}

    @staticmethod
    def _local_frames(points: F64) -> F64:
        tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.0e-12)
        world_up = np.zeros_like(tangent)
        world_up[:, 2] = 1.0
        lateral = np.cross(world_up, tangent)
        small = np.linalg.norm(lateral, axis=1) < 1.0e-6
        lateral[small] = np.asarray([0.0, 1.0, 0.0])
        lateral /= np.maximum(np.linalg.norm(lateral, axis=1, keepdims=True), 1.0e-12)
        up = np.cross(tangent, lateral)
        up /= np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1.0e-12)
        return np.stack([tangent, lateral, up], axis=-1)

    def _smooth(self, parameters: F64) -> F64:
        values = parameters.reshape(len(self.base_track.gates), 3).copy()
        for _ in range(self.smooth_passes):
            values = 0.25 * np.roll(values, 1, axis=0) + 0.5 * values + 0.25 * np.roll(values, -1, axis=0)
        return values

    def _positions(self, parameters: F64) -> F64:
        local = self._smooth(parameters)
        displacement = np.einsum("nij,nj->ni", self._frames, local)
        displacement -= displacement.mean(axis=0, keepdims=True)
        return self._base_points + displacement

    @staticmethod
    def _spline_tangents(points: F64) -> F64:
        closed = np.concatenate([points, points[:1]], axis=0)
        tck, phase = splprep(
            closed.T.copy(), s=0.0, per=True, k=min(3, len(points) - 1)
        )
        tangent = np.stack(splev(phase[:-1], tck, der=1), axis=-1)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.0e-12)
        return tangent

    @staticmethod
    def _align_vectors(source: F64, target: F64) -> F64:
        """Return the minimum proper rotation taking source onto target."""

        source = source / max(float(np.linalg.norm(source)), 1.0e-12)
        target = target / max(float(np.linalg.norm(target)), 1.0e-12)
        cross = np.cross(source, target)
        cosine = float(np.clip(source @ target, -1.0, 1.0))
        sine = float(np.linalg.norm(cross))
        if sine < 1.0e-10:
            if cosine > 0.0:
                return np.eye(3)
            axis = np.cross(source, np.asarray([1.0, 0.0, 0.0]))
            if np.linalg.norm(axis) < 1.0e-6:
                axis = np.cross(source, np.asarray([0.0, 1.0, 0.0]))
            axis /= np.linalg.norm(axis)
            return 2.0 * np.outer(axis, axis) - np.eye(3)
        skew = np.asarray([
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ])
        return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))

    def materialize(self, parameters: F64, *, name: str | None = None) -> Track:
        values = np.asarray(parameters, np.float64)
        expected = (len(self.base_track.gates), 3)
        if values.shape not in {expected, (expected[0] * expected[1],)}:
            raise ValueError(f"parameters must have shape {expected} or {(expected[0] * 3,)}")
        values = values.reshape(expected)
        points = self._positions(values)
        tangents = self._spline_tangents(points)
        gates: list[Gate] = []
        for gate, position, source_tangent, tangent in zip(
            self.base_track.gates, points, self._base_spline_tangents, tangents
        ):
            transported = self._align_vectors(source_tangent, tangent) @ gate.rotation
            gates.append(replace(
                gate,
                position=position.astype(np.float32),
                quaternion_wxyz=matrix_quaternion(transported),
            ))
        margin = self.bounds_margin_m
        bounds = np.stack([
            points.min(axis=0) - margin,
            points.max(axis=0) + margin,
        ], axis=1).astype(np.float32)
        bounds[2, 0] = min(bounds[2, 0], 0.0)
        metadata = dict(self.base_track.metadata or {})
        metadata.update({
            "generator": "spline-manifold-local-inverse-v1",
            "source_track": self.base_track.name,
            "source_fingerprint": self.base_track.fingerprint,
            "local_spline_parameters": values.tolist(),
        })
        return Track(
            name=name or f"{self.base_track.name}_manifold_extension",
            gates=tuple(gates),
            bounds=bounds,
            loop=True,
            metadata=metadata,
        )

    def validate(self, track: Track, parameters: F64) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
        adjacent = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        if float(adjacent.min()) < self.minimum_gate_spacing_m:
            reasons.append("adjacent-gate-spacing")
        distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        count = len(points)
        mask = np.ones((count, count), dtype=np.bool_)
        np.fill_diagonal(mask, False)
        for index in range(count):
            mask[index, (index + 1) % count] = False
            mask[(index + 1) % count, index] = False
        if np.any(mask) and float(distance[mask].min()) < self.minimum_nonadjacent_spacing_m:
            reasons.append("nonadjacent-gate-spacing")
        displacement = np.linalg.norm(points - self._base_points, axis=1)
        if float(displacement.max()) > self.maximum_gate_displacement_m + 1.0e-6:
            reasons.append("maximum-gate-displacement")
        geometry = track.geometry_report(minimum_alignment=-1.0)
        source_geometry = self.base_track.geometry_report(minimum_alignment=-1.0)
        for source_gate, candidate_gate in zip(
            source_geometry["gates"], geometry["gates"]
        ):
            if not bool(candidate_gate["inside_bounds"]):
                reasons.append("gate-aperture-outside-bounds")
                break
            # Several real tracks intentionally use a gate plane that is
            # nearly orthogonal to one adjacent centreline segment.  Reject a
            # deformation only when it materially worsens the source contract,
            # rather than declaring the qualified source itself invalid.
            if (
                float(candidate_gate["incoming_alignment"])
                < float(source_gate["incoming_alignment"]) - 0.15
                or float(candidate_gate["outgoing_alignment"])
                < float(source_gate["outgoing_alignment"]) - 0.15
            ):
                reasons.append("gate-orientation-degradation")
                break
        if not np.all(np.isfinite(parameters)):
            reasons.append("nonfinite-parameters")
        return not reasons, tuple(reasons)

    def descriptor_jacobian(
        self,
        *,
        epsilon_m: float = 0.08,
        phase_samples: int = 64,
    ) -> tuple[TrackGeometryProfile, F64]:
        """Central finite-difference map from spline parameters to descriptors."""

        cache_key = (float(epsilon_m), int(phase_samples))
        cached = self._jacobian_cache.get(cache_key)
        if cached is not None:
            profile, jacobian = cached
            return profile, jacobian.copy()
        count = len(self.base_track.gates) * 3
        zero = np.zeros(count, np.float64)
        base = analyze_track_geometry(self.materialize(zero), phase_samples=phase_samples)
        jacobian = np.empty((len(base.feature_vector), count), np.float64)
        for index in range(count):
            positive = zero.copy()
            negative = zero.copy()
            positive[index] = epsilon_m
            negative[index] = -epsilon_m
            high = analyze_track_geometry(
                self.materialize(positive), phase_samples=phase_samples
            ).feature_vector
            low = analyze_track_geometry(
                self.materialize(negative), phase_samples=phase_samples
            ).feature_vector
            jacobian[:, index] = (high - low) / (2.0 * epsilon_m)
        self._jacobian_cache[cache_key] = (base, jacobian.copy())
        return base, jacobian

    def propose(
        self,
        atlas: ManifoldAtlas,
        *,
        target: TrackGeometryProfile | None = None,
        feature_delta: Mapping[str, float] | None = None,
        standardized_direction: F64 | None = None,
        standardized_step: float = 0.50,
        ridge: float = 0.08,
        epsilon_m: float = 0.08,
        name: str | None = None,
    ) -> ManifoldExtensionProposal:
        """Take one constrained local step in a requested descriptor direction."""

        supplied = sum(item is not None for item in (
            target, feature_delta, standardized_direction
        ))
        if supplied != 1:
            raise ValueError("provide exactly one target, feature_delta, or direction")
        base, jacobian = self.descriptor_jacobian(epsilon_m=epsilon_m)
        if base.feature_names != atlas.feature_names:
            raise ValueError("atlas and generator descriptor contracts differ")
        if target is not None:
            requested = (
                target.feature_vector - base.feature_vector
            ) / atlas.feature_scale
            norm = float(np.linalg.norm(requested))
            if norm > standardized_step:
                requested *= standardized_step / norm
        elif feature_delta is not None:
            requested = np.zeros(len(base.feature_vector), np.float64)
            for feature, delta in feature_delta.items():
                index = base.feature_names.index(feature)
                requested[index] = float(delta) / atlas.feature_scale[index]
        else:
            requested = np.asarray(standardized_direction, np.float64).copy()
            if requested.shape != base.feature_vector.shape:
                raise ValueError("standardized direction has the wrong shape")
            norm = float(np.linalg.norm(requested))
            if norm <= 1.0e-12:
                raise ValueError("standardized direction cannot be zero")
            requested *= standardized_step / norm
        controllable = np.asarray([
            name in CONTROLLABLE_FEATURE_NAMES for name in base.feature_names
        ])
        requested[~controllable] = 0.0
        scaled_jacobian = jacobian[controllable] / atlas.feature_scale[controllable, None]
        requested_controllable = requested[controllable]
        gram = scaled_jacobian @ scaled_jacobian.T
        parameters = scaled_jacobian.T @ np.linalg.solve(
            gram + ridge * np.eye(len(gram)), requested_controllable
        )
        reshaped = parameters.reshape(len(self.base_track.gates), 3)
        row_norm = np.linalg.norm(reshaped, axis=1, keepdims=True)
        reshaped *= np.minimum(
            1.0,
            self.maximum_gate_displacement_m / np.maximum(row_norm, 1.0e-12),
        )
        parameters = reshaped.reshape(-1)
        best: tuple[float, Track, TrackGeometryProfile, F64, bool, tuple[str, ...]] | None = None
        for scale in (1.0, 0.75, 0.5, 0.25, 0.125):
            candidate_parameters = scale * parameters
            candidate = self.materialize(candidate_parameters, name=name)
            valid, reasons = self.validate(candidate, candidate_parameters)
            profile = analyze_track_geometry(candidate)
            achieved = (
                profile.feature_vector - base.feature_vector
            ) / atlas.feature_scale
            residual = float(np.linalg.norm(
                requested_controllable - achieved[controllable]
            ))
            row = (scale, candidate, profile, achieved, valid, reasons)
            if best is None or (valid, -residual) > (
                best[4],
                -float(np.linalg.norm(
                    requested_controllable - best[3][controllable]
                )),
            ):
                best = row
            if valid and float(
                achieved[controllable] @ requested_controllable
            ) > 0.0:
                best = row
                break
        assert best is not None
        scale, track, profile, achieved, valid, reasons = best
        alignment = float(
            achieved[controllable] @ requested_controllable
            / max(float(
                np.linalg.norm(achieved[controllable])
                * np.linalg.norm(requested_controllable)
            ), 1.0e-12)
        )
        return ManifoldExtensionProposal(
            track=track,
            profile=profile,
            parameters=scale * parameters,
            requested_standardized_delta=requested,
            achieved_standardized_delta=achieved,
            alignment=alignment,
            residual=float(np.linalg.norm(
                requested_controllable - achieved[controllable]
            )),
            line_search_scale=scale,
            valid=bool(valid and alignment > 0.0),
            reasons=reasons + (() if alignment > 0.0 else ("nonpositive-direction-alignment",)),
        )


class RouteSplineManifoldGenerator:
    """Inverse-design local geometry using the exact route-six actor contract.

    The older generator inverted hand-written geometry descriptors.  Those
    descriptors were useful for diversity reporting but did not predict local
    policy transfer.  Here the finite-difference Jacobian maps spline edits
    directly into ``Track.flight_plan(..., 6)`` token statistics and order.
    MPCC remains a mandatory downstream gate; this class only proposes.
    """

    def __init__(
        self,
        base_track: Track,
        target_track: Track,
        *,
        future_gates: int = 6,
        phase_samples: int = 24,
        include_gate_frames: bool = True,
        include_gate_apertures: bool = True,
        maximum_gate_rotation_degrees: float = 35.0,
        maximum_log_aperture_scale: float = 0.30,
        **spline_options: Any,
    ) -> None:
        self.base_track = base_track
        self.target_track = target_track
        self.future_gates = int(future_gates)
        self.phase_samples = int(phase_samples)
        self.include_gate_frames = bool(include_gate_frames)
        self.include_gate_apertures = bool(include_gate_apertures)
        self.maximum_gate_rotation_radians = float(
            np.deg2rad(maximum_gate_rotation_degrees)
        )
        self.maximum_log_aperture_scale = float(maximum_log_aperture_scale)
        if self.maximum_gate_rotation_radians <= 0.0:
            raise ValueError("maximum gate rotation must be positive")
        if self.maximum_log_aperture_scale <= 0.0:
            raise ValueError("maximum aperture scale must be positive")
        self.spline = SplineManifoldGenerator(base_track, **spline_options)
        self.base_profile = route_grammar_profile(
            base_track, future_gates=self.future_gates,
            phase_samples=self.phase_samples,
        )
        self.target_profile = route_grammar_profile(
            target_track, future_gates=self.future_gates,
            phase_samples=self.phase_samples,
        )
        if self.base_profile.embedding.shape != self.target_profile.embedding.shape:
            raise ValueError("source and target route contracts differ")
        self._jacobian_cache: dict[float, F64] = {}

    @property
    def parameters_per_gate(self) -> int:
        return 3 + 3 * int(self.include_gate_frames) + 2 * int(
            self.include_gate_apertures
        )

    @property
    def parameter_count(self) -> int:
        return len(self.base_track.gates) * self.parameters_per_gate

    @staticmethod
    def _rotation_from_vector(vector: F64) -> F64:
        angle = float(np.linalg.norm(vector))
        if angle <= 1.0e-12:
            return np.eye(3, dtype=np.float64)
        axis = np.asarray(vector, np.float64) / angle
        skew = np.asarray([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        return (
            np.eye(3)
            + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * (skew @ skew)
        )

    def _split_parameters(self, parameters: F64) -> tuple[F64, F64, F64]:
        values = np.asarray(parameters, np.float64).reshape(
            len(self.base_track.gates), self.parameters_per_gate,
        )
        offset = 3
        positions = values[:, :3]
        rotations = np.zeros((len(values), 3), np.float64)
        apertures = np.zeros((len(values), 2), np.float64)
        if self.include_gate_frames:
            rotations = values[:, offset:offset + 3]
            offset += 3
        if self.include_gate_apertures:
            apertures = values[:, offset:offset + 2]
        return positions, rotations, apertures

    def _clip_parameters(self, parameters: F64) -> F64:
        positions, rotations, apertures = self._split_parameters(parameters)
        position_norm = np.linalg.norm(positions, axis=1, keepdims=True)
        positions = positions * np.minimum(
            1.0,
            self.spline.maximum_gate_displacement_m
            / np.maximum(position_norm, 1.0e-12),
        )
        rotation_norm = np.linalg.norm(rotations, axis=1, keepdims=True)
        rotations = rotations * np.minimum(
            1.0,
            self.maximum_gate_rotation_radians
            / np.maximum(rotation_norm, 1.0e-12),
        )
        apertures = np.clip(
            apertures,
            -self.maximum_log_aperture_scale,
            self.maximum_log_aperture_scale,
        )
        blocks = [positions]
        if self.include_gate_frames:
            blocks.append(rotations)
        if self.include_gate_apertures:
            blocks.append(apertures)
        return np.concatenate(blocks, axis=1).reshape(-1)

    def materialize(self, parameters: F64, *, name: str | None = None) -> Track:
        positions, rotations, apertures = self._split_parameters(parameters)
        positioned = self.spline.materialize(positions, name=name)
        gates = []
        for gate, local_rotation, log_aperture in zip(
            positioned.gates, rotations, apertures,
        ):
            world_rotation = gate.rotation @ local_rotation
            frame = self._rotation_from_vector(world_rotation) @ gate.rotation
            size = np.asarray(gate.size, np.float64) * np.exp(log_aperture)
            gates.append(replace(
                gate,
                quaternion_wxyz=matrix_quaternion(frame),
                size=size.astype(np.float32),
            ))
        metadata = dict(positioned.metadata or {})
        metadata.update({
            "generator": "route-six-inverse-spline-v2",
            "route_parameter_contract": {
                "position_per_gate": 3,
                "frame_rotation_per_gate": 3 if self.include_gate_frames else 0,
                "log_aperture_per_gate": 2 if self.include_gate_apertures else 0,
            },
        })
        return replace(positioned, gates=tuple(gates), metadata=metadata)

    def validate(self, track: Track, parameters: F64) -> tuple[bool, tuple[str, ...]]:
        positions, rotations, apertures = self._split_parameters(parameters)
        valid, base_reasons = self.spline.validate(track, positions)
        reasons = list(base_reasons)
        if np.any(np.linalg.norm(rotations, axis=1) > self.maximum_gate_rotation_radians + 1e-8):
            reasons.append("maximum-gate-frame-rotation")
        if np.any(np.abs(apertures) > self.maximum_log_aperture_scale + 1e-8):
            reasons.append("maximum-gate-aperture-change")
        if any(np.min(gate.size) <= 0.0 for gate in track.gates):
            reasons.append("nonpositive-gate-aperture")
        return bool(valid and not reasons), tuple(dict.fromkeys(reasons))

    def route_jacobian(self, *, epsilon_m: float = 0.04) -> F64:
        epsilon_m = float(epsilon_m)
        if epsilon_m <= 0.0:
            raise ValueError("route Jacobian epsilon must be positive")
        cached = self._jacobian_cache.get(epsilon_m)
        if cached is not None:
            return cached.copy()
        count = self.parameter_count
        jacobian = np.empty((len(self.base_profile.embedding), count), np.float64)
        for index in range(count):
            positive = np.zeros(count, np.float64)
            negative = np.zeros(count, np.float64)
            positive[index] = epsilon_m
            negative[index] = -epsilon_m
            high = route_grammar_profile(
                self.materialize(positive),
                future_gates=self.future_gates,
                phase_samples=self.phase_samples,
            ).embedding
            low = route_grammar_profile(
                self.materialize(negative),
                future_gates=self.future_gates,
                phase_samples=self.phase_samples,
            ).embedding
            jacobian[:, index] = (high - low) / (2.0 * epsilon_m)
        self._jacobian_cache[epsilon_m] = jacobian.copy()
        return jacobian

    def propose(
        self,
        *,
        target_fraction_step: float = 0.02,
        ridge_values: tuple[float, ...] = (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2),
        line_search_scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.125),
        epsilon_m: float = 0.04,
        off_axis_penalty: float = 0.25,
        name: str | None = None,
    ) -> RouteExtensionProposal:
        if not 0.0 < target_fraction_step <= 1.0:
            raise ValueError("target fraction step must lie in (0, 1]")
        if not ridge_values or min(ridge_values) <= 0.0:
            raise ValueError("route inverse ridge values must be positive")
        if not line_search_scales or min(line_search_scales) <= 0.0:
            raise ValueError("line search scales must be positive")
        jacobian = self.route_jacobian(epsilon_m=epsilon_m)
        direction = self.target_profile.embedding - self.base_profile.embedding
        baseline_distance = float(np.linalg.norm(direction))
        if baseline_distance <= 1.0e-12:
            raise ValueError("base and target route contracts are indistinguishable")
        desired = float(target_fraction_step) * direction
        gram = jacobian.T @ jacobian
        rhs = jacobian.T @ desired
        projected_direction = jacobian @ np.linalg.lstsq(
            jacobian, direction, rcond=1.0e-6,
        )[0]
        tangent_reachable_fraction = float(
            np.linalg.norm(projected_direction) / baseline_distance
        )
        tangent_residual_fraction = float(
            np.linalg.norm(direction - projected_direction) / baseline_distance
        )
        best: tuple[
            float, float, Track, RouteGrammarProfile, RouteDirectedPosition,
            F64, float, bool, tuple[str, ...],
        ] | None = None
        for ridge in ridge_values:
            parameters = np.linalg.solve(
                gram + float(ridge) * np.eye(gram.shape[0]), rhs,
            )
            parameters = self._clip_parameters(parameters)
            for scale in line_search_scales:
                values = float(scale) * parameters
                track = self.materialize(values, name=name)
                geometry_valid, reasons = self.validate(track, values)
                profile = route_grammar_profile(
                    track, future_gates=self.future_gates,
                    phase_samples=self.phase_samples,
                )
                position = route_directed_position(
                    self.base_profile, self.target_profile, profile,
                )
                local_gain = float(
                    1.0 - position.target_distance / baseline_distance
                )
                residual = float(np.linalg.norm(
                    (profile.embedding - self.base_profile.embedding) - desired
                ) / baseline_distance)
                score = local_gain - float(off_axis_penalty) * position.off_axis_ratio
                valid = bool(geometry_valid and local_gain > 0.0 and position.progress > 0.0)
                row = (
                    valid, score, track, profile, position, values, residual,
                    float(ridge), reasons,
                )
                if best is None or (row[0], row[1]) > (best[0], best[1]):
                    best = row
        assert best is not None
        valid, _, track, profile, position, parameters, residual, ridge, reasons = best
        if position.progress <= 0.0:
            reasons = reasons + ("nonpositive-route-progress",)
        if position.target_gain_fraction <= 0.0:
            reasons = reasons + ("no-route-target-gain",)
        return RouteExtensionProposal(
            track=track,
            profile=profile,
            position=position,
            parameters=parameters,
            requested_target_fraction=float(target_fraction_step),
            achieved_local_target_gain_fraction=float(position.target_gain_fraction),
            residual=residual,
            line_search_scale=float(
                np.linalg.norm(parameters) / max(
                    np.linalg.norm(np.linalg.solve(
                        gram + ridge * np.eye(gram.shape[0]), rhs,
                    )), 1.0e-12,
                )
            ),
            ridge=ridge,
            tangent_reachable_fraction=tangent_reachable_fraction,
            tangent_residual_fraction=tangent_residual_fraction,
            valid=bool(valid),
            reasons=reasons,
        )
