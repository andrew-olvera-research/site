"""Explicit topology-normalized bridges between named racing courses."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splev, splprep

from ..tracks import Gate, Track, forward_up_quaternion


F64 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MorphAlignment:
    phase_shift: int
    yaw_radians: float
    translation: F64
    rms_correspondence_m: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "phase_shift": self.phase_shift,
            "yaw_radians": self.yaw_radians,
            "translation": self.translation.tolist(),
            "rms_correspondence_m": self.rms_correspondence_m,
        }


class PhaseAlignedTrackMorpher:
    """Interpolate source geometry toward a phase-aligned target proxy.

    Local spline inverse design preserves gate count and therefore cannot move
    between topology classes.  This bridge first resamples the target closed
    curve to the source gate count, then solves cyclic phase and global-yaw
    alignment.  It creates an explicit same-topology waypoint; reaching the
    original target still requires a separately qualified gate merge/delete.
    """

    def __init__(
        self,
        source: Track,
        target: Track,
        *,
        bounds_margin_m: float = 5.0,
        target_anchor_gate: str | None = None,
        source_anchor_index: int = 0,
    ) -> None:
        if not source.loop or not target.loop:
            raise ValueError("phase-aligned morphing currently requires loop tracks")
        if len(source.gates) < 4 or len(target.gates) < 4:
            raise ValueError("phase-aligned morphing requires at least four gates")
        self.source = source
        self.target = target
        self.bounds_margin_m = float(bounds_margin_m)
        self.target_anchor_gate = (
            None if target_anchor_gate is None else str(target_anchor_gate)
        )
        self.source_anchor_index = int(source_anchor_index)
        if not 0 <= self.source_anchor_index < len(source.gates):
            raise ValueError("source_anchor_index lies outside the source route")
        if self.target_anchor_gate is None:
            self._target_anchor_index: int | None = None
        else:
            target_names = [gate.name for gate in target.gates]
            if self.target_anchor_gate not in target_names:
                raise ValueError(
                    f"unknown target anchor gate: {self.target_anchor_gate!r}"
                )
            self._target_anchor_index = target_names.index(self.target_anchor_gate)
        self._source_points = np.stack([gate.position for gate in source.gates]).astype(np.float64)
        (
            points, sizes, normal, up, inserted, target_indices,
        ) = self._expand_target_preserving_gates(
            len(source.gates)
        )
        (
            self._target_points,
            self._target_sizes,
            self._target_normal,
            self._target_up,
            self._target_inserted,
            self._target_gate_indices,
            self.alignment,
        ) = (
            self._align_target(
                points, sizes, normal, up, inserted, target_indices,
            )
        )

    def _expand_target_preserving_gates(
        self, count: int,
    ) -> tuple[
        F64, F64, F64, F64, NDArray[np.bool_], NDArray[np.int64]
    ]:
        """Add pass-through checkpoints without moving target gates.

        Uniformly resampling a target spline erases the real gate locations and
        orientations.  That produced geometrically smooth endpoints which were
        not the target control problem and which MPCC could not reliably fly.
        A topology proxy must instead retain every target gate exactly and add
        only removable checkpoints on long inter-gate arcs.
        """

        points = np.stack([gate.position for gate in self.target.gates]).astype(np.float64)
        target_count = len(points)
        if count < target_count:
            raise ValueError(
                "target has more gates than the requested proxy; preserving all "
                "target gates requires expanding the source topology first"
            )
        if count == target_count:
            return (
                points,
                np.stack([gate.size for gate in self.target.gates]).astype(np.float64),
                np.stack([gate.normal for gate in self.target.gates]).astype(np.float64),
                np.stack([gate.up for gate in self.target.gates]).astype(np.float64),
                np.zeros(target_count, dtype=bool),
                np.arange(target_count, dtype=np.int64),
            )

        gate_phase = np.arange(target_count, dtype=np.float64) / target_count
        # ``splprep(..., per=True)`` mutates the final sample of some strided
        # NumPy views in-place.  Passing ``points.T`` here used to overwrite
        # the final real target gate with gate zero, silently creating a
        # duplicate physical gate in every topology proxy.  This is exactly
        # the kind of geometric corruption that MPCC should quarantine, but
        # the generator must not manufacture it in the first place.
        tck, _ = splprep(
            points.T.copy(),
            u=gate_phase,
            s=0.0,
            per=True,
            k=min(3, target_count - 1),
        )
        dense_phase = np.linspace(0.0, 1.0, 4097)
        dense = np.stack(splev(dense_phase, tck), axis=-1)
        cumulative = np.concatenate([
            [0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))
        ])

        # Repeatedly split the longest current arc. This gives well-separated,
        # low-curvature removable checkpoints instead of clustering them in one
        # long straight or perturbing the target's physical gates.
        phases = [
            (float(value), False, index) for index, value in enumerate(gate_phase)
        ]
        for _ in range(count - target_count):
            phases.sort(key=lambda row: row[0])
            best: tuple[float, int, float] | None = None
            for index, (left, _, _) in enumerate(phases):
                right = phases[(index + 1) % len(phases)][0]
                if index == len(phases) - 1:
                    right += 1.0
                left_length = float(np.interp(left, dense_phase, cumulative))
                right_length = float(np.interp(right, dense_phase, cumulative))
                row = (right_length - left_length, index, 0.5 * (left + right))
                if best is None or row[0] > best[0]:
                    best = row
            assert best is not None
            phases.append((best[2] % 1.0, True, -1))
        phases.sort(key=lambda row: row[0])
        query_phase = np.asarray([row[0] for row in phases], np.float64)
        inserted = np.asarray([row[1] for row in phases], bool)
        target_indices = np.asarray([row[2] for row in phases], np.int64)

        sampled = np.stack(splev(query_phase, tck), axis=-1)
        # Restore the exact released gate coordinates; spline evaluation at a
        # knot is numerically close but should not become a silent geometry edit.
        for index, phase in enumerate(query_phase):
            original = np.flatnonzero(np.isclose(gate_phase, phase, atol=1.0e-10))
            if len(original):
                sampled[index] = points[int(original[0])]
        tangent = np.stack(splev(query_phase, tck, der=1), axis=-1)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.0e-12)

        raw_sizes = np.stack([gate.size for gate in self.target.gates]).astype(np.float64)
        raw_normal = np.stack([gate.normal for gate in self.target.gates]).astype(np.float64)
        raw_up = np.stack([gate.up for gate in self.target.gates]).astype(np.float64)
        sizes: list[F64] = []
        normal: list[F64] = []
        up: list[F64] = []
        for phase, is_inserted, local_tangent in zip(query_phase, inserted, tangent):
            original = np.flatnonzero(np.isclose(gate_phase, phase, atol=1.0e-10))
            if len(original):
                source_index = int(original[0])
                sizes.append(raw_sizes[source_index])
                normal.append(raw_normal[source_index])
                up.append(raw_up[source_index])
                continue
            # A removable checkpoint is deliberately generous. Its only role is
            # to make the gate-count transition explicit while preserving the
            # exact target's physical gates and racing line.
            left = int(np.floor(phase * target_count)) % target_count
            right = (left + 1) % target_count
            sizes.append(np.maximum(1.75 * 0.5 * (raw_sizes[left] + raw_sizes[right]), 4.0))
            normal.append(local_tangent)
            world_up = np.asarray([0.0, 0.0, 1.0])
            projected = world_up - local_tangent * float(world_up @ local_tangent)
            if np.linalg.norm(projected) < 1.0e-6:
                projected = np.asarray([0.0, 1.0, 0.0])
                projected -= local_tangent * float(projected @ local_tangent)
            up.append(projected / np.linalg.norm(projected))
        return (
            sampled,
            np.asarray(sizes, np.float64),
            np.asarray(normal, np.float64),
            np.asarray(up, np.float64),
            inserted,
            target_indices,
        )

    def _align_target(
        self,
        points: F64,
        sizes: F64,
        normal: F64,
        up: F64,
        inserted: NDArray[np.bool_],
        target_indices: NDArray[np.int64],
    ) -> tuple[
        F64, F64, F64, F64, NDArray[np.bool_], NDArray[np.int64],
        MorphAlignment,
    ]:
        source_center = self._source_points.mean(axis=0)
        source_centered = self._source_points - source_center
        best: tuple[
            float, int, float, F64, F64, F64, F64, NDArray[np.bool_],
            NDArray[np.int64],
        ] | None = None
        for shift in range(len(points)):
            shifted = np.roll(points, shift, axis=0)
            shifted_sizes = np.roll(sizes, shift, axis=0)
            shifted_normal = np.roll(normal, shift, axis=0)
            shifted_up = np.roll(up, shift, axis=0)
            shifted_inserted = np.roll(inserted, shift, axis=0)
            shifted_target_indices = np.roll(target_indices, shift, axis=0)
            # Cyclic invariance is correct for geometry analysis, but a
            # controller's reset gate is part of the task contract.  When an
            # anchor is supplied, constrain the correspondence so source gate
            # zero (or another requested reset index) maps to the named target
            # start gate.  This prevents a low-RMS phase shift from training a
            # different transition sequence than the real target executes.
            if (
                self._target_anchor_index is not None
                and int(shifted_target_indices[self.source_anchor_index])
                != self._target_anchor_index
            ):
                continue
            target_center = shifted.mean(axis=0)
            centered = shifted - target_center
            numerator = float(np.sum(
                centered[:, 0] * source_centered[:, 1]
                - centered[:, 1] * source_centered[:, 0]
            ))
            denominator = float(np.sum(
                centered[:, 0] * source_centered[:, 0]
                + centered[:, 1] * source_centered[:, 1]
            ))
            yaw = float(np.arctan2(numerator, denominator))
            rotation = np.asarray([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ])
            transformed = centered @ rotation.T + source_center
            transformed_normal = shifted_normal @ rotation.T
            transformed_up = shifted_up @ rotation.T
            rms = float(np.sqrt(np.mean(np.sum((transformed - self._source_points) ** 2, axis=1))))
            row = (
                rms, shift, yaw, transformed, shifted_sizes,
                transformed_normal, transformed_up, shifted_inserted,
                shifted_target_indices,
            )
            if best is None or rms < best[0]:
                best = row
        assert best is not None
        (
            rms, shift, yaw, transformed, shifted_sizes,
            transformed_normal, transformed_up, shifted_inserted,
            shifted_target_indices,
        ) = best
        translation = source_center - np.roll(points, shift, axis=0).mean(axis=0) @ np.asarray([
            [np.cos(yaw), np.sin(yaw), 0.0],
            [-np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        return (
            transformed, shifted_sizes, transformed_normal, transformed_up,
            shifted_inserted,
            shifted_target_indices,
            MorphAlignment(
            phase_shift=int(shift), yaw_radians=yaw,
            translation=np.asarray(translation, np.float64),
            rms_correspondence_m=rms,
            ),
        )

    def materialize(
        self,
        alpha: float,
        *,
        name: str | None = None,
        demote_inserted_checkpoints: bool = False,
    ) -> Track:
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("morph alpha must lie in [0, 1]")
        return self.materialize_gatewise(
            np.full(len(self.source.gates), alpha, np.float64),
            name=name,
            demote_inserted_checkpoints=demote_inserted_checkpoints,
        )

    def materialize_target_localized(
        self,
        *,
        default_alpha: float,
        target_gate_alphas: Mapping[str, float],
        name: str | None = None,
        demote_inserted_checkpoints: bool = False,
    ) -> Track:
        """Move selected target-correspondence gates on an independent schedule.

        A whole-course interpolation can spend most of its route distance on
        transitions unrelated to a policy's observed failure.  This operation
        keeps the same phase-aligned topology proxy but lets a named physical
        target chain (for example CDRA G1->G2->G3) advance independently.  It
        is still only a geometry proposal: MPCC and frozen-policy admission
        remain mandatory before the result enters training.
        """

        default_alpha = float(default_alpha)
        if not 0.0 <= default_alpha <= 1.0:
            raise ValueError("default_alpha must lie in [0, 1]")
        requested = {str(key): float(value) for key, value in target_gate_alphas.items()}
        if any(not 0.0 <= value <= 1.0 for value in requested.values()):
            raise ValueError("target gate alphas must lie in [0, 1]")
        known = {gate.name for gate in self.target.gates}
        unknown = sorted(set(requested) - known)
        if unknown:
            raise ValueError(f"unknown target gate names: {unknown}")

        alphas = np.full(len(self.source.gates), default_alpha, np.float64)
        matched: set[str] = set()
        for index, target_index in enumerate(self._target_gate_indices):
            if target_index < 0:
                continue
            target_name = self.target.gates[int(target_index)].name
            if target_name in requested:
                alphas[index] = requested[target_name]
                matched.add(target_name)
        missing = sorted(set(requested) - matched)
        if missing:
            raise ValueError(f"target gates have no aligned correspondence: {missing}")
        track = self.materialize_gatewise(
            alphas,
            name=name,
            demote_inserted_checkpoints=demote_inserted_checkpoints,
        )
        metadata = dict(track.metadata or {})
        localized = dict(metadata["phase_aligned_morph"])
        localized.update({
            "operation": "target_gate_localized_morph",
            "default_alpha": default_alpha,
            "target_gate_alphas": requested,
        })
        metadata["phase_aligned_morph"] = localized
        return replace(track, metadata=metadata)

    def materialize_gatewise(
        self,
        alphas: Sequence[float] | F64,
        *,
        name: str | None = None,
        demote_inserted_checkpoints: bool = False,
    ) -> Track:
        """Materialize one independently bounded interpolation per route gate."""

        alpha_values = np.asarray(alphas, np.float64)
        if alpha_values.shape != (len(self.source.gates),):
            raise ValueError(
                f"gatewise alphas must have shape {(len(self.source.gates),)}"
            )
        if not np.all(np.isfinite(alpha_values)):
            raise ValueError("gatewise alphas must be finite")
        if np.any(alpha_values < 0.0) or np.any(alpha_values > 1.0):
            raise ValueError("gatewise alphas must lie in [0, 1]")
        blend = alpha_values[:, None]
        points = (1.0 - blend) * self._source_points + blend * self._target_points
        source_sizes = np.stack([gate.size for gate in self.source.gates])
        sizes = (1.0 - blend) * source_sizes + blend * self._target_sizes
        source_normal = np.stack([gate.normal for gate in self.source.gates]).astype(np.float64)
        normal = (1.0 - blend) * source_normal + blend * self._target_normal
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-12)
        source_up = np.stack([gate.up for gate in self.source.gates]).astype(np.float64)
        up = (1.0 - blend) * source_up + blend * self._target_up
        gates = tuple(
            Gate(
                position=point.astype(np.float32),
                quaternion_wxyz=forward_up_quaternion(direction, up_hint),
                size=size.astype(np.float32),
                name=(
                    f"route_checkpoint_{index:02d}"
                    if demote_inserted_checkpoints and self._target_inserted[index]
                    else (
                        self.target.gates[
                            int(self._target_gate_indices[index])
                        ].name
                        if demote_inserted_checkpoints
                        and self._target_gate_indices[index] >= 0
                        else f"gate_{index:02d}"
                    )
                ),
                kind=(
                    "maneuver_route"
                    if demote_inserted_checkpoints and self._target_inserted[index]
                    else (
                        self.target.gates[
                            int(self._target_gate_indices[index])
                        ].kind
                        if demote_inserted_checkpoints
                        and self._target_gate_indices[index] >= 0
                        else "gate"
                    )
                ),
                render=(
                    bool(self.target.gates[
                        int(self._target_gate_indices[index])
                    ].render)
                    if demote_inserted_checkpoints
                    and self._target_gate_indices[index] >= 0
                    else not bool(
                        demote_inserted_checkpoints
                        and self._target_inserted[index]
                    )
                ),
            )
            for index, (point, direction, up_hint, size) in enumerate(zip(points, normal, up, sizes))
        )
        margin = self.bounds_margin_m
        bounds = np.stack([points.min(axis=0) - margin, points.max(axis=0) + margin], axis=1)
        bounds[2, 0] = min(bounds[2, 0], 0.0)
        metadata = dict(self.source.metadata or {})
        metadata["phase_aligned_morph"] = {
            "schema": "starscream-phase-aligned-track-morph-v1",
            "source": self.source.name,
            "target": self.target.name,
            "target_gate_count": len(self.target.gates),
            "proxy_gate_count": len(self.source.gates),
            "inserted_target_checkpoint_indices": np.flatnonzero(
                self._target_inserted
            ).tolist(),
            "alpha": (
                float(alpha_values[0])
                if np.allclose(alpha_values, alpha_values[0]) else None
            ),
            "gatewise_alphas": alpha_values.tolist(),
            "inserted_checkpoints_demoted": bool(demote_inserted_checkpoints),
            "target_anchor_gate": self.target_anchor_gate,
            "source_anchor_index": self.source_anchor_index,
            "alignment": self.alignment.to_mapping(),
            "scope": (
                "same-topology bridge preserving every target gate; inserted "
                "pass-through checkpoints remain explicit"
            ),
        }
        default_name = (
            f"{self.source.name}_to_{self.target.name}_gatewise_"
            f"{float(alpha_values.mean()):.3f}"
        )
        return Track(
            name=name or default_name,
            gates=gates,
            bounds=bounds.astype(np.float32),
            loop=True,
            metadata=metadata,
        )
