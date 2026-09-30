"""Gate-centric trajectory composition for benchmark-sealed racing data.

The whole-course generator describes *where* gates are.  This module records
the dynamically evidenced local flight problems that an MPCC expert solved and
recombines those problems before asking MPCC to project the resulting course
back onto the feasible racing manifold.

Only XY yaw and translation are treated as exact geometric symmetries.  CTBR
commands are never warped into labels for a new course: generated courses must
be independently qualified and relabeled by MPCC before entering DAgger.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from ..procedural_tracks import geometry_fingerprint
from ..tracks import Gate, Track, forward_up_quaternion, load_track
from .grammar import ManeuverGrammar
from .schema import CourseProgram, RacingDistributionConfig
from .validator import RacingDistributionValidator, TrackValidationReport


def _unit(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, np.float64)
    return array / max(float(np.linalg.norm(array)), 1.0e-12)


def _yaw_rotate(value: np.ndarray, yaw: float) -> np.ndarray:
    result = np.asarray(value, np.float64).copy()
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    result[..., 0], result[..., 1] = (
        c * value[..., 0] - s * value[..., 1],
        s * value[..., 0] + c * value[..., 1],
    )
    return result


def _signed_angle(left: np.ndarray, right: np.ndarray, axis: np.ndarray) -> float:
    a, b, n = _unit(left), _unit(right), _unit(axis)
    return float(np.arctan2(float(n @ np.cross(a, b)), float(np.clip(a @ b, -1.0, 1.0))))


def _gate_roll(gate: Gate) -> float:
    normal = _unit(gate.normal)
    base = np.asarray([0.0, 0.0, 1.0])
    base -= normal * float(base @ normal)
    if np.linalg.norm(base) < 1.0e-6:
        base = np.asarray([0.0, 1.0, 0.0])
        base -= normal * float(base @ normal)
    up = np.asarray(gate.up, np.float64)
    up -= normal * float(up @ normal)
    return _signed_angle(base, up, normal)


def _normal_yaw_offset(gate: Gate, outgoing: np.ndarray) -> float:
    tangent = np.asarray(outgoing, np.float64).copy()
    normal = np.asarray(gate.normal, np.float64).copy()
    tangent[2] = normal[2] = 0.0
    if min(np.linalg.norm(tangent), np.linalg.norm(normal)) < 1.0e-6:
        return 0.0
    return _signed_angle(tangent, normal, np.asarray([0.0, 0.0, 1.0]))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class FlightSegment:
    """One MPCC-evidenced maneuver expressed in its incoming yaw frame."""

    segment_id: str
    primitive: str
    source_track: str
    source_geometry_fingerprint: str
    source_macro_index: int
    vectors_local: tuple[tuple[float, float, float], ...]
    gate_roll_radians: tuple[float, ...]
    gate_normal_yaw_offsets: tuple[float, ...]
    gate_sizes: tuple[tuple[float, float], ...]
    qualified_speed_mps: float
    entry_speed_mps: float
    exit_speed_mps: float
    mean_collective_mps2: float
    maximum_collective_mps2: float
    trajectory_evidenced: bool

    @property
    def gate_count(self) -> int:
        return len(self.vectors_local)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FlightSegment":
        return cls(
            segment_id=str(payload["segment_id"]),
            primitive=str(payload["primitive"]),
            source_track=str(payload["source_track"]),
            source_geometry_fingerprint=str(payload["source_geometry_fingerprint"]),
            source_macro_index=int(payload["source_macro_index"]),
            vectors_local=tuple(tuple(float(x) for x in row) for row in payload["vectors_local"]),
            gate_roll_radians=tuple(float(x) for x in payload["gate_roll_radians"]),
            gate_normal_yaw_offsets=tuple(float(x) for x in payload["gate_normal_yaw_offsets"]),
            gate_sizes=tuple(tuple(float(x) for x in row) for row in payload["gate_sizes"]),
            qualified_speed_mps=float(payload["qualified_speed_mps"]),
            entry_speed_mps=float(payload["entry_speed_mps"]),
            exit_speed_mps=float(payload["exit_speed_mps"]),
            mean_collective_mps2=float(payload["mean_collective_mps2"]),
            maximum_collective_mps2=float(payload["maximum_collective_mps2"]),
            trajectory_evidenced=bool(payload["trajectory_evidenced"]),
        )


@dataclass(frozen=True, slots=True)
class FlightSegmentLibrary:
    schema: str
    source_manifest: str
    source_manifest_sha256: str
    dataset_root: str | None
    segments: tuple[FlightSegment, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_manifest": self.source_manifest,
            "source_manifest_sha256": self.source_manifest_sha256,
            "dataset_root": self.dataset_root,
            "segments": [item.to_mapping() for item in self.segments],
        }

    @classmethod
    def load(cls, path: str | Path) -> "FlightSegmentLibrary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "starscream-flight-segment-library-v1":
            raise ValueError("unsupported flight segment library")
        return cls(
            schema=str(payload["schema"]),
            source_manifest=str(payload["source_manifest"]),
            source_manifest_sha256=str(payload["source_manifest_sha256"]),
            dataset_root=(None if payload.get("dataset_root") is None else str(payload["dataset_root"])),
            segments=tuple(FlightSegment.from_mapping(item) for item in payload["segments"]),
        )


def _episode_for_track(dataset_root: Path | None, split: str, name: str) -> Path | None:
    if dataset_root is None:
        return None
    root = dataset_root / split / name
    preferred = root / "expert-nominal-000.h5"
    if preferred.exists():
        return preferred
    values = sorted(root.glob("expert-*.h5"))
    return values[0] if values else None


def _behavior_by_macro(
    episode_path: Path | None,
    macro_indices: Sequence[int],
) -> dict[int, dict[str, float]]:
    if episode_path is None:
        return {}
    with h5py.File(episode_path, "r", swmr=True) as archive:
        gate_index = np.asarray(archive["transition/gate_index"], np.int64)
        state_key = next((key for key in (
            "observation/privileged/state", "observation/state",
        ) if key in archive), None)
        if state_key is None:
            return {}
        state = np.asarray(archive[state_key], np.float64)[: len(gate_index)]
        action = np.asarray(archive["action/ctbr"], np.float64)[: len(gate_index)]
    speed = np.linalg.norm(state[:, 7:10], axis=1)
    result: dict[int, dict[str, float]] = {}
    macro_array = np.asarray(macro_indices, np.int64)
    for macro in sorted(set(macro_indices)):
        gates = np.flatnonzero(macro_array == macro)
        selected = np.flatnonzero(np.isin(gate_index, gates))
        if not len(selected):
            continue
        result[int(macro)] = {
            "entry_speed_mps": float(speed[selected[0]]),
            "exit_speed_mps": float(speed[selected[-1]]),
            "mean_collective_mps2": float(np.mean(action[selected, 0])),
            "maximum_collective_mps2": float(np.max(action[selected, 0])),
        }
    return result


def extract_track_segments(
    track: Track,
    record: Mapping[str, Any],
    *,
    episode_path: Path | None = None,
) -> tuple[FlightSegment, ...]:
    """Extract macro-aligned local geometry and optional expert behavior."""

    metadata = dict(track.metadata or {})
    macro_indices = tuple(int(x) for x in metadata.get("expanded_macro_indices", ()))
    macro_labels = tuple(str(x) for x in metadata.get("macro_labels", ()))
    if len(macro_indices) != len(track.gates) or not macro_labels:
        return ()
    behavior = _behavior_by_macro(episode_path, macro_indices)
    points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
    vectors = np.roll(points, -1, axis=0) - points
    qualified_speed = float(record.get("qualified_speed_mps") or 0.0)
    result: list[FlightSegment] = []
    for macro, primitive in enumerate(macro_labels):
        indices = np.flatnonzero(np.asarray(macro_indices) == macro)
        if not len(indices):
            continue
        selected_vectors = vectors[indices]
        first = selected_vectors[0]
        incoming_yaw = float(np.arctan2(first[1], first[0]))
        local = _yaw_rotate(selected_vectors, -incoming_yaw)
        stats = behavior.get(macro, {})
        payload = (
            f"{track.name}:{geometry_fingerprint(track)}:{macro}:{primitive}"
        ).encode()
        result.append(FlightSegment(
            segment_id=hashlib.sha256(payload).hexdigest()[:20],
            primitive=primitive,
            source_track=track.name,
            source_geometry_fingerprint=geometry_fingerprint(track),
            source_macro_index=macro,
            vectors_local=tuple(tuple(float(x) for x in row) for row in local),
            gate_roll_radians=tuple(_gate_roll(track.gates[int(i)]) for i in indices),
            gate_normal_yaw_offsets=tuple(
                _normal_yaw_offset(track.gates[int(i)], vectors[int(i)]) for i in indices
            ),
            gate_sizes=tuple(tuple(float(x) for x in track.gates[int(i)].size) for i in indices),
            qualified_speed_mps=qualified_speed,
            entry_speed_mps=float(stats.get("entry_speed_mps", qualified_speed)),
            exit_speed_mps=float(stats.get("exit_speed_mps", qualified_speed)),
            mean_collective_mps2=float(stats.get("mean_collective_mps2", 0.0)),
            maximum_collective_mps2=float(stats.get("maximum_collective_mps2", 0.0)),
            trajectory_evidenced=bool(stats),
        ))
    return tuple(result)


def build_segment_library(
    manifest: str | Path,
    output: str | Path,
    *,
    dataset_root: str | Path | None = None,
) -> Path:
    manifest_path = Path(manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = None if dataset_root is None else Path(dataset_root).resolve()
    segments: list[FlightSegment] = []
    for record in payload.get("records", ()):
        if str(record.get("split")) != "train":
            continue
        track = load_track(manifest_path.parent / str(record["path"]))
        episode = _episode_for_track(dataset, "train", track.name)
        segments.extend(extract_track_segments(track, record, episode_path=episode))
    if not segments:
        raise RuntimeError("segment library requires macro-annotated training tracks")
    library = FlightSegmentLibrary(
        schema="starscream-flight-segment-library-v1",
        source_manifest=str(manifest_path),
        source_manifest_sha256=_sha256(manifest_path),
        dataset_root=None if dataset is None else str(dataset),
        segments=tuple(segments),
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(library.to_mapping(), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return destination


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * float(axis @ vector) * (1.0 - np.cos(angle))
    )


class FlightMimicComposer:
    """Compose gate-centric expert segments, leaving action labels to MPCC."""

    def __init__(
        self,
        library: FlightSegmentLibrary,
        config: RacingDistributionConfig | None = None,
    ) -> None:
        self.library = library
        self.config = config or RacingDistributionConfig()
        self.by_primitive: dict[str, tuple[FlightSegment, ...]] = {}
        for primitive in sorted({item.primitive for item in library.segments}):
            self.by_primitive[primitive] = tuple(
                item for item in library.segments if item.primitive == primitive
            )

    def _select_segments(
        self, program: CourseProgram, rng: np.random.Generator,
    ) -> tuple[FlightSegment, ...]:
        selected: list[FlightSegment] = []
        previous_speed: float | None = None
        usage: dict[str, int] = {}
        for spec in program.maneuvers:
            values = self.by_primitive.get(spec.kind.value, ())
            if not values:
                raise ValueError(f"segment library has no {spec.kind.value!r} primitive")
            jitter = rng.uniform(0.0, 0.20, len(values))
            scores = np.asarray([
                (0.0 if previous_speed is None else abs(previous_speed - item.entry_speed_mps) / 8.0)
                + 0.35 * usage.get(item.source_track, 0)
                + (0.0 if item.trajectory_evidenced else 0.50)
                + float(jitter[index])
                for index, item in enumerate(values)
            ])
            chosen = values[int(np.argmin(scores))]
            selected.append(chosen)
            previous_speed = chosen.exit_speed_mps
            usage[chosen.source_track] = usage.get(chosen.source_track, 0) + 1
        return tuple(selected)

    def compose(
        self,
        *,
        program: CourseProgram,
        seed: int,
        split: str,
        name: str,
    ) -> Track:
        if split != "train":
            raise ValueError("FlightMimic composition is training-only in the matched ablation")
        cfg = self.config
        rng = np.random.default_rng(int(seed))
        # Segment identity is itself an augmentation variable.  Greedy local
        # boundary matching can create a globally open chain even though every
        # constituent maneuver was feasible.  Draw several lineage-preserving
        # assignments and retain the one requiring the smallest distributed
        # loop-closure correction.  This is geometry selection only: MPCC must
        # still qualify and relabel the resulting course.
        best: tuple[
            float, tuple[FlightSegment, ...], np.ndarray,
            list[float], list[float], list[np.ndarray], list[int], list[str],
        ] | None = None
        for _ in range(64):
            trial_segments = self._select_segments(program, rng)
            trial_vectors: list[np.ndarray] = []
            trial_rolls: list[float] = []
            trial_normal_offsets: list[float] = []
            trial_sizes: list[np.ndarray] = []
            trial_macro_indices: list[int] = []
            trial_labels: list[str] = []
            heading = 0.0
            for macro, (spec, segment) in enumerate(zip(program.maneuvers, trial_segments)):
                local = np.asarray(segment.vectors_local, np.float64).copy()
                local[:, :2] *= float(spec.length_scale) * float(rng.uniform(0.96, 1.04))
                local[:, 2] *= float(spec.vertical_scale) * float(rng.uniform(0.96, 1.04))
                transformed = _yaw_rotate(local, heading)
                trial_vectors.extend(transformed)
                trial_rolls.extend(segment.gate_roll_radians)
                trial_normal_offsets.extend(segment.gate_normal_yaw_offsets)
                trial_sizes.extend(np.asarray(segment.gate_sizes, np.float64))
                trial_macro_indices.extend([macro] * segment.gate_count)
                trial_labels.extend([segment.primitive] * segment.gate_count)
                final = transformed[-1]
                if np.linalg.norm(final[:2]) > 1.0e-6:
                    heading = float(np.arctan2(final[1], final[0]))
            trial_values = np.stack(trial_vectors)
            trial_closure = trial_values.mean(axis=0)
            trial_correction = float(
                np.linalg.norm(trial_closure)
                / max(float(np.median(np.linalg.norm(trial_values, axis=1))), 1.0e-9)
            )
            boundary_mismatch = float(np.mean([
                abs(left.exit_speed_mps - right.entry_speed_mps)
                for left, right in zip(
                    trial_segments, trial_segments[1:] + trial_segments[:1]
                )
            ]))
            objective = trial_correction + 0.01 * boundary_mismatch
            if best is None or objective < best[0]:
                best = (
                    objective, trial_segments, trial_values, trial_rolls,
                    trial_normal_offsets, trial_sizes, trial_macro_indices,
                    trial_labels,
                )
        assert best is not None
        (
            _, segments, values, rolls, normal_offsets, sizes, macro_indices,
            primitive_labels,
        ) = best
        values = _yaw_rotate(values, float(rng.uniform(-np.pi, np.pi)))
        if not cfg.minimum_gates <= len(values) <= cfg.maximum_gates:
            raise ValueError("composed gate count outside distribution contract")
        closure = values.mean(axis=0)
        correction = float(
            np.linalg.norm(closure)
            / max(float(np.median(np.linalg.norm(values, axis=1))), 1.0e-9)
        )
        if correction > cfg.maximum_closure_correction_fraction:
            raise ValueError("composed closure correction exceeds distribution contract")
        values -= closure
        raw_length = float(np.sum(np.linalg.norm(values, axis=1)))
        target_length = float(np.clip(
            raw_length * rng.uniform(0.94, 1.06),
            cfg.minimum_length_m + 2.0,
            cfg.maximum_length_m - 2.0,
        ))
        values[:, :2] *= target_length / max(raw_length, 1.0e-9)
        points = np.concatenate([np.zeros((1, 3)), np.cumsum(values[:-1], axis=0)])
        points[:, 2] -= float(points[:, 2].min())
        excursion = float(np.ptp(points[:, 2]))
        vertical_room = cfg.maximum_altitude_m - cfg.minimum_altitude_m - 0.5
        if excursion > vertical_room:
            points[:, 2] *= vertical_room / excursion
        points[:, 2] += cfg.minimum_altitude_m + float(rng.uniform(0.15, 0.55))
        points[:, :2] += rng.uniform(-2.0, 2.0, 2)

        gates: list[Gate] = []
        for index, point in enumerate(points):
            incoming = _unit(point - points[(index - 1) % len(points)])
            outgoing = _unit(points[(index + 1) % len(points)] - point)
            tangent = incoming + outgoing
            if np.linalg.norm(tangent) < 0.12:
                tangent = outgoing
            tangent = _unit(tangent)
            tangent = _yaw_rotate(tangent, float(normal_offsets[index]))
            tangent = _unit(tangent)
            up = np.asarray([0.0, 0.0, 1.0])
            up -= tangent * float(up @ tangent)
            if np.linalg.norm(up) < 1.0e-6:
                up = np.asarray([0.0, 1.0, 0.0])
                up -= tangent * float(up @ tangent)
            up = _rotate_about_axis(_unit(up), tangent, float(rolls[index]))
            size = np.asarray(sizes[index], np.float64) * rng.uniform(0.96, 1.04, 2)
            gates.append(Gate(
                position=point.astype(np.float32),
                quaternion_wxyz=forward_up_quaternion(tangent, up),
                size=size.astype(np.float32),
                name=f"gate_{index:02d}_{primitive_labels[index]}",
            ))
        margin = float(cfg.bounds_margin_m)
        lower = points.min(axis=0) - margin
        upper = points.max(axis=0) + margin
        lower[2] = 0.0
        upper[2] = max(float(upper[2]), cfg.maximum_altitude_m + 1.0)
        return Track(
            name=name,
            gates=tuple(gates),
            bounds=np.stack([lower, upper], axis=1).astype(np.float32),
            loop=True,
            metadata={
                "source": "flight-mimic-composition-v1",
                "scientific_scope": "generator-only-no-real-course-input",
                "backend": "flight_mimic_composition",
                "split": split,
                "seed": int(seed),
                "program": program.to_mapping(),
                "macro_labels": [item.kind.value for item in program.maneuvers],
                "expanded_macro_indices": macro_indices,
                "primitive_labels": primitive_labels,
                "source_segment_ids": [item.segment_id for item in segments],
                "source_tracks": sorted({item.source_track for item in segments}),
                "trajectory_evidenced_fraction": float(np.mean([
                    item.trajectory_evidenced for item in segments
                ])),
                "boundary_speed_mismatch_mean_mps": float(np.mean([
                    abs(left.exit_speed_mps - right.entry_speed_mps)
                    for left, right in zip(segments, segments[1:] + segments[:1])
                ])),
                "closure_correction_fraction": correction,
                "action_label_contract": "must-requalify-and-query-mpcc-no-warped-ctbr",
            },
        )


def compose_valid_candidates(
    library: FlightSegmentLibrary,
    *,
    count: int,
    seed: int,
    config: RacingDistributionConfig | None = None,
    cache_directory: str | Path | None = None,
    candidate_multiplier: int = 8,
) -> tuple[tuple[Track, TrackValidationReport], ...]:
    cfg = config or RacingDistributionConfig()
    grammar = ManeuverGrammar(cfg)
    composer = FlightMimicComposer(library, cfg)
    validator = RacingDistributionValidator(cfg, cache_directory=cache_directory)
    programs = grammar.coverage_programs(
        count=max(count * candidate_multiplier, count + 8), seed=seed, split="train"
    )
    valid: list[tuple[Track, TrackValidationReport]] = []
    seen: set[str] = set()
    rejected: Counter[str] = Counter()
    for index, program in enumerate(programs):
        track_seed = int(seed + 104729 * (index + 1))
        try:
            track = composer.compose(
                program=program, seed=track_seed, split="train",
                name=f"flight_mimic_train_{index:04d}_{track_seed}",
            )
            report = validator.validate_track(track)
        except (FloatingPointError, RuntimeError, ValueError) as error:
            rejected[f"compose:{type(error).__name__}:{error}"] += 1
            continue
        fingerprint = geometry_fingerprint(track)
        if report.valid and fingerprint not in seen:
            valid.append((track, report))
            seen.add(fingerprint)
        elif fingerprint in seen:
            rejected["duplicate_geometry"] += 1
        else:
            audit = dict(report.audit)
            reasons = audit.get("reasons") or audit.get("failures") or ()
            if reasons:
                for reason in reasons:
                    rejected[f"validate:{reason}"] += 1
            else:
                rejected["validate:unspecified"] += 1
        if len(valid) >= count:
            break
    if len(valid) < count:
        summary = ", ".join(
            f"{key}={value}" for key, value in rejected.most_common(12)
        )
        raise RuntimeError(
            f"only composed {len(valid)} valid tracks for {count} requested; {summary}"
        )
    return tuple(valid[:count])


def compose_valid_programs(
    library: FlightSegmentLibrary,
    programs: Sequence[CourseProgram],
    *,
    seed: int,
    config: RacingDistributionConfig | None = None,
    cache_directory: str | Path | None = None,
    attempts_per_program: int = 16,
) -> tuple[tuple[Track, TrackValidationReport], ...]:
    """Compose one valid augmentation for every fixed source program.

    This preserves the maneuver grammar and task complexity of the matched
    whole-course arm. Only local source segments and bounded symmetry
    transforms vary; MPCC qualification remains mandatory afterward.
    """

    cfg = config or RacingDistributionConfig()
    composer = FlightMimicComposer(library, cfg)
    validator = RacingDistributionValidator(cfg, cache_directory=cache_directory)
    valid: list[tuple[Track, TrackValidationReport]] = []
    seen: set[str] = set()
    failures: Counter[str] = Counter()
    for program_index, program in enumerate(programs):
        accepted: tuple[Track, TrackValidationReport] | None = None
        for attempt in range(max(int(attempts_per_program), 1)):
            track_seed = int(seed + 104729 * (program_index + 1) + 7919 * attempt)
            try:
                track = composer.compose(
                    program=program, seed=track_seed, split="train",
                    name=(
                        f"flight_mimic_train_{program_index:04d}_"
                        f"{attempt:02d}_{track_seed}"
                    ),
                )
                report = validator.validate_track(track)
            except (FloatingPointError, RuntimeError, ValueError) as error:
                failures[f"compose:{type(error).__name__}:{error}"] += 1
                continue
            fingerprint = geometry_fingerprint(track)
            if report.valid and fingerprint not in seen:
                accepted = (track, report)
                seen.add(fingerprint)
                break
            reasons = dict(report.audit).get("reasons") or ("unspecified",)
            for reason in reasons:
                failures[f"validate:{reason}"] += 1
        if accepted is None:
            summary = ", ".join(
                f"{key}={value}" for key, value in failures.most_common(10)
            )
            raise RuntimeError(
                f"could not compose source program {program_index} after "
                f"{attempts_per_program} attempts; {summary}"
            )
        valid.append(accepted)
    return tuple(valid)
