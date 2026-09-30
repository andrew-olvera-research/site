"""Materialize validated racing distributions for BC, DAgger, and PPO."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..procedural_tracks import geometry_fingerprint, read_manifest, save_track_yaml
from ..tracks import load_track
from .archive import DescriptorGrid, DYNAMIC_ARCHIVE_FIELDS, STATIC_ARCHIVE_FIELDS
from .descriptors import CourseDescriptor, descriptor_distance
from .generator import RacingTaskGenerator
from .grammar import (
    COMPOSITION_HOLDOUT_TRIGRAMS, ManeuverGrammar, REQUIRED_BIGRAMS,
    REQUIRED_TRIGRAMS, required_ngrams,
)
from .schema import GeneratorBackend, RacingDistributionConfig
from .validator import RacingDistributionValidator, TrackValidationReport


MANIFEST_SCHEMA = "starscream-procedural-track-manifest-v1"
ENGINE_VERSION = "zero-shot-racing-distribution-v1"


_QUALIFICATION_EVIDENCE_FIELDS = (
    "qualified_speed_mps",
    "qualification",
    "qualification_regimes",
    "qualification_suite",
)


def _materialized_geometry_equivalent(left: Any, right: Any) -> bool:
    """Compare track geometry across harmless float32 YAML re-normalization."""

    if bool(left.loop) != bool(right.loop) or len(left.gates) != len(right.gates):
        return False
    if not np.allclose(left.bounds, right.bounds, rtol=0.0, atol=5.0e-7):
        return False
    for left_gate, right_gate in zip(left.gates, right.gates):
        if (
            left_gate.name != right_gate.name
            or left_gate.kind != right_gate.kind
            or bool(left_gate.render) != bool(right_gate.render)
            or bool(left_gate.enter_from_opposite_side)
            != bool(right_gate.enter_from_opposite_side)
        ):
            return False
        if not np.allclose(
            left_gate.position, right_gate.position, rtol=0.0, atol=5.0e-7,
        ) or not np.allclose(
            left_gate.size, right_gate.size, rtol=0.0, atol=5.0e-7,
        ):
            return False
        left_quaternion = np.asarray(left_gate.quaternion_wxyz, dtype=np.float64)
        right_quaternion = np.asarray(right_gate.quaternion_wxyz, dtype=np.float64)
        left_quaternion /= max(float(np.linalg.norm(left_quaternion)), 1.0e-12)
        right_quaternion /= max(float(np.linalg.norm(right_quaternion)), 1.0e-12)
        # q and -q encode the same orientation. The tolerance only admits the
        # one-ULP drift introduced by a second float32 normalization pass.
        if 1.0 - abs(float(np.dot(left_quaternion, right_quaternion))) > 5.0e-12:
            return False
    return True


def _passes_qualification_contract(
    record: Mapping[str, Any], required_regimes: Sequence[str],
) -> bool:
    """Return whether one record has complete conservative regime evidence."""

    required = tuple(str(item) for item in required_regimes)
    if not required:
        return bool(
            record.get("qualified_speed_mps") is not None
            and record.get("qualification") is not None
        )
    regimes = dict(record.get("qualification_regimes") or {})
    if any(
        regime not in regimes
        or not bool(regimes[regime].get("passed"))
        or regimes[regime].get("qualified_speed_mps") is None
        for regime in required
    ):
        return False
    speeds = [float(regimes[regime]["qualified_speed_mps"]) for regime in required]
    suite = dict(record.get("qualification_suite") or {})
    return bool(
        suite.get("passed")
        and record.get("qualification") is not None
        and record.get("qualified_speed_mps") is not None
        and np.isclose(float(record["qualified_speed_mps"]), min(speeds), atol=1.0e-9)
    )


def merge_qualification_evidence(
    source: str | Path, targets: Sequence[str | Path],
) -> Mapping[str, Any]:
    """Copy completed qualification evidence back to candidate archives.

    QD selection may be followed by a more expensive admission suite. This
    helper lets a subsequent strict re-selection reuse those completed jobs
    rather than re-running every selected track. Geometry fingerprints are
    checked before any evidence is copied.
    """

    source_path = Path(source)
    source_payload = load_distribution_manifest(source_path)
    evidence = {
        str(record["name"]): record
        for record in source_payload["records"]
        if record.get("qualification_regimes")
    }
    reports: list[dict[str, Any]] = []
    for target_value in targets:
        target_path = Path(target_value)
        target_payload = dict(load_distribution_manifest(target_path))
        merged = 0
        for record in target_payload["records"]:
            source_record = evidence.get(str(record["name"]))
            if source_record is None:
                continue
            if str(record.get("geometry_fingerprint")) != str(
                source_record.get("geometry_fingerprint")
            ):
                source_track = load_track(
                    source_path.parent / str(source_record["path"])
                )
                target_track = load_track(target_path.parent / str(record["path"]))
                if not _materialized_geometry_equivalent(source_track, target_track):
                    raise ValueError(
                        f"qualification evidence geometry mismatch for {record['name']!r}"
                    )
            for field in _QUALIFICATION_EVIDENCE_FIELDS:
                if field in source_record:
                    record[field] = source_record[field]
            merged += 1
        temporary = target_path.with_suffix(target_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(target_payload, indent=2, sort_keys=True), encoding="utf-8",
        )
        temporary.replace(target_path)
        reports.append({"manifest": str(target_path), "merged_records": merged})
    return {
        "source": str(source_path),
        "source_records_with_evidence": len(evidence),
        "targets": reports,
    }


def _candidate_quality(report: TrackValidationReport) -> float:
    audit = report.audit
    descriptor = report.descriptor
    if not report.valid or descriptor is None:
        return -1.0e9
    clearance = min(float(audit["minimum_route_clearance_m"]), 4.0)
    spacing = min(float(audit["minimum_nonadjacent_spacing_m"]), 5.0)
    curvature_margin = max(0.0, 1.2 - float(audit["p99_curvature_m_inv"]))
    dynamic_quality = 0.0
    if descriptor.dynamic_available:
        dynamic_quality = (
            0.30 * float(descriptor.values["teacher_average_speed_mps"])
            - 8.0 * float(descriptor.values["teacher_solver_failure_fraction"])
            - 5.0 * float(descriptor.values["teacher_recovery_fraction"])
        )
    return float(2.0 * clearance + 0.35 * spacing + curvature_margin + dynamic_quality)


def _select_diverse(
    candidates: Sequence[tuple[Any, TrackValidationReport, int]],
    *,
    count: int,
    config: RacingDistributionConfig,
    obligations_override: Iterable[tuple[str, ...]] | None = None,
    ngram_order_override: int | None = None,
    grid_fields: Sequence[str] | None = None,
) -> list[tuple[Any, TrackValidationReport, int]]:
    """Greedy MAP-Elites-style selection with topology and novelty bonuses."""

    if len(candidates) < count:
        raise RuntimeError(f"only {len(candidates)} valid candidates for {count} requested tracks")
    grid = DescriptorGrid(config, fields=grid_fields)
    order = int(ngram_order_override or config.required_ngram_order)
    obligations = set(obligations_override or required_ngrams(order))
    selected: list[tuple[Any, TrackValidationReport, int]] = []
    remaining = list(candidates)
    occupied: set[tuple[int, ...]] = set()
    covered: set[tuple[str, ...]] = set()
    while remaining and len(selected) < count:
        def score(item: tuple[Any, TrackValidationReport, int]) -> tuple[float, float, str]:
            track, report, _ = item
            assert report.descriptor is not None
            descriptor = report.descriptor
            grams = (
                set(descriptor.primitive_trigrams)
                if order == 3
                else set(descriptor.primitive_bigrams)
            )
            topology_gain = len((grams & obligations) - covered)
            cell_gain = int(grid.key(descriptor) not in occupied)
            novelty = min(
                (descriptor_distance(descriptor, chosen[1].descriptor) for chosen in selected),
                default=2.0,
            )
            value = (
                12.0 * topology_gain + 5.0 * cell_gain
                + 1.5 * min(novelty, 3.0) + _candidate_quality(report)
            )
            return float(value), _candidate_quality(report), track.name
        best = max(remaining, key=score)
        remaining.remove(best)
        selected.append(best)
        descriptor = best[1].descriptor
        assert descriptor is not None
        occupied.add(grid.key(descriptor))
        covered.update(
            descriptor.primitive_trigrams
            if order == 3 else descriptor.primitive_bigrams
        )
    return selected


def _candidate_programs(
    backend: GeneratorBackend,
    *,
    count: int,
    seed: int,
    split: str,
    grammar: ManeuverGrammar,
) -> Sequence[Any | None]:
    if backend == GeneratorBackend.MANEUVER_GRAMMAR:
        return grammar.coverage_programs(count=count, seed=seed, split=split)
    return (None,) * count


def generate_distribution_manifest(
    output_directory: str | Path,
    *,
    backend: GeneratorBackend | str,
    train_count: int = 64,
    validation_count: int = 24,
    composition_holdout_count: int = 8,
    seed: int = 20260825,
    candidate_multiplier: int = 4,
    config: RacingDistributionConfig | None = None,
    forbidden_geometry_fingerprints: Iterable[str] = (),
) -> Path:
    """Generate, validate, select, and atomically materialize one distribution.

    Benchmark geometry is not accepted by this API.  Only opaque forbidden
    fingerprints may cross the seal boundary, and only for leakage rejection.
    """

    cfg = config or RacingDistributionConfig()
    selected_backend = GeneratorBackend(backend)
    if min(train_count, validation_count, candidate_multiplier) < 1 or composition_holdout_count < 0:
        raise ValueError("track counts must be nonnegative and primary counts positive")
    root = Path(output_directory)
    track_root = root / "tracks"
    cache_root = root / "racing-lines"
    root.mkdir(parents=True, exist_ok=True)
    generator = RacingTaskGenerator(cfg)
    validator = RacingDistributionValidator(cfg, cache_directory=cache_root)
    grammar = ManeuverGrammar(cfg)
    forbidden = set(str(item) for item in forbidden_geometry_fingerprints)
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    generation_summary: dict[str, Any] = {}
    for split, requested, offset in (
        ("train", train_count, 0),
        ("validation", validation_count, 20_000_000),
        ("composition_holdout", composition_holdout_count, 40_000_000),
    ):
        if requested == 0:
            continue
        target_candidates = max(requested * candidate_multiplier, requested + 8)
        programs = _candidate_programs(
            selected_backend, count=target_candidates,
            seed=seed + offset, split=split, grammar=grammar,
        )
        candidates: list[tuple[Any, TrackValidationReport, int]] = []
        failures: Counter[str] = Counter()
        attempts = 0
        program_index = 0
        maximum_attempts = max(cfg.generation_attempts, target_candidates * 12)
        composition_obligations = {
            tuple(item.value for item in gram) for gram in COMPOSITION_HOLDOUT_TRIGRAMS
        }
        candidate_compositions: set[tuple[str, ...]] = set()
        while (
            len(candidates) < target_candidates
            or (
                split == "composition_holdout"
                and not composition_obligations <= candidate_compositions
            )
        ) and attempts < maximum_attempts:
            attempts += 1
            track_seed = int(seed + offset + attempts * 104729)
            program = programs[program_index] if program_index < len(programs) else None
            program_index += 1
            if selected_backend == GeneratorBackend.MANEUVER_GRAMMAR and program is None:
                forced_pool = tuple(sorted(
                    (
                        COMPOSITION_HOLDOUT_TRIGRAMS
                        if split == "composition_holdout"
                        else REQUIRED_TRIGRAMS
                        if cfg.required_ngram_order == 3
                        else REQUIRED_BIGRAMS
                    ),
                    key=lambda gram: tuple(item.value for item in gram),
                ))
                forced = forced_pool[(program_index - len(programs) - 1) % len(forced_pool)]
                program = grammar.sample_program(
                    seed=track_seed, split=split, forced_ngram=forced,
                )
            name = f"zsr_{selected_backend.value}_{split}_{attempts:04d}_{track_seed}"
            try:
                track = generator.generate(
                    backend=selected_backend, seed=track_seed, split=split,
                    name=name, program=program,
                )
                report = validator.validate_track(track)
            except (FloatingPointError, RuntimeError, ValueError) as error:
                failures[f"generation:{type(error).__name__}"] += 1
                continue
            fingerprint = geometry_fingerprint(track)
            if fingerprint in seen or fingerprint in forbidden:
                failures["geometry-duplicate-or-forbidden"] += 1
                continue
            if not report.valid:
                failures.update(report.reasons)
                continue
            seen.add(fingerprint)
            candidates.append((track, report, int((track.metadata or {})["seed"])))
            assert report.descriptor is not None
            candidate_compositions.update(
                set(report.descriptor.primitive_trigrams) & composition_obligations
            )
        if (
            split == "composition_holdout"
            and not composition_obligations <= candidate_compositions
        ):
            missing = sorted(composition_obligations - candidate_compositions)
            raise RuntimeError(
                f"could not realize all composition-holdout obligations; missing={missing}"
            )
        chosen = _select_diverse(
            candidates, count=requested, config=cfg,
            obligations_override=(
                composition_obligations if split == "composition_holdout" else None
            ),
            ngram_order_override=(3 if split == "composition_holdout" else None),
        )
        generation_summary[split] = {
            "attempts": attempts,
            "valid_candidates": len(candidates),
            "selected": len(chosen),
            "candidate_acceptance_yield": len(candidates) / max(attempts, 1),
            "rejections": dict(sorted(failures.items())),
        }
        for track, report, track_seed in chosen:
            path = save_track_yaml(track, track_root / split / f"{track.name}.yaml")
            # Fingerprints are contracts over the materialized float32 artifact,
            # not over the transient object.  A second quaternion normalization
            # during YAML loading can alter a final float32 bit.
            materialized = load_track(path)
            descriptor = report.descriptor
            assert descriptor is not None
            static_audit = dict(report.audit)
            static_audit["track_fingerprint"] = materialized.fingerprint
            static_audit["geometry_fingerprint"] = geometry_fingerprint(materialized)
            records.append({
                "name": materialized.name,
                "path": str(path.relative_to(root)),
                "split": split,
                "family": selected_backend.value,
                "backend": selected_backend.value,
                "seed": track_seed,
                "track_fingerprint": materialized.fingerprint,
                "geometry_fingerprint": geometry_fingerprint(materialized),
                "static_audit": static_audit,
                "descriptor": descriptor.to_mapping(),
                "qd_cell": DescriptorGrid(cfg).key_mapping(DescriptorGrid(cfg).key(descriptor)),
                "qualified_speed_mps": None,
                "qualification": None,
            })
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "engine": ENGINE_VERSION,
        "scientific_scope": "generator-only-no-real-course-input",
        "backend": selected_backend.value,
        "seed": int(seed),
        "config": cfg.to_mapping(),
        "family_weights": {selected_backend.value: 1.0},
        "generation": generation_summary,
        "records": records,
    }
    destination = root / "manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_distribution_manifest(path: str | Path) -> Mapping[str, Any]:
    payload = read_manifest(path)
    if payload.get("engine") != ENGINE_VERSION:
        raise ValueError(f"manifest is not a {ENGINE_VERSION} distribution")
    if payload.get("scientific_scope") != "generator-only-no-real-course-input":
        raise ValueError("distribution manifest violates source-provenance contract")
    return payload


def freeze_distribution_manifest(
    source: str | Path,
    output: str | Path,
    *,
    required_qualification_regimes: Sequence[str] = (
        "nominal_multistart", "randomized_multistart",
    ),
) -> Path:
    """Write an immutable-by-contract learner manifest after teacher admission.

    Collection can consume this artifact with ``--no-manifest-update``.  The
    adjacent lock records a byte hash, while ``content_sha256`` protects the
    geometry, split, and complete qualification evidence from accidental drift.
    """

    source_path = Path(source)
    payload = dict(load_distribution_manifest(source_path))
    records = [dict(item) for item in payload["records"]]
    required = tuple(str(item) for item in required_qualification_regimes)
    if not required or len(required) != len(set(required)):
        raise ValueError("required qualification regimes must be unique and nonempty")
    errors: list[str] = []
    for record in records:
        name = str(record["name"])
        regimes = dict(record.get("qualification_regimes") or {})
        missing = sorted(set(required) - set(regimes))
        if missing:
            errors.append(f"{name}: missing qualification regimes {missing}")
        failed = sorted(
            regime for regime in required
            if regime in regimes and not bool(regimes[regime].get("passed"))
        )
        if failed:
            errors.append(f"{name}: failed qualification regimes {failed}")
        speeds = [
            float(regimes[regime]["qualified_speed_mps"])
            for regime in required
            if regime in regimes and regimes[regime].get("qualified_speed_mps") is not None
        ]
        if len(speeds) != len(required):
            errors.append(f"{name}: incomplete qualified-speed evidence")
        elif record.get("qualified_speed_mps") is None or not np.isclose(
            float(record["qualified_speed_mps"]), min(speeds), atol=1.0e-9,
        ):
            errors.append(f"{name}: conservative speed is not the regime minimum")
        if not bool(dict(record.get("qualification_suite") or {}).get("passed")):
            errors.append(f"{name}: aggregate qualification suite did not pass")
        record.pop("collection", None)
    if errors:
        raise RuntimeError("distribution cannot be frozen: " + "; ".join(errors[:12]))
    payload["records"] = records
    payload["frozen"] = {
        "contract": "starscream-racing-distribution-freeze-v1",
        "required_qualification_regimes": list(required),
        "record_count": len(records),
    }
    protected = {
        "engine": payload.get("engine"),
        "scientific_scope": payload.get("scientific_scope"),
        "config": payload.get("config"),
        "records": records,
    }
    content_sha = hashlib.sha256(json.dumps(
        protected, sort_keys=True, separators=(",", ":"), allow_nan=True,
    ).encode("utf-8")).hexdigest()
    payload["frozen"]["content_sha256"] = content_sha
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    file_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
    lock_path = destination.with_suffix(destination.suffix + ".lock.json")
    lock_path.write_text(json.dumps({
        "contract": "starscream-racing-distribution-freeze-lock-v1",
        "manifest": destination.name,
        "file_sha256": file_sha,
        "content_sha256": content_sha,
        "record_count": len(records),
        "required_qualification_regimes": list(required),
    }, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def _backend_quotas(total: int, backends: Sequence[str], spline_fraction: float) -> dict[str, int]:
    if not 0.0 <= spline_fraction <= 1.0:
        raise ValueError("spline_fraction must be in [0, 1]")
    spline = GeneratorBackend.INFORMED_SPLINE.value
    grammar = GeneratorBackend.MANEUVER_GRAMMAR.value
    available = set(backends)
    if available == {spline, grammar}:
        spline_count = int(round(total * spline_fraction))
        if total >= 2 and 0.0 < spline_fraction < 1.0:
            spline_count = int(np.clip(spline_count, 1, total - 1))
        return {spline: spline_count, grammar: total - spline_count}
    if len(available) != 1:
        raise ValueError(f"unsupported qualified-union backends: {sorted(available)}")
    return {next(iter(available)): total}


def select_qualified_distribution_manifest(
    manifests: Sequence[str | Path],
    output_directory: str | Path,
    *,
    train_count: int,
    validation_count: int,
    composition_holdout_count: int = 0,
    spline_fraction: float = 0.20,
    required_qualification_regimes: Sequence[str] = (),
) -> Path:
    """Select a teacher-feasible backend union in dynamic descriptor space."""

    if not manifests:
        raise ValueError("qualified union needs at least one source manifest")
    if min(train_count, validation_count) < 1 or composition_holdout_count < 0:
        raise ValueError("invalid qualified-union split counts")
    source_paths = [Path(item) for item in manifests]
    payloads = [load_distribution_manifest(path) for path in source_paths]
    config_mapping = dict(payloads[0]["config"])
    if any(dict(payload["config"]) != config_mapping for payload in payloads[1:]):
        raise ValueError("qualified-union source manifests use different generator contracts")
    config = RacingDistributionConfig(**config_mapping)
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    validator = RacingDistributionValidator(config, cache_directory=root / "racing-lines")
    pools: dict[str, list[tuple[Any, TrackValidationReport, int]]] = {
        "train": [], "validation": [], "composition_holdout": [],
    }
    source_records: dict[str, Mapping[str, Any]] = {}
    source_manifest_by_name: dict[str, Path] = {}
    for source_path, payload in zip(source_paths, payloads):
        for record in payload["records"]:
            if not _passes_qualification_contract(
                record, required_qualification_regimes,
            ):
                continue
            split = str(record["split"])
            if split not in pools:
                continue
            track = load_track(source_path.parent / str(record["path"]))
            report = validator.validate_track(track, qualification=record["qualification"])
            if not report.valid or report.descriptor is None or not report.descriptor.dynamic_available:
                continue
            if track.name in source_records:
                raise ValueError(f"qualified-union duplicate task name {track.name!r}")
            pools[split].append((track, report, int(record["seed"])))
            source_records[track.name] = record
            source_manifest_by_name[track.name] = source_path

    requested_counts = {
        "train": train_count,
        "validation": validation_count,
        "composition_holdout": composition_holdout_count,
    }
    selected: dict[str, list[tuple[Any, TrackValidationReport, int]]] = {}
    dynamic_fields = STATIC_ARCHIVE_FIELDS + DYNAMIC_ARCHIVE_FIELDS
    composition_obligations = {
        tuple(item.value for item in gram) for gram in COMPOSITION_HOLDOUT_TRIGRAMS
    }
    qualified_composition_grams = {
        gram
        for _, report, _ in pools["composition_holdout"]
        if report.descriptor is not None
        for gram in report.descriptor.primitive_trigrams
    }
    if composition_holdout_count and not (
        composition_obligations <= qualified_composition_grams
    ):
        missing = sorted(composition_obligations - qualified_composition_grams)
        raise RuntimeError(
            "qualified candidate pool cannot satisfy composition-holdout "
            f"obligations; missing={missing}"
        )
    for split, count in requested_counts.items():
        if count == 0:
            selected[split] = []
            continue
        available_backends = sorted({
            str((item[0].metadata or {})["backend"]) for item in pools[split]
        })
        quotas = _backend_quotas(
            count, available_backends,
            0.0 if split == "composition_holdout" else spline_fraction,
        )
        chosen: list[tuple[Any, TrackValidationReport, int]] = []
        for backend, quota in quotas.items():
            if quota == 0:
                continue
            candidates = [
                item for item in pools[split]
                if str((item[0].metadata or {})["backend"]) == backend
            ]
            chosen.extend(_select_diverse(
                candidates, count=quota, config=config,
                obligations_override=(
                    composition_obligations if split == "composition_holdout" else None
                ),
                ngram_order_override=(3 if split == "composition_holdout" else None),
                grid_fields=dynamic_fields,
            ))
        selected[split] = chosen

    selected_composition_grams = {
        gram
        for _, report, _ in selected["composition_holdout"]
        if report.descriptor is not None
        for gram in report.descriptor.primitive_trigrams
    }
    if composition_holdout_count and not (
        composition_obligations <= selected_composition_grams
    ):
        missing = sorted(composition_obligations - selected_composition_grams)
        raise RuntimeError(
            "strict selection failed to preserve composition-holdout "
            f"obligations; missing={missing}"
        )

    records: list[dict[str, Any]] = []
    grid = DescriptorGrid(config, fields=dynamic_fields)
    for split in ("train", "validation", "composition_holdout"):
        for track, report, seed in selected[split]:
            destination = save_track_yaml(
                track, root / "tracks" / split / f"{track.name}.yaml",
            )
            materialized = load_track(destination)
            descriptor = report.descriptor
            assert descriptor is not None
            source = source_records[track.name]
            audit = dict(report.audit)
            audit["track_fingerprint"] = materialized.fingerprint
            audit["geometry_fingerprint"] = geometry_fingerprint(materialized)
            selected_record = {
                "name": materialized.name,
                "path": str(destination.relative_to(root)),
                "split": split,
                "family": str((track.metadata or {})["backend"]),
                "backend": str((track.metadata or {})["backend"]),
                "seed": seed,
                "track_fingerprint": materialized.fingerprint,
                "geometry_fingerprint": geometry_fingerprint(materialized),
                "static_audit": audit,
                "descriptor": descriptor.to_mapping(),
                "qd_cell": grid.key_mapping(grid.key(descriptor)),
                "qualified_speed_mps": float(source["qualified_speed_mps"]),
                "qualification": source["qualification"],
                "source_manifest": str(source_manifest_by_name[track.name].resolve()),
            }
            for field in ("qualification_regimes", "qualification_suite"):
                if field in source:
                    selected_record[field] = source[field]
            records.append(selected_record)
    backend_counts = Counter(str(item["backend"]) for item in records)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "engine": ENGINE_VERSION,
        "scientific_scope": "generator-only-no-real-course-input",
        "backend": "qualified_qd_union",
        "seed": None,
        "config": config.to_mapping(),
        "family_weights": {
            name: count / max(len(records), 1) for name, count in sorted(backend_counts.items())
        },
        "generation": {
            "method": (
                "strict-qualified-dynamic-qd-union-v1"
                if required_qualification_regimes
                else "qualified-dynamic-qd-union-v1"
            ),
            "source_manifests": [str(path.resolve()) for path in source_paths],
            "spline_fraction": float(spline_fraction),
            "required_qualification_regimes": [
                str(item) for item in required_qualification_regimes
            ],
        },
        "records": records,
    }
    destination = root / "manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return destination
