"""Static, distributional, and artifact validation for generated racing tasks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..procedural_tracks import geometry_fingerprint
from ..tracks import Track, load_track
from .archive import DescriptorGrid
from .descriptors import CourseDescriptor, course_descriptor
from .grammar import COMPOSITION_HOLDOUT_TRIGRAMS, required_ngrams
from .schema import GeneratorBackend, HARD_MANEUVERS, RacingDistributionConfig, VERTICAL_MANEUVERS


def _segment_distance(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    """Shortest distance between two finite 3-D line segments."""

    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa, bb, cc = float(u @ u), float(u @ v), float(v @ v)
    dd, ee = float(u @ w), float(v @ w)
    denominator = aa * cc - bb * bb
    small = 1.0e-12
    if denominator < small:
        s_num, s_den = 0.0, 1.0
        t_num, t_den = ee, cc
    else:
        s_num, s_den = bb * ee - cc * dd, denominator
        t_num, t_den = aa * ee - bb * dd, denominator
        if s_num < 0.0:
            s_num, t_num, t_den = 0.0, ee, cc
        elif s_num > s_den:
            s_num, t_num, t_den = s_den, ee + bb, cc
    if t_num < 0.0:
        t_num = 0.0
        if -dd < 0.0:
            s_num = 0.0
        elif -dd > aa:
            s_num, s_den = s_den, s_den
        else:
            s_num, s_den = -dd, aa
    elif t_num > t_den:
        t_num = t_den
        if -dd + bb < 0.0:
            s_num = 0.0
        elif -dd + bb > aa:
            s_num, s_den = s_den, s_den
        else:
            s_num, s_den = -dd + bb, aa
    sc = 0.0 if abs(s_num) < small else s_num / max(s_den, small)
    tc = 0.0 if abs(t_num) < small else t_num / max(t_den, small)
    return float(np.linalg.norm(w + sc * u - tc * v))


def _nonadjacent_distances(points: np.ndarray) -> tuple[list[float], list[float]]:
    gate_distances: list[float] = []
    route_distances: list[float] = []
    count = len(points)
    for left in range(count):
        for right in range(left + 1, count):
            cyclic = min(right - left, count - (right - left))
            if cyclic > 1:
                gate_distances.append(float(np.linalg.norm(points[left] - points[right])))
            # Segment i connects i to i+1. Adjacent segments share a vertex.
            segment_cyclic = min(right - left, count - (right - left))
            if segment_cyclic > 1:
                route_distances.append(_segment_distance(
                    points[left], points[(left + 1) % count],
                    points[right], points[(right + 1) % count],
                ))
    return gate_distances, route_distances


@dataclass(frozen=True, slots=True)
class TrackValidationReport:
    name: str
    valid: bool
    reasons: tuple[str, ...]
    audit: Mapping[str, Any]
    descriptor: CourseDescriptor | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "valid": bool(self.valid),
            "reasons": list(self.reasons),
            "audit": dict(self.audit),
            "descriptor": None if self.descriptor is None else self.descriptor.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class DistributionValidationReport:
    passed: bool
    reasons: tuple[str, ...]
    metrics: Mapping[str, Any]
    track_reports: tuple[TrackValidationReport, ...]

    def to_mapping(self, *, include_track_reports: bool = True) -> dict[str, Any]:
        result = {
            "passed": bool(self.passed),
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }
        if include_track_reports:
            result["track_reports"] = [item.to_mapping() for item in self.track_reports]
        return result


class RacingDistributionValidator:
    """Gatekeeper between candidate generation and any learner-facing manifest."""

    def __init__(
        self,
        config: RacingDistributionConfig | None = None,
        *,
        cache_directory: str | Path | None = None,
    ) -> None:
        self.config = config or RacingDistributionConfig()
        self.cache_directory = None if cache_directory is None else Path(cache_directory)

    def validate_track(
        self,
        track: Track,
        *,
        qualification: Mapping[str, Any] | None = None,
    ) -> TrackValidationReport:
        from ...mpcc import RacingLinePlanner, RacingLinePlannerConfig

        cfg = self.config
        reasons: list[str] = []
        points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
        geometry = track.geometry_report(minimum_alignment=0.12)
        segments = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        nonadjacent_gates, nonadjacent_route = _nonadjacent_distances(points)
        if not geometry["feasible"]:
            reasons.append("gate-geometry")
        if not cfg.minimum_gates <= len(track.gates) <= cfg.maximum_gates:
            reasons.append("gate-count")
        if float(segments.min()) < cfg.minimum_gate_spacing_m:
            reasons.append("gate-spacing-minimum")
        if float(segments.max()) > cfg.maximum_gate_spacing_m:
            reasons.append("gate-spacing-maximum")
        if nonadjacent_gates and min(nonadjacent_gates) < cfg.minimum_nonadjacent_gate_spacing_m:
            reasons.append("nonadjacent-gate-overlap")
        if nonadjacent_route and min(nonadjacent_route) < cfg.minimum_route_clearance_m:
            reasons.append("route-self-clearance")
        if float(points[:, 2].min()) < cfg.minimum_altitude_m - 1.0e-5:
            reasons.append("altitude-minimum")
        if float(points[:, 2].max()) > cfg.maximum_altitude_m + 1.0e-5:
            reasons.append("altitude-maximum")
        if any(gate.kind != "gate" or not gate.render for gate in track.gates):
            reasons.append("nonphysical-route-checkpoint")
        if float(np.max(0.5 * np.ptp(track.bounds[:2], axis=1))) > cfg.bounds_xy_m[1] + 1.0e-5:
            reasons.append("horizontal-flight-volume")

        line = None
        curvature = np.asarray([np.inf], np.float64)
        try:
            line = RacingLinePlanner(RacingLinePlannerConfig(
                sample_count=cfg.racing_line_samples,
                offset_iterations=cfg.racing_line_offset_iterations,
                cache_directory=(
                    None if self.cache_directory is None else str(self.cache_directory)
                ),
            )).plan(track)
            query = np.linspace(0.0, line.length, 2400, endpoint=False)
            curvature = np.asarray(line.evaluate(query)["curvature"], np.float64)
            if not cfg.minimum_length_m <= float(line.length) <= cfg.maximum_length_m:
                reasons.append("course-length")
            if float(np.quantile(curvature, 0.99)) > cfg.maximum_p99_curvature_m_inv:
                reasons.append("p99-curvature")
            if float(curvature.max()) > cfg.maximum_peak_curvature_m_inv:
                reasons.append("peak-curvature")
        except (FloatingPointError, RuntimeError, ValueError) as error:
            reasons.append(f"racing-line:{type(error).__name__}")

        macros = tuple(str(item).split(":", 1)[0] for item in (track.metadata or {}).get(
            "macro_labels", (track.metadata or {}).get("primitive_labels", ())
        ))
        vertical_names = {item.value for item in VERTICAL_MANEUVERS}
        hard_names = {item.value for item in HARD_MANEUVERS}
        vertical_expected = bool(set(macros) & vertical_names)
        directions = np.roll(points, -1, axis=0) - points
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-12)
        local_turn = np.degrees(np.arccos(np.clip(
            np.sum(directions * np.roll(directions, 1, axis=0), axis=1), -1.0, 1.0,
        )))
        local_slope = np.abs(np.roll(points[:, 2], -1) - points[:, 2]) / np.maximum(
            segments, 1.0e-12,
        )
        hard_expected = bool(
            set(macros) & hard_names
            or np.max(local_turn) >= 65.0
            or np.max(local_slope) >= 0.30
        )
        vertical_excursion = float(np.ptp(points[:, 2]))
        up = np.stack([gate.up for gate in track.gates]).astype(np.float64)
        up_span = float(np.degrees(np.max(np.arccos(np.clip(up @ up[0], -1.0, 1.0)))))
        if vertical_expected and vertical_excursion < 1.0:
            reasons.append("semantic-vertical-collapse")
        if any(name.startswith("split_s") for name in macros) and up_span < 90.0:
            reasons.append("semantic-split-s-roll")
        macro_indices = tuple(int(item) for item in (track.metadata or {}).get(
            "expanded_macro_indices", ()
        ))
        macro_labels = tuple(str(item) for item in (track.metadata or {}).get(
            "macro_labels", ()
        ))
        if macro_labels and len(macro_indices) == len(track.gates):
            gate_up = np.stack([gate.up for gate in track.gates]).astype(np.float64)
            signed_turn = np.degrees(np.arctan2(
                np.roll(directions, 1, axis=0)[:, 0] * directions[:, 1]
                - np.roll(directions, 1, axis=0)[:, 1] * directions[:, 0],
                np.sum(np.roll(directions, 1, axis=0)[:, :2] * directions[:, :2], axis=1),
            ))
            for macro_index, label in enumerate(macro_labels):
                selected = np.flatnonzero(np.asarray(macro_indices) == macro_index)
                if not len(selected):
                    reasons.append(f"semantic-empty-{label}")
                    continue
                macro_dz = float(np.sum(directions[selected, 2] * segments[selected]))
                vertical_path = np.r_[0.0, np.cumsum(
                    directions[selected, 2] * segments[selected]
                )]
                macro_vertical_span = float(np.ptp(vertical_path))
                macro_turn = float(np.sum(local_turn[selected]))
                macro_up_span = float(np.degrees(np.max(np.arccos(np.clip(
                    gate_up[selected] @ gate_up[selected[0]], -1.0, 1.0,
                )))))
                failed = False
                if label == "climb":
                    failed = macro_dz < 1.5
                elif label == "dive":
                    failed = macro_dz > -1.4
                elif label.startswith("hairpin"):
                    failed = macro_turn < 70.0
                elif label.startswith("slalom"):
                    failed = not (
                        np.min(signed_turn[selected]) < -5.0
                        and np.max(signed_turn[selected]) > 5.0
                    )
                elif label.startswith("split_s"):
                    failed = (
                        macro_vertical_span < 1.5 or macro_turn < 120.0
                        or macro_up_span < 110.0
                    )
                elif label.startswith("corkscrew"):
                    failed = (
                        macro_vertical_span < 1.0 or macro_turn < 75.0
                        or macro_up_span < 90.0
                    )
                elif label == "stacked_reversal":
                    failed = (
                        macro_vertical_span < 2.0 or macro_turn < 145.0
                        or macro_up_span < 120.0
                    )
                if failed:
                    reasons.append(f"semantic-{label}")

        descriptor = None
        if line is not None:
            descriptor = course_descriptor(
                track, racing_line=line, qualification=qualification,
            )
        peak_curvature = float(curvature.max())
        p99_curvature = (
            float(np.quantile(curvature, 0.99))
            if np.all(np.isfinite(curvature)) else float("inf")
        )
        audit: dict[str, Any] = {
            "valid": not reasons,
            "reasons": list(dict.fromkeys(reasons)),
            "track": track.name,
            "backend": str((track.metadata or {}).get("backend", "unknown")),
            "track_fingerprint": track.fingerprint,
            "geometry_fingerprint": geometry_fingerprint(track),
            "gate_count": len(track.gates),
            "racing_line_length_m": None if line is None else float(line.length),
            "minimum_gate_spacing_m": float(segments.min()),
            "maximum_gate_spacing_m": float(segments.max()),
            "minimum_nonadjacent_spacing_m": (
                min(nonadjacent_gates) if nonadjacent_gates else float("inf")
            ),
            "minimum_route_clearance_m": (
                min(nonadjacent_route) if nonadjacent_route else float("inf")
            ),
            "vertical_excursion_m": vertical_excursion,
            "gate_up_span_degrees": up_span,
            "maximum_curvature_m_inv": peak_curvature,
            "p99_curvature_m_inv": p99_curvature,
            "vertical_semantics": vertical_expected,
            "hard_semantics": hard_expected,
            "geometry": geometry,
        }
        unique_reasons = tuple(dict.fromkeys(reasons))
        return TrackValidationReport(
            name=track.name, valid=not unique_reasons, reasons=unique_reasons,
            audit=audit, descriptor=descriptor,
        )

    def validate_manifest(
        self,
        path: str | Path,
        *,
        forbidden_geometry_fingerprints: Iterable[str] = (),
        require_dynamic: bool = False,
    ) -> DistributionValidationReport:
        manifest_path = Path(path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = list(payload.get("records") or ())
        reasons: list[str] = []
        reports: list[TrackValidationReport] = []
        artifact_errors: list[str] = []
        forbidden = set(str(item) for item in forbidden_geometry_fingerprints)
        observed_fingerprints: list[str] = []
        split_names = ("train", "validation", "composition_holdout")
        split_fingerprints: dict[str, set[str]] = {name: set() for name in split_names}
        source_violations: list[str] = []
        for record in records:
            name = str(record.get("name", "missing-name"))
            try:
                track = load_track(manifest_path.parent / str(record["path"]))
                fingerprint = geometry_fingerprint(track)
                observed_fingerprints.append(fingerprint)
                split_fingerprints.setdefault(str(record.get("split")), set()).add(fingerprint)
                if track.name != name:
                    artifact_errors.append(f"{name}: YAML name mismatch")
                if record.get("track_fingerprint") != track.fingerprint:
                    artifact_errors.append(f"{name}: track fingerprint mismatch")
                if record.get("geometry_fingerprint") != fingerprint:
                    artifact_errors.append(f"{name}: geometry fingerprint mismatch")
                metadata = dict(track.metadata or {})
                if metadata.get("scientific_scope") != "generator-only-no-real-course-input":
                    source_violations.append(name)
                report = self.validate_track(track, qualification=record.get("qualification"))
                reports.append(report)
            except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                artifact_errors.append(f"{name}: {type(error).__name__}: {error}")
                reports.append(TrackValidationReport(
                    name=name, valid=False, reasons=("artifact-contract",),
                    audit={"vertical_semantics": False, "hard_semantics": False},
                    descriptor=None,
                ))
        if not records:
            reasons.append("empty-manifest")
        if artifact_errors:
            reasons.append("artifact-contract")
        if source_violations:
            reasons.append("source-provenance")
        if len(observed_fingerprints) != len(set(observed_fingerprints)):
            reasons.append("geometry-duplicates")
        split_overlap = {
            f"{left}:{right}": len(split_fingerprints[left] & split_fingerprints[right])
            for left_index, left in enumerate(split_names)
            for right in split_names[left_index + 1:]
        }
        if any(split_overlap.values()):
            reasons.append("split-geometry-leakage")
        forbidden_overlap = set(observed_fingerprints) & forbidden
        if forbidden_overlap:
            reasons.append("sealed-benchmark-leakage")
        if any(not report.valid for report in reports):
            reasons.append("static-validation")

        by_split = {
            split: [
                report for report, record in zip(reports, records)
                if str(record.get("split")) == split and report.descriptor is not None
            ]
            for split in split_names
        }
        train_grams = {
            gram for item in by_split["train"] for gram in item.descriptor.primitive_trigrams
        }
        validation_grams = {
            gram for item in by_split["validation"] for gram in item.descriptor.primitive_trigrams
        }
        obligations = required_ngrams(self.config.required_ngram_order)
        if self.config.required_ngram_order == 2:
            train_grams = {
                gram for item in by_split["train"] for gram in item.descriptor.primitive_bigrams
            }
            validation_grams = {
                gram for item in by_split["validation"] for gram in item.descriptor.primitive_bigrams
            }
        required_coverage = len(train_grams & obligations) / max(len(obligations), 1)
        validation_support = len(validation_grams & train_grams) / max(len(validation_grams), 1)
        train_primitives = {
            label for item in by_split["train"] for label in item.descriptor.primitive_labels
        }
        validation_primitives = {
            label for item in by_split["validation"] for label in item.descriptor.primitive_labels
        }
        validation_primitive_support = len(
            validation_primitives & train_primitives
        ) / max(len(validation_primitives), 1)
        backends = {str(record.get("backend", record.get("family", ""))) for record in records}
        grammar_present = GeneratorBackend.MANEUVER_GRAMMAR.value in backends
        composition_grams = {
            gram for item in by_split["composition_holdout"]
            for gram in item.descriptor.primitive_trigrams
        }
        composition_primitives = {
            label for item in by_split["composition_holdout"]
            for label in item.descriptor.primitive_labels
        }
        composition_primitive_support = len(
            composition_primitives & train_primitives
        ) / max(len(composition_primitives), 1)
        holdout_obligations = {
            tuple(item.value for item in gram) for gram in COMPOSITION_HOLDOUT_TRIGRAMS
        }
        composition_obligation_coverage = len(
            composition_grams & holdout_obligations
        ) / max(len(holdout_obligations), 1)
        leaked_holdout_grams = train_grams & holdout_obligations
        if grammar_present and required_coverage < self.config.minimum_train_ngram_coverage:
            reasons.append("train-topology-coverage")
        # Exact validation trigrams are a compositional extrapolation metric,
        # not an admission condition: demanding every combination in train
        # would make the validation split an imitation of train.  Admission
        # requires support for every constituent maneuver vocabulary item.
        if (
            validation_primitives
            and validation_primitive_support < self.config.minimum_validation_ngram_support
        ):
            reasons.append("validation-topology-support")
        if grammar_present and by_split["composition_holdout"]:
            if composition_primitive_support < self.config.minimum_validation_ngram_support:
                reasons.append("composition-primitive-support")
            if composition_obligation_coverage < 1.0:
                reasons.append("composition-obligation-coverage")
            if leaked_holdout_grams:
                reasons.append("composition-holdout-leakage")

        descriptors = [report.descriptor for report in reports if report.descriptor is not None]
        grid = DescriptorGrid(self.config)
        joint_cells = {grid.key(item) for item in descriptors}
        marginal = {
            name: len({grid.key(item)[index] for item in descriptors}) / (len(grid.edges[name]) + 1)
            for index, name in enumerate(grid.fields)
        }
        marginal_occupancy = float(np.mean(list(marginal.values()))) if marginal else 0.0
        if descriptors and marginal_occupancy < self.config.minimum_qd_occupancy_fraction:
            reasons.append("quality-diversity-occupancy")

        vertical_fraction = (
            sum(bool(item.audit["vertical_semantics"]) for item in reports) / max(len(reports), 1)
        )
        hard_fraction = (
            sum(bool(item.audit["hard_semantics"]) for item in reports) / max(len(reports), 1)
        )
        if reports and vertical_fraction < self.config.minimum_vertical_track_fraction:
            reasons.append("vertical-track-coverage")
        if reports and hard_fraction < self.config.minimum_hard_track_fraction:
            reasons.append("hard-track-coverage")

        dynamic = [item for item in descriptors if item.dynamic_available]
        attempted_by_split = {
            split: sum(
                str(record.get("split")) == split and record.get("qualification") is not None
                for record in records
            )
            for split in split_names
        }
        dynamic_yield = len(dynamic) / max(sum(attempted_by_split.values()), 1)
        dynamic_coverage = sum(attempted_by_split.values()) / max(len(descriptors), 1)
        split_yield: dict[str, float] = {}
        for split, items in by_split.items():
            split_yield[split] = sum(
                item.descriptor.dynamic_available for item in items
            ) / max(attempted_by_split[split], 1)
        comparable_yields = [
            split_yield[split] for split in split_yield if attempted_by_split[split] > 0
        ]
        dynamic_gap = (
            float(max(comparable_yields) - min(comparable_yields))
            if len(comparable_yields) >= 2 else 0.0
        )
        if require_dynamic and not dynamic:
            reasons.append("missing-dynamic-qualification")
        if dynamic and dynamic_gap > self.config.maximum_dynamic_qualification_yield_gap:
            reasons.append("dynamic-qualification-yield-gap")
        attempted_cells: set[tuple[int, ...]] = set()
        qualified_cells: set[tuple[int, ...]] = set()
        for record, report in zip(records, reports):
            if report.descriptor is None or record.get("qualification") is None:
                continue
            cell = grid.key(report.descriptor)
            attempted_cells.add(cell)
            if report.descriptor.dynamic_available:
                qualified_cells.add(cell)
        qualified_speeds = [
            float(item.values["qualified_speed_mps"]) for item in dynamic
        ]
        metrics: dict[str, Any] = {
            "record_count": len(records),
            "valid_track_count": sum(item.valid for item in reports),
            "artifact_errors": artifact_errors,
            "source_violations": source_violations,
            "split_geometry_overlap": split_overlap,
            "forbidden_geometry_overlap": len(forbidden_overlap),
            "train_required_ngram_coverage": float(required_coverage),
            "validation_ngram_support": float(validation_support),
            "validation_primitive_support": float(validation_primitive_support),
            "composition_primitive_support": float(composition_primitive_support),
            "composition_holdout_obligation_coverage": float(composition_obligation_coverage),
            "composition_holdout_leaked_trigrams": [list(item) for item in sorted(leaked_holdout_grams)],
            "composition_holdout_unique_trigrams": len(composition_grams),
            "train_unique_ngrams": len(train_grams),
            "validation_unique_ngrams": len(validation_grams),
            "qd_joint_occupied_cells": len(joint_cells),
            "qd_joint_occupancy_fraction": len(joint_cells) / max(grid.cell_count, 1),
            "qd_marginal_occupancy_fraction": marginal_occupancy,
            "qd_marginal_occupancy": marginal,
            "vertical_track_fraction": vertical_fraction,
            "hard_track_fraction": hard_fraction,
            "dynamic_qualification_yield": dynamic_yield,
            "dynamic_qualification_coverage": dynamic_coverage,
            "dynamic_qualification_attempted_by_split": attempted_by_split,
            "dynamic_qualification_yield_by_split": split_yield,
            "dynamic_qualification_yield_gap": dynamic_gap,
            "dynamic_qualification_yield_comparable_across_splits": len(comparable_yields) >= 2,
            "dynamic_qualification_attempted_qd_cells": len(attempted_cells),
            "dynamic_qualification_qualified_qd_cells": len(qualified_cells),
            "dynamic_qualification_qd_cell_yield": (
                len(qualified_cells) / max(len(attempted_cells), 1)
            ),
            "qualified_speed_mps": {
                "minimum": min(qualified_speeds) if qualified_speeds else None,
                "mean": float(np.mean(qualified_speeds)) if qualified_speeds else None,
                "maximum": max(qualified_speeds) if qualified_speeds else None,
            },
        }
        return DistributionValidationReport(
            passed=not reasons, reasons=tuple(dict.fromkeys(reasons)),
            metrics=metrics, track_reports=tuple(reports),
        )
