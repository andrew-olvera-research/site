"""Overlay closed-loop policy evidence on a geometry manifold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import json
import numpy as np
from scipy.stats import spearmanr

from .atlas import ManifoldAtlas
from .descriptors import TrackGeometryProfile


@dataclass(frozen=True, slots=True)
class BehaviorManifoldReport:
    evaluations: tuple[Mapping[str, Any], ...]
    phase_probes: tuple[Mapping[str, Any], ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "starscream-racing-manifold-behavior-v1",
            "evaluations": list(self.evaluations),
            "phase_probes": list(self.phase_probes),
            "scope": (
                "Geometry/performance association. Correlation is diagnostic, "
                "not evidence that geometric distance alone causes failure."
            ),
        }


def _load_payload(value: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _metric_candidates(payload: Any) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    if isinstance(payload, Mapping):
        if any(str(key).startswith("track/") for key in payload):
            candidates.append(payload)
        for value in payload.values():
            candidates.extend(_metric_candidates(value))
    elif isinstance(payload, list):
        for value in payload:
            candidates.extend(_metric_candidates(value))
    return candidates


def extract_track_metrics(
    payload: str | Path | Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Extract ``track/<name>/<metric>`` values from evaluation JSON."""

    raw = _load_payload(payload)
    candidates = _metric_candidates(raw)
    if not candidates:
        return {}
    metrics = max(
        candidates,
        key=lambda item: sum(str(key).startswith("track/") for key in item),
    )
    output: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        text = str(key)
        if not text.startswith("track/") or not isinstance(value, (int, float)):
            continue
        _, track, metric = text.split("/", 2)
        if np.isfinite(float(value)):
            output.setdefault(track, {})[metric] = float(value)
    return output


def analyze_evaluation_payloads(
    atlas: ManifoldAtlas,
    payloads: Sequence[str | Path | Mapping[str, Any]],
    *,
    training_names: Sequence[str] | None = None,
) -> BehaviorManifoldReport:
    """Associate per-track checkpoint metrics with atlas support distance."""

    profile_by_name = {profile.name: profile for profile in atlas.profiles}
    train_names = tuple(training_names or atlas.names)
    train_indices = [atlas.names.index(name) for name in train_names]
    rows: list[Mapping[str, Any]] = []
    for source in payloads:
        metrics_by_track = extract_track_metrics(source)
        for name, metrics in metrics_by_track.items():
            profile = profile_by_name.get(name)
            if profile is None:
                continue
            total, descriptor, ordered = atlas.distances(profile)
            eligible = train_indices
            nearest = min(eligible, key=lambda index: total[index])
            gate_count = float(profile.values["gate_count"])
            mean_gates = metrics.get("mean_gates", metrics.get("final_active_gate_index", 0.0))
            rows.append({
                "source": str(source) if not isinstance(source, Mapping) else "in-memory",
                "track": name,
                "nearest_training_track": atlas.names[nearest],
                "distance_to_training": float(total[nearest]),
                "descriptor_distance_to_training": float(descriptor[nearest]),
                "ordered_distance_to_training": float(ordered[nearest]),
                "full_course_success": float(metrics.get("full_course_success", 0.0)),
                "mean_gates": float(mean_gates),
                "normalized_gate_progress": float(mean_gates / max(gate_count, 1.0)),
                "mean_gate_speed_mps": float(metrics.get("mean_gate_speed_mps", 0.0)),
                "mean_speed_mps": float(metrics.get("mean_speed_mps", 0.0)),
                "crash_rate": float(metrics.get("crash_rate", metrics.get("ground_contact", 0.0))),
            })
    return BehaviorManifoldReport(evaluations=tuple(rows))


def analyze_phase_probe(
    profile: TrackGeometryProfile,
    payload: str | Path | Mapping[str, Any],
    *,
    domain: str = "randomized",
    transition_chain_support: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Map all-start evaluation failures onto local transition geometry."""

    raw = _load_payload(payload)
    checkpoints = list(raw.get("checkpoints") or ())
    reports: list[Mapping[str, Any]] = []
    for checkpoint in checkpoints:
        domains = dict(checkpoint.get("domains") or {})
        selected = dict(domains.get(domain) or {})
        by_gate = dict(selected.get("by_start_gate") or {})
        if not by_gate:
            continue
        phase_rows: list[dict[str, Any]] = []
        local = profile.transition_features
        for raw_name, metrics in by_gate.items():
            gate_index = int(str(raw_name).split("_")[-1]) - 1
            if not 0 <= gate_index < len(local):
                continue
            row = {
                "gate": gate_index + 1,
                "segment_length_m": float(local[gate_index, 0]),
                "turn_3d_degrees": float(local[gate_index, 2]),
                "absolute_slope": float(local[gate_index, 4]),
                "gate_normal_mismatch_degrees": float(local[gate_index, 5]),
                "gate_up_change_degrees": float(local[gate_index, 6]),
                "curvature_proxy_m_inv": float(local[gate_index, 7]),
                "one_gate_success": float(metrics.get("one_gate_success", 0.0)),
                "three_gate_success": float(metrics.get("three_gate_success", 0.0)),
                "mean_gates": float(metrics.get("mean_gates", 0.0)),
                "mean_gate_speed_mps": float(metrics.get("mean_gate_speed_mps", 0.0)),
            }
            if transition_chain_support is not None:
                distance = transition_chain_support.get("per_start_gate_distance", ())
                if gate_index < len(distance):
                    row["transition_chain_support_distance"] = float(distance[gate_index])
            phase_rows.append(row)
        correlations: dict[str, float | None] = {}
        for feature in (
            "segment_length_m", "turn_3d_degrees", "absolute_slope",
            "gate_normal_mismatch_degrees", "gate_up_change_degrees",
            "curvature_proxy_m_inv",
        ):
            result = spearmanr(
                [row[feature] for row in phase_rows],
                [row["three_gate_success"] for row in phase_rows],
            )
            correlations[feature] = (
                None if not np.isfinite(result.statistic) else float(result.statistic)
            )
        reports.append({
            "track": profile.name,
            "checkpoint": str(checkpoint.get("checkpoint", "unknown")),
            "round": int(checkpoint.get("round", -1)),
            "domain": domain,
            "three_gate_success_spearman": correlations,
            "chain_support_distance_spearman": (
                None
                if not phase_rows
                or "transition_chain_support_distance" not in phase_rows[0]
                else _finite_spearman(
                    [row["transition_chain_support_distance"] for row in phase_rows],
                    [row["three_gate_success"] for row in phase_rows],
                )
            ),
            "hardest_start_phases": sorted(
                phase_rows,
                key=lambda row: (row["three_gate_success"], row["mean_gates"]),
            )[:5],
            "phases": phase_rows,
        })
    return tuple(reports)


def _finite_spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    result = spearmanr(left, right)
    return None if not np.isfinite(result.statistic) else float(result.statistic)
