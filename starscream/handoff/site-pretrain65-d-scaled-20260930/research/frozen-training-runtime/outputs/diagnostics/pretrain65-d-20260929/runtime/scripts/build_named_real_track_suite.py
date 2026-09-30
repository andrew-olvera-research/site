#!/usr/bin/env python3
"""Freeze and statically audit the named real-track benchmark pool.

This is deliberately a *pool* builder.  It does not decide train/report roles;
that split is frozen only after every canonical course passes static and MPCC
qualification.  Timing comparisons are enabled only when the repository
geometry and the published timing geometry have a defensible correspondence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_multigp_v1_suite import _route_audit
from starscream.env.multigp_tracks import (
    global_qualifier_2026, utt1, utt2, utt3, utt4, utt5, utt6, utt8, utt10,
)
from starscream.env.procedural_tracks import geometry_fingerprint, save_track_yaml
from starscream.env.tracks import Track, load_track


@dataclass(frozen=True)
class BenchmarkTrack:
    key: str
    display_name: str
    modality: str
    factory: Callable[[], Track]
    provenance_class: str
    source_url: str
    comparison_eligible: bool
    reference_best_seconds: float | None = None
    reference_second_seconds: float | None = None
    reference_top10_mean_seconds: float | None = None
    reference_median_seconds: float | None = None
    reference_label: str | None = None
    leaderboard_url: str | None = None
    caveat: str | None = None


def _asset(name: str) -> Callable[[], Track]:
    return lambda: load_track(ROOT / "starscream" / "assets" / "tracks" / name)


# MultiGP values are verified single-lap leaderboard snapshots.  Swift is the
# author-released physical course and A2RL is the published competition result,
# but the latter's local geometry is a source-consistent reconstruction.
POOL: tuple[BenchmarkTrack, ...] = (
    BenchmarkTrack(
        "swift_2022", "Swift champion course (2022)", "compact 3-D split-S",
        _asset("swift_champion_2022_exact.yaml"), "exact_author_release",
        "https://doi.org/10.5281/zenodo.7955278", True,
        reference_median_seconds=5.76,
        reference_label="human champion single-lap median (Vanover); Swift median was 5.52 s",
        leaderboard_url="https://doi.org/10.1038/s41586-023-06419-4",
    ),
    BenchmarkTrack(
        "a2rl_s2_2026", "A2RL Season 2 (2026)", "dense 3-D ladder and split-S",
        _asset("a2rl_s2_2026_source_consistent_v2.yaml"),
        "source_consistent_reconstruction",
        "https://arxiv.org/abs/2603.02742", False,
        reference_best_seconds=12.032, reference_second_seconds=12.832,
        reference_label="official autonomous championship laps",
        leaderboard_url="https://a2rl.io/news/45/A2RL-Drone-Championship-Sets-the-Pace-for-AI-in-Autonomous-Flight",
        caveat="Topology and metric envelope are sourced, but poses are not an organizer survey.",
    ),
    BenchmarkTrack(
        "multigp_cdra_2026", "MultiGP CDRA qualification (2026)",
        "compact planar collegiate qualifier",
        _asset("multigp_cdra_2026_reconstructed.yaml"),
        "official_trilateration_reconstruction",
        "https://www.multigp.com/wp-content/uploads/2025/08/2026-CDRA-Track-Diagram.pdf",
        False,
        caveat="Gate centers are reconstructed from the official build guide; flags are not modeled.",
    ),
    BenchmarkTrack(
        "multigp_utt01", "MultiGP UTT 1", "long straight / large planar turns",
        utt1, "official_dimensioned_plan",
        "https://www.multigp.com/wp-content/uploads/2017/05/MultiGP-universal-time-trial-track1.pdf",
        True, 8.829, reference_top10_mean_seconds=9.498,
        reference_label="verified single-lap leaderboard",
        leaderboard_url="https://www.multigp.com/leaderboards/utt1/",
    ),
    BenchmarkTrack(
        "multigp_utt02", "MultiGP UTT 2 — Tsunami", "reversal / teardrop",
        utt2, "official_dimensioned_plan_with_route_constraints",
        "https://www.multigp.com/wp-content/uploads/2017/05/MultiGP-universal-time-trial-track-2-Tsnuami-001.pdf",
        True, 5.485, reference_top10_mean_seconds=6.123,
        reference_label="verified single-lap leaderboard",
        leaderboard_url="https://www.multigp.com/leaderboards/utt2/",
    ),
    BenchmarkTrack(
        "multigp_utt03", "MultiGP UTT 3 — Bessel Run", "fast planar offsets",
        utt3, "official_dimensioned_plan",
        "https://www.multigp.com/wp-content/uploads/2017/05/MultiGP-universal-time-trial-track-3-BesselRun-002.pdf",
        True, 5.806, reference_top10_mean_seconds=6.558,
        reference_label="verified single-lap leaderboard",
        leaderboard_url="https://www.multigp.com/leaderboards/utt3/",
    ),
    BenchmarkTrack(
        "multigp_utt04", "MultiGP UTT 4 — High Voltage", "bow-tie / over-gate",
        utt4, "official_dimensioned_plan_with_route_constraints",
        "https://www.multigp.com/wp-content/uploads/2017/05/MultiGP-universal-time-trial-track-4-high-voltage-001.pdf",
        True, 7.86, reference_top10_mean_seconds=8.965,
        reference_label="verified single-lap leaderboard",
        leaderboard_url="https://www.multigp.com/leaderboards/utt4/",
    ),
    BenchmarkTrack(
        "multigp_utt05", "MultiGP UTT 5 — Nautilus", "logarithmic spiral",
        utt5, "official_dimensioned_plan",
        "https://www.multigp.com/wp-content/uploads/2017/05/MultiGP-universal-time-trial-track-5-nautilus-manual-001.pdf",
        True, 5.606, reference_top10_mean_seconds=5.981,
        reference_label="verified single-lap leaderboard",
        leaderboard_url="https://www.multigp.com/leaderboards/utt5/",
    ),
    BenchmarkTrack(
        "multigp_utt06", "MultiGP UTT 6 — Fury", "over-under / flags / reversals",
        utt6, "official_dimensioned_plan_with_route_constraints",
        "https://www.multigp.com/wp-content/uploads/2017/05/MultiGP-universal-time-trial-track-6-fury.pdf",
        True, 7.72, reference_top10_mean_seconds=9.336,
        reference_label="verified single-lap leaderboard",
        leaderboard_url="https://www.multigp.com/leaderboards/utt6-fury/",
    ),
    BenchmarkTrack(
        "multigp_utt08", "MultiGP UTT 8 — Revenge", "flag slalom / open speed",
        utt8, "official_dimensioned_plan_with_route_constraints",
        "https://www.multigp.com/wp-content/uploads/2019/01/MultiGP-universal-time-trial-8-manual.pdf",
        True, 9.44, reference_top10_mean_seconds=11.088,
        reference_label="verified single-lap leaderboard",
        leaderboard_url="https://www.multigp.com/leaderboards/utt-8-leaderboard-revenge/",
    ),
    BenchmarkTrack(
        "multigp_utt10", "MultiGP UTT 10 — Prairie Rage", "dense flags / mixed turns",
        utt10, "official_dimensioned_plan_with_route_constraints",
        "https://www.multigp.com/wp-content/uploads/2020/06/MultiGP-universal-time-trial-track-10-prairie-rage-manual1.pdf",
        True, 8.088, reference_top10_mean_seconds=9.540,
        reference_label="verified single-lap leaderboard",
        leaderboard_url="https://www.multigp.com/leaderboards/utt-10-prairie-rage-1-lap-leaderboard/",
    ),
    BenchmarkTrack(
        "multigp_gq_2026", "MultiGP Global Qualifier (2026)",
        "long mixed gate / flag qualification course",
        global_qualifier_2026, "official_simulator_checkpoint_export",
        "https://www.multigp.com/multigp-2026-global-qualifier-track/", False,
        caveat=(
            "Official checkpoint transforms are available, but the source's reported "
            "length disagrees with its checkpoint chord sum; timing claims are disabled."
        ),
    ),
)


# Never round-trip repository reference geometries through Track's float32
# runtime representation.  The benchmark copy must retain the source YAML
# byte-for-byte; provenance belongs in the manifest, not in mutated geometry.
ASSET_SOURCE_BY_KEY = {
    "swift_2022": ROOT / "starscream" / "assets" / "tracks" / "swift_champion_2022_exact.yaml",
    "a2rl_s2_2026": ROOT / "starscream" / "assets" / "tracks" / "a2rl_s2_2026_source_consistent_v2.yaml",
    "multigp_cdra_2026": ROOT / "starscream" / "assets" / "tracks" / "multigp_cdra_2026_reconstructed.yaml",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/workspace/outputs/procedural-tracks/named-real-v4.2"),
    )
    return parser.parse_args()


def _geometry_checks(track: Track) -> list[str]:
    reasons: list[str] = []
    positions = np.stack([gate.position for gate in track.gates])
    quaternions = np.stack([gate.quaternion_wxyz for gate in track.gates])
    sizes = np.stack([gate.size for gate in track.gates])
    if not np.all(np.isfinite(positions)):
        reasons.append("nonfinite-position")
    if not np.allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=2e-4):
        reasons.append("nonunit-quaternion")
    if np.any(sizes <= 0.0):
        reasons.append("nonpositive-aperture")
    if np.any(positions < track.bounds[:, 0] - 1e-5) or np.any(
        positions > track.bounds[:, 1] + 1e-5
    ):
        reasons.append("checkpoint-outside-bounds")
    if not track.loop:
        reasons.append("not-a-lap-track")
    return reasons


def main() -> None:
    args = arguments()
    root = args.output.resolve()
    canonical = root / "tracks" / "canonical"
    cache = root / "racing-lines"
    records: list[dict[str, object]] = []
    fingerprints: set[str] = set()
    for spec in POOL:
        original = spec.factory()
        path = canonical / f"{original.name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        source_asset = ASSET_SOURCE_BY_KEY.get(spec.key)
        if source_asset is not None:
            shutil.copyfile(source_asset, path)
        else:
            metadata = dict(original.metadata or {})
            metadata["benchmark_key"] = spec.key
            metadata["benchmark_display_name"] = spec.display_name
            metadata["benchmark_modality"] = spec.modality
            metadata["benchmark_provenance_class"] = spec.provenance_class
            metadata["benchmark_source_url"] = spec.source_url
            metadata["benchmark_timing_comparison_eligible"] = spec.comparison_eligible
            save_track_yaml(replace(original, metadata=metadata), path)
        track = load_track(path)
        geometry_reasons = _geometry_checks(track)
        route = _route_audit(track, cache)
        reasons = geometry_reasons + list(route["reasons"])
        fingerprint = geometry_fingerprint(track)
        if fingerprint in fingerprints:
            reasons.append("duplicate-geometry")
        fingerprints.add(fingerprint)
        records.append({
            "key": spec.key,
            "name": track.name,
            "display_name": spec.display_name,
            "modality": spec.modality,
            "path": str(path.relative_to(root)),
            "track_fingerprint": track.fingerprint,
            "geometry_fingerprint": fingerprint,
            "gate_count": len(track.gates),
            "rendered_gate_count": sum(gate.render for gate in track.gates),
            "provenance_class": spec.provenance_class,
            "source_url": spec.source_url,
            "comparison_eligible": spec.comparison_eligible,
            "reference_timing": {
                "best_seconds": spec.reference_best_seconds,
                "second_seconds": spec.reference_second_seconds,
                "top10_mean_seconds": spec.reference_top10_mean_seconds,
                "median_seconds": spec.reference_median_seconds,
                "label": spec.reference_label,
                "url": spec.leaderboard_url,
                "snapshot_utc": "2026-08-31",
            },
            "caveat": spec.caveat,
            "static_audit": {**route, "valid": not reasons, "reasons": reasons},
            "mpcc_qualification": None,
        })

    failures = [str(row["key"]) for row in records if not row["static_audit"]["valid"]]  # type: ignore[index]
    payload = {
        "schema": "starscream-named-real-track-pool-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_status": "unassigned_pool",
        "split_policy": (
            "Assign roles only after canonical MPCC qualification. Canonical report tracks "
            "must never be augmented, sampled, or used for model selection."
        ),
        "records": records,
        "excluded": [
            {"name": "MultiGP UTT 7 Tiny WhUTT", "reason": "whoop-scale plant mismatch"},
            {"name": "MultiGP UTT 9 Mega UTT", "reason": "mega-scale plant and reconstructed route"},
        ],
        "static_audit_passed": not failures,
        "static_failures": failures,
    }
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({
        "manifest": str(destination),
        "tracks": len(records),
        "comparison_eligible": sum(bool(row["comparison_eligible"]) for row in records),
        "static_audit_passed": not failures,
        "static_failures": failures,
    }, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
