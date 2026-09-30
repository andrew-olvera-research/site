#!/usr/bin/env python3
"""Build a reproducible evaluation-only pace map from frozen course evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "configs/eval/v6_22_real100_hard_v2.yaml"
MANIFEST = ROOT / "configs/eval/v6_22_real100_hard_v2.manifest.json"
DAGGER_CONFIG = ROOT / "configs/exp/v6.21.1/update_fix_dagger.yaml"
RL_CONFIG = ROOT / "configs/exp/v6.21.1.rl.1/reward_frontier_100m.yaml"
OUTPUT = ROOT / "configs/eval/pushed_speed_commands_v1.json"


def workspace_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else Path("/workspace") / path)


def pushed_command(qualified: float | None) -> float:
    if qualified is None:
        return 16.0
    return min(24.0, max(16.0, 1.2 * float(qualified)))


def main() -> None:
    suite = yaml.safe_load(SUITE.read_text())
    manifest = json.loads(MANIFEST.read_text())
    records = {row["name"]: row for row in manifest["records"]}
    commands: dict[str, float] = {}
    sources: dict[str, str] = {}
    for row in suite["active"]:
        record = records[row["name"]]
        evidence = record["qualification"]
        qualified = (evidence.get("qualified") is True
                     and evidence.get("fingerprint") == row["geometry_fingerprint"])
        speed = evidence.get("speed_command") if qualified else None
        path = workspace_path(row["track"])
        commands[path] = pushed_command(speed)
        sources[path] = "pushed_mpcc_qualification" if speed is not None else "unqualified_fallback_16"

    # These validation tracks lack a shared qualified-speed map. Use an
    # explicit pushed diagnostic command while preserving the hard-v2 values
    # for any concrete track path shared with that suite.
    for path in (DAGGER_CONFIG, RL_CONFIG):
        config = yaml.safe_load(path.read_text())
        section = config["dagger" if path == DAGGER_CONFIG else "ppo"]
        for stage_key in ("evaluation_curriculum", "reporting_evaluation_curriculum"):
            stage = section.get(stage_key, {})
            if isinstance(stage, dict):
                for track in stage.get("tracks", []):
                    resolved = workspace_path(track)
                    if resolved not in commands:
                        commands[resolved] = 19.8
                        sources[resolved] = "validation_probe_19_8_no_qualified_speed"

    result = {
        "schema": "starscream-evaluation-speed-commands-v1",
        "description": "Pushed evaluation commands; MPCC values are feasibility profiles, not time optima",
        "rule": "qualified: clamp(1.2 * MPCC command, 16, 24); unqualified hard-v2: 16; other validation: 19.8",
        "hard_v2_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "commands": dict(sorted(commands.items())),
        "sources": dict(sorted(sources.items())),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUTPUT}: {len(commands)} concrete track commands")


if __name__ == "__main__":
    main()
