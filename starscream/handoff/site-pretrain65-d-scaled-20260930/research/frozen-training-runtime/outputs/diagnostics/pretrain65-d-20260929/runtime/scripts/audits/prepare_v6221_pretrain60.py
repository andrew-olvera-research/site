"""Promote the validated v6.22 pretrain60 recipe to its 240-round run."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "configs/exp/v6.22/pretraining_behavior60_dagger.yaml"
OUT = ROOT / "configs/exp/v6.22.1/pretrain60_dagger.yaml"

def main():
    cfg = yaml.safe_load(SOURCE.read_text())
    s = cfg["dagger"]
    if s["rounds"] != 75 or len(s["curriculum"]["tracks"]) != 60:
        raise ValueError("source is not the validated v6.22 60-course recipe")
    name = "starscream-v6.22.1-pretrain60"
    s["run_name"] = name
    s["rounds"] = 240
    s["tags"] = ["v6.22.1" if x == "v6.22" else x for x in s["tags"]]
    cfg["checkpoint"]["run_name"] = name
    cfg["wandb"]["run_name"] = name
    cfg["wandb"]["group"] = "starscream-v6.22.1-pretraining"
    cfg["wandb"]["tags"] = s["tags"]
    cfg["wandb"]["local_event_path"] = f"/workspace/outputs/logs/{name}.events.jsonl"
    cfg["experiment_notes"] = dict(cfg.get("experiment_notes") or {},
        status="PREPARED; v6.22.1 240-round pretrain60 run",
        budget="240 rounds x 480 episodes; retain the validated 1.5M-per-course initial target and continue at the same v6.21-optimal DAgger settings",
        mpcc_targets="all 60 manifest teacher profiles are active; the ten new bridge targets come from their individual robust pace frontiers")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cfg, indent=2) + "\n")
    manifest = Path(s["track_manifest"])
    hardware = ROOT / "configs/hardware/dagger_local_16c_v625.yaml"
    normalization = Path(s["dagger_initialization_stats_checkpoint"])
    atomic_json(OUT.with_suffix(".preflight.json"), {
        "config_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "normalization_sha256": hashlib.sha256(normalization.read_bytes()).hexdigest(),
        "hardware_profile": "configs/hardware/dagger_local_16c_v625.yaml",
        "hardware_profile_sha256": hashlib.sha256(hardware.read_bytes()).hexdigest(),
        "train_courses": 60, "rounds": 240, "episodes_per_round": 480,
        "updates_per_round": s["updates_per_round"],
        "evaluation_episodes": s["evaluation_episodes"],
        "reporting_evaluation_episodes": s["reporting_evaluation_episodes"],
    })
    print(json.dumps({"config": str(OUT), "run_name": name, "rounds": s["rounds"],
                      "courses": len(s["curriculum"]["tracks"])}, indent=2))

if __name__ == "__main__": main()
