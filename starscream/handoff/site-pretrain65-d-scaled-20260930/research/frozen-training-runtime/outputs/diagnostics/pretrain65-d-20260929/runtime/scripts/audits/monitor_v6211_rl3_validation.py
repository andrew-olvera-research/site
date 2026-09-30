"""Run a small fixed-seed validation panel between full PPO evaluations."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace")
RUN = "starscream-v6.21.1.rl.3-plant-recovery-fix-clean-100m"
LATEST = ROOT / "outputs/checkpoints" / RUN / "latest.pt"
OUTPUT = ROOT / "outputs/evals/v6211-rl3-live-val"
SUMMARY = OUTPUT / "cadence-100.jsonl"
CONFIG = ROOT / "configs/exp/v6.21.1.rl.3/plant_recovery_fix_clean_100m.yaml"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    completed = set()
    if SUMMARY.exists():
        completed = {int(json.loads(line)["target_cycle"]) for line in SUMMARY.open()}
    next_cycle = 75
    while next_cycle <= 465:
        if next_cycle in completed:
            next_cycle += 15
            continue
        try:
            payload = torch.load(LATEST, map_location="cpu", weights_only=False)
            cycle = int(payload["cycle"])
            step = int(payload["environment_steps"])
            del payload
        except (FileNotFoundError, EOFError, RuntimeError, KeyError):
            time.sleep(20)
            continue
        if cycle < next_cycle:
            time.sleep(20)
            continue
        snapshot = OUTPUT / f"snapshot-step-{step}.pt"
        result = OUTPUT / f"cadence-step-{step}.json"
        try:
            shutil.copy2(LATEST, snapshot)
            command = [
                sys.executable, str(ROOT / "scripts/eval_privileged_dagger.py"),
                "--config", str(CONFIG), "--checkpoint", str(snapshot),
                "--section", "ppo", "--curriculum", "validation",
                "--episodes", "100", "--seed", "2036091708",
                "--workers", "8", "--device", "cuda", "--output", str(result),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            metrics = json.loads(result.read_text())["metrics"]
            record = {
                "target_cycle": next_cycle, "cycle": cycle, "step": step,
                "episodes": 100,
                "full_course_success": metrics["full_course_success"],
                "missed_gate_termination_rate": metrics["missed_gate_termination_rate"],
                "gate_dwell_termination_rate": metrics["gate_dwell_termination_rate"],
                "successful_median_steps": metrics["successful_median_steps"],
                "successful_p90_steps": metrics["successful_p90_steps"],
            }
            with SUMMARY.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        except (subprocess.CalledProcessError, OSError, KeyError) as exc:
            print(f"validation cycle={cycle} failed: {exc}", file=sys.stderr, flush=True)
        finally:
            snapshot.unlink(missing_ok=True)
        next_cycle += 15


if __name__ == "__main__":
    main()
