"""Run the matched mini broad/ bounded DAgger baselines sequentially."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "outputs/diagnostics/mini-slalom-recovery-baselines/status.json"
CONFIGS = (
    ROOT / "configs/exp/v6.21.1.1/mini_slalom_broad.yaml",
    ROOT / "configs/exp/v6.21.1.1/mini_slalom_bounded.yaml",
)


def write_status(data: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    if STATUS.exists():
        raise RuntimeError(f"Existing baseline queue status: {STATUS}")
    if not all(p.exists() for p in CONFIGS):
        raise FileNotFoundError("Prepare both mini baseline configs first")
    status = {"started_unix": time.time(), "state": "running", "legs": []}
    write_status(status)
    for config in CONFIGS:
        name = config.stem
        log = ROOT / f"outputs/logs/{name}.console.log"
        command = ["/opt/conda/bin/python", "-u", "scripts/train_privileged_racing.py",
                   "--config", str(config.relative_to(ROOT)), "--stage", "dagger", "--device", "cuda"]
        leg = {"name": name, "config": str(config.relative_to(ROOT)),
               "log": str(log.relative_to(ROOT)), "state": "running",
               "started_unix": time.time()}
        status["legs"].append(leg)
        write_status(status)
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       env={**os.environ, "PYTHONUNBUFFERED": "1"})
            leg["pid"] = process.pid
            write_status(status)
            result = process.wait()
        leg.update(state="complete" if result == 0 else "failed",
                   exit_code=result, finished_unix=time.time())
        write_status(status)
        if result != 0:
            status["state"] = "failed"
            status["finished_unix"] = time.time()
            write_status(status)
            return result
    status["state"] = "complete"
    status["finished_unix"] = time.time()
    write_status(status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
