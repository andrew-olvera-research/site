"""Atomic, resumable experiment checkpoints and evaluation summaries."""

from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-")
    if not name:
        raise ValueError("run_name must contain at least one filename-safe character")
    return name


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().item() if value.numel() == 1 else value.detach().float().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


@dataclass(frozen=True)
class RankedCheckpoint:
    path: str
    score: float
    step: int


class CheckpointManager:
    """Own one run's latest checkpoint, top-k checkpoints, and eval records.

    Checkpoint payloads are deliberately caller-defined so optimizers, schedulers,
    EMA teachers, temporary training heads, RNG state, and other throwaway modules
    can all be included without this utility knowing their types.
    """

    def __init__(
        self,
        run_name: str,
        *,
        output_root: str | Path = "outputs",
        monitor: str = "loss",
        mode: str = "min",
        top_k: int = 3,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("checkpoint mode must be 'min' or 'max'")
        if top_k < 1:
            raise ValueError("checkpoint top_k must be positive")
        self.run_name = _safe_name(run_name)
        self.output_root = Path(output_root)
        self.checkpoint_dir = self.output_root / "checkpoints" / self.run_name
        self.eval_dir = self.output_root / "evals" / self.run_name
        self.monitor = monitor
        self.mode = mode
        self.top_k = int(top_k)
        self.latest_path = self.checkpoint_dir / "latest.pt"
        self.manifest_path = self.checkpoint_dir / "top-k.json"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CheckpointManager":
        wandb_config = config.get("wandb", {})
        checkpoint_config = config.get("checkpoint", {})
        run_name = checkpoint_config.get("run_name") or wandb_config.get("run_name")
        if not run_name:
            raise ValueError("config must define wandb.run_name or checkpoint.run_name")
        return cls(
            str(run_name),
            output_root=config.get("output_root", "outputs"),
            monitor=str(checkpoint_config.get("monitor", "loss")),
            mode=str(checkpoint_config.get("mode", "min")),
            top_k=int(checkpoint_config.get("top_k", 3)),
        )

    def _atomic_torch_save(self, payload: Mapping[str, Any], destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(handle)
        temporary_path = Path(temporary)
        try:
            torch.save(dict(payload), temporary_path)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _read_manifest(self) -> list[RankedCheckpoint]:
        if not self.manifest_path.exists():
            return []
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return [RankedCheckpoint(**item) for item in raw.get("checkpoints", [])]

    def _write_manifest(self, entries: list[RankedCheckpoint]) -> None:
        payload = {
            "monitor": self.monitor,
            "mode": self.mode,
            "top_k": self.top_k,
            "checkpoints": [entry.__dict__ for entry in entries],
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.manifest_path)

    def _rank(self, entries: list[RankedCheckpoint]) -> list[RankedCheckpoint]:
        return sorted(entries, key=lambda item: item.score, reverse=self.mode == "max")

    def save(
        self,
        payload: Mapping[str, Any],
        *,
        step: int,
        metrics: Mapping[str, Any] | None = None,
        rank: bool = True,
    ) -> Path:
        state = dict(payload)
        state["step"] = int(step)
        state["metrics"] = _jsonable(metrics or {})
        self._atomic_torch_save(state, self.latest_path)
        if not rank or metrics is None or self.monitor not in metrics:
            return self.latest_path
        score = float(metrics[self.monitor])
        if not math.isfinite(score):
            return self.latest_path
        entries = [entry for entry in self._read_manifest() if (self.checkpoint_dir / entry.path).exists()]
        candidate_name = f"best-step-{step:09d}-{self.monitor}-{score:.8g}.pt"
        candidate = RankedCheckpoint(candidate_name, score, int(step))
        ranked = self._rank(entries + [candidate])
        keep, discard = ranked[: self.top_k], ranked[self.top_k :]
        if candidate in keep:
            self._atomic_torch_save(state, self.checkpoint_dir / candidate_name)
        for entry in discard:
            (self.checkpoint_dir / entry.path).unlink(missing_ok=True)
        self._write_manifest(keep)
        return self.latest_path

    def resume(self, enabled: bool, *, map_location: str | torch.device = "cpu") -> dict[str, Any] | None:
        if not enabled:
            return None
        if not self.latest_path.is_file():
            raise FileNotFoundError(
                f"resume=true but no checkpoint exists at {self.latest_path}"
            )
        return torch.load(self.latest_path, map_location=map_location, weights_only=False)

    def save_eval(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
        summary: str,
        probes: Mapping[str, Any] | None = None,
        kind: str = "validation",
    ) -> Path:
        destination = self.eval_dir / f"{_safe_name(kind)}-step-{step:09d}.json"
        payload = {
            "run_name": self.run_name,
            "step": int(step),
            "summary": summary,
            "kind": kind,
            "metrics": _jsonable(metrics),
            "probes": _jsonable(probes or {}),
        }
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        return destination


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].detach().cpu().to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.detach().cpu().to(torch.uint8) for item in state["cuda"]])
