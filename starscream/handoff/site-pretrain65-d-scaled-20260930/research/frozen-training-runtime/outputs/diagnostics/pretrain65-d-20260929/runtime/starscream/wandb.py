"""Small, consistent Weights & Biases integration for every trainer."""

from __future__ import annotations

import os
import json
import re
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Mapping


def load_project_env(path: str | Path | None = None) -> Path | None:
    """Load missing variables from the project .env without printing secrets."""
    candidate = Path(path) if path else Path(__file__).resolve().parents[1] / ".env"
    if not candidate.is_file():
        return None
    for raw_line in candidate.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
    return candidate


def _generate_run_id(wandb_module) -> str:
    """W&B run id across SDK versions (wandb.util.generate_id was removed in 0.30)."""
    for path in ("sdk.lib.runid", "util"):
        module = wandb_module
        try:
            for part in path.split("."):
                module = getattr(module, part)
            generate = getattr(module, "generate_id")
        except AttributeError:
            try:
                import importlib
                module = importlib.import_module(f"wandb.{path}")
                generate = getattr(module, "generate_id")
            except (ImportError, AttributeError):
                continue
        return str(generate())
    import secrets
    import string
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(8))


class WandbLogger:
    def __init__(self, config: Mapping[str, Any], *, full_config: Mapping[str, Any]) -> None:
        self.run = None
        # A coordinator can own one remote run while segment trainers emit
        # local scalar events. No W&B connection/run is created in this mode.
        self.event_path = Path(config['event_path']) if config.get('event_path') else None
        # Unlike event_path (coordinator-only mode), this is a local mirror of
        # a normal remote run. Preserve the existing coordinator contract.
        self.local_event_path = (Path(config['local_event_path'])
                                 if config.get('local_event_path') else None)
        # Full diagnostics are local only; remote allowlists never affect
        # evaluation/checkpoint decisions or this audit stream.
        self.local_full_event_path = (Path(config['local_full_event_path'])
                                     if config.get('local_full_event_path') else None)
        if self.local_full_event_path is not None:
            self.local_full_event_path.parent.mkdir(parents=True, exist_ok=True)
        if self.local_event_path is not None:
            self.local_event_path.parent.mkdir(parents=True, exist_ok=True)
        if self.event_path is not None:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self.enabled = bool(config.get("enabled", True))
        self.train_metric_allowlist = tuple(config.get("train_metric_allowlist", ()))
        self.eval_metric_allowlist = tuple(config.get("eval_metric_allowlist", ()))
        if not self.enabled or self.event_path is not None:
            return
        load_project_env(config.get("env_file"))
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            raise RuntimeError("W&B is enabled but WANDB_API_KEY is absent from the environment and .env")
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("W&B is enabled; install the project's training dependencies including wandb") from error
        project = str(config.get("project", "starscream"))
        run_name = config.get("run_name")
        if not run_name:
            raise ValueError("wandb.run_name is required")
        run_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(run_name)).strip(".-")
        output_root = Path(full_config.get("output_root", "outputs"))
        metadata_path = output_root / "checkpoints" / run_slug / "wandb-run.json"
        resume_requested = bool(full_config.get("resume", False)) or any(
            bool(value.get("resume", False))
            for value in full_config.values()
            if isinstance(value, Mapping)
        )
        configured_id = config.get("run_id")
        if resume_requested and metadata_path.is_file():
            configured_id = json.loads(metadata_path.read_text(encoding="utf-8"))["run_id"]
        run_id = str(configured_id or _generate_run_id(wandb))
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps({"run_id": run_id, "project": project, "run_name": run_name}, indent=2) + "\n", encoding="utf-8")
        wandb.login(key=api_key, relogin=False)
        self.run = wandb.init(
            project=project,
            name=str(run_name),
            tags=list(config.get("tags", [])),
            entity=config.get("entity"),
            mode=config.get("mode"),
            config=dict(full_config),
            resume="must" if resume_requested and configured_id else "never",
            id=run_id,
        )
        self.run.define_metric("train/step")
        self.run.define_metric("train/*", step_metric="train/step")
        self.run.define_metric("eval/step")
        self.run.define_metric("eval/*", step_metric="eval/step")

    @staticmethod
    def _filtered(
        metrics: Mapping[str, Any], allowlist: tuple[str, ...],
    ) -> dict[str, float]:
        if not allowlist:
            return {key: float(value) for key, value in metrics.items()}
        return {
            key: float(value)
            for key, value in metrics.items()
            if any(
                fnmatchcase(key, pattern)
                for pattern in allowlist
            )
        }

    def log_train(self, metrics: Mapping[str, Any], step: int) -> None:
        self._emit_full('train', metrics, step)
        if self.run is not None or self.event_path is not None:
            selected = self._filtered(metrics, self.train_metric_allowlist)
            self._emit({"train/step": step, **{f"train/{key}": value for key, value in selected.items()}}, step)

    def log_eval(self, metrics: Mapping[str, Any], step: int) -> None:
        self._emit_full('eval', metrics, step)
        if self.run is not None or self.event_path is not None:
            selected = self._filtered(metrics, self.eval_metric_allowlist)
            self._emit({"eval/step": step, **{f"eval/{key}": value for key, value in selected.items()}}, step)

    def _emit_full(self, namespace, metrics, step):
        if self.local_full_event_path is not None:
            payload = {f'{namespace}/step': int(step),
                       **{f'{namespace}/{key}': float(value) for key, value in metrics.items()}}
            with self.local_full_event_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=int(step), metrics=payload))+'\n')

    def _emit(self, payload: dict[str, Any], step: int) -> None:
        if self.local_event_path is not None:
            with self.local_event_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=int(step), metrics=payload))+'\n')
        if self.event_path is not None:
            with self.event_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=int(step), metrics=payload))+'\n')
        elif self.run is not None:
            self.run.log(payload)

    def log_images(self, images: Mapping[str, Any], step: int) -> None:
        if self.run is None:
            return
        import wandb
        payload = {"eval/step": step}
        for key, value in images.items():
            if hasattr(value, "detach"):
                value = value.detach().float().cpu().numpy()
            payload[f"eval/{key}"] = wandb.Image(value)
        self.run.log(payload)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def init_wandb(config: Mapping[str, Any]) -> WandbLogger:
    return WandbLogger(config.get("wandb", {"enabled": False}), full_config=config)
