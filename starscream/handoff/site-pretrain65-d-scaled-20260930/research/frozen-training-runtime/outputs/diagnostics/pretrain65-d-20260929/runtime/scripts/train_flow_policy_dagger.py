#!/usr/bin/env python3
"""Robot-gated DAgger and stabilizing flow distillation for gate chaining."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import yaml

from starscream.checkpoint_manager import CheckpointManager, capture_rng_state
from starscream.DiT import TokenizerFlowPolicy
from starscream.env import FlightmareEnv
from starscream.flow_policy import tokenizer_policy_tokens
from starscream.racing_curriculum import (
    RaceResetArchive,
    RacingCurriculumStage,
    sample_curriculum_spawn,
)
from starscream.wandb import init_wandb
try:  # Support both direct script execution and pytest/module imports.
    from scripts.eval_flight import ObservationHistory, ctbr_to_normalized, normalized_to_ctbr
    from scripts.train_actor_bc import load_frozen_actor_stack
    from scripts.train_flow_policy_fpo import make_mpcc
except ModuleNotFoundError:
    from eval_flight import ObservationHistory, ctbr_to_normalized, normalized_to_ctbr
    from train_actor_bc import load_frozen_actor_stack
    from train_flow_policy_fpo import make_mpcc


_TEACHER_CACHE: dict[tuple[str, float], Any] = {}


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    seen = set() if seen is None else seen
    if resolved in seen:
        raise ValueError(f"cyclic config inheritance at {resolved}")
    seen.add(resolved)
    with resolved.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    parent = config.get("inherits")
    if parent is None:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return merge_config(load_config(parent_path, seen), config)


def arguments() -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    preliminary, _ = parser.parse_known_args()
    config = load_config(preliminary.config)
    parser.set_defaults(device=str(config["flow_dagger"].get("device", "cuda")))
    args = parser.parse_args()
    if args.smoke:
        config = copy.deepcopy(config)
        settings = config["flow_dagger"]
        settings.update(
            rounds=1, episodes_per_round=2, evaluation_episodes=2,
            updates_per_round=2, batch_size=2, max_steps=18,
            replay_capacity=64, uncertainty_samples=2,
        )
        config["output_root"] = "/tmp/starscream-flow-dagger-smoke"
        config["checkpoint"] = {
            "run_name": "flow-dagger-smoke", "monitor": "selection_score",
            "mode": "max", "top_k": 1,
        }
        config["wandb"] = {"enabled": False, "run_name": "flow-dagger-smoke"}
    return args, config


@dataclass(slots=True)
class DAggerSample:
    observation_tokens: torch.Tensor
    action_tokens: torch.Tensor
    previous_action: torch.Tensor
    target: torch.Tensor
    priority: float
    source: str


class PrioritizedReplay:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.samples: list[DAggerSample] = []
        self.cursor = 0
        self.rng = np.random.default_rng(seed)

    def add(self, sample: DAggerSample) -> None:
        if len(self.samples) < self.capacity:
            self.samples.append(sample)
        else:
            self.samples[self.cursor] = sample
            self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, count: int, device: str) -> dict[str, torch.Tensor]:
        if not self.samples:
            raise RuntimeError("DAgger replay is empty")
        priorities = np.asarray([item.priority for item in self.samples], np.float64)
        probabilities = priorities / priorities.sum()
        indices = self.rng.choice(
            len(self.samples), size=min(int(count), len(self.samples)),
            replace=len(self.samples) < count, p=probabilities,
        )
        selected = [self.samples[int(index)] for index in indices]
        return {
            key: torch.stack([getattr(item, key) for item in selected]).to(
                device, non_blocking=True
            ).float()
            for key in ("observation_tokens", "action_tokens", "previous_action", "target")
        }

    def metrics(self) -> dict[str, float]:
        total = max(len(self.samples), 1)
        return {
            "replay_size": float(len(self.samples)),
            "replay_post_fraction": sum(
                item.source == "transition_post" for item in self.samples
            ) / total,
            "replay_priority_mean": float(np.mean([
                item.priority for item in self.samples
            ])) if self.samples else 0.0,
        }


def schedule(start: float, end: float, round_index: int, rounds: int) -> float:
    fraction = round_index / max(rounds - 1, 1)
    return float(start + fraction * (end - start))


def load_archive(settings: dict[str, Any]) -> RaceResetArchive | None:
    path = settings.get("archive_checkpoint")
    if not path:
        return None
    archive = RaceResetArchive({
        "enabled": True,
        "capacity": int(settings.get("archive_capacity", 4096)),
        "procedural_probability": 0.0,
        "gate_probability": 0.0,
        "failure_probability": 0.0,
        "intergate_probability": 0.0,
        "transition_pre_probability": 0.0,
        "transition_post_probability": 1.0,
        "minimum_source_outcomes": 1,
        "minimum_replay_scale": 1.0,
        "position_noise_gate_frame": [0.0, 0.0, 0.0],
        "velocity_noise_gate_frame": [0.0, 0.0, 0.0],
        "attitude_noise_degrees": 0.0,
        "body_rate_noise": 0.0,
    })
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    archive.load_state_dict(checkpoint.get("reset_archive"))
    archive._outcomes["transition_post"].clear()
    return archive


@torch.no_grad()
def encode_history(
    history: ObservationHistory, encoder, action_tokenizer, actor: TokenizerFlowPolicy,
    *, device: str, amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = history.batch(device, repeat_first_padding=True)
    observation, actions = tokenizer_policy_tokens(
        encoder, action_tokenizer, batch,
        history_raw_steps=actor.history_steps * encoder.temporal_patch_size,
        amp=amp,
    )
    return observation.float(), actions.float(), batch["previous_action"][:, -1]


@torch.no_grad()
def learner_chunk(
    actor: TokenizerFlowPolicy, observation: torch.Tensor, actions: torch.Tensor,
    previous: torch.Tensor, settings: dict[str, Any], *, deterministic: bool,
) -> tuple[np.ndarray, float]:
    if deterministic:
        chunk = actor.sample(
            observation, actions,
            steps=int(settings.get("sampling_steps", 1)), method="euler",
            previous_action=previous, deterministic=True,
        )[0]
        return chunk.float().cpu().numpy(), 0.0
    count = int(settings.get("uncertainty_samples", 4))
    samples = actor.sample(
        observation.expand(count, -1, -1, -1),
        actions.expand(count, -1, -1, -1),
        steps=int(settings.get("sampling_steps", 1)), method="euler",
        previous_action=previous.expand(count, -1), deterministic=False,
    ).float()
    uncertainty = float(samples[:, 0].std(0, unbiased=False).mean())
    return samples.mean(0).cpu().numpy(), uncertainty


def expert_label(command, *, strict_solver: bool) -> tuple[np.ndarray | None, bool, bool]:
    """Return a normalized label only when the teacher contract is clean.

    DAgger deliberately visits learner-induced recovery states.  An MPCC
    fallback at one of those states can be finite while no longer being an
    optimizer solution, so it must not silently become a distillation target.
    """

    solver_failed = int(command.solver_status) != 0
    valid = bool(command.valid) and (not strict_solver or not solver_failed)
    action = command.action.as_array()
    valid = valid and bool(np.all(np.isfinite(action)))
    return (ctbr_to_normalized(action) if valid else None), valid, solver_failed


def cached_teacher(
    env: FlightmareEnv, settings: dict[str, Any], action_delay: float
):
    """Reuse one generated ACADOS solver per track within the sequential run."""

    key = (env.track.fingerprint, float(action_delay))
    teacher = _TEACHER_CACHE.get(key)
    if teacher is None:
        teacher = make_mpcc(env, settings, action_delay)
        _TEACHER_CACHE[key] = teacher
    teacher.reset()
    return teacher


def episode_spawn(
    env: FlightmareEnv, stage: RacingCurriculumStage, archive: RaceResetArchive | None,
    *, source_post: bool, seed: int, episode_index: int,
):
    if source_post and archive is not None:
        return archive.sample_spawn(
            env.track, stage, seed=seed, episode_index=episode_index
        )
    return sample_curriculum_spawn(
        env.track, stage, seed=seed, episode_index=episode_index
    )


@torch.no_grad()
def collect_episode(
    actor: TokenizerFlowPolicy, encoder, action_tokenizer,
    replay: PrioritizedReplay, archive: RaceResetArchive | None,
    stage: RacingCurriculumStage, settings: dict[str, Any], *,
    round_index: int, episode_index: int, device: str, evaluation: bool,
) -> dict[str, float]:
    seed = int(settings.get("seed", 0)) + 1000003 * round_index + 7919 * episode_index
    rng = np.random.default_rng(seed)
    track_name = stage.tracks[episode_index % len(stage.tracks)]
    env = FlightmareEnv(
        track=track_name, next_gates=3, image_size=(160, 128),
        control_dt=1.0 / 90.0, render_observations=False,
        mask_source="geometry", mask_size=(160, 128),
        geometry_renderer=str(settings.get("geometry_renderer", "exact")),
        image_delay=float(settings.get("image_delay", 0.033)),
        action_delay=stage.action_delay, terminate_on_collision=True,
    )
    rounds = int(settings.get("rounds", 12))
    post_fraction = float(settings.get("post_spawn_fraction", 0.5))
    source_post = bool(not evaluation and archive is not None and rng.random() < post_fraction)
    spawn = episode_spawn(
        env, stage, archive, source_post=source_post, seed=seed,
        episode_index=episode_index + round_index * int(settings.get("episodes_per_round", 24)),
    )
    source = str(spawn.metadata.get("archive_kind", "procedural"))
    options: dict[str, Any] = {
        "gate_index": spawn.gate_index, "state": spawn.state,
        "spawn": dict(spawn.metadata),
    }
    if spawn.previous_action is not None:
        options["previous_action"] = spawn.previous_action
    observation, _ = env.reset(seed=seed, options=options)
    history = ObservationHistory(
        actor.history_steps * encoder.temporal_patch_size, tuple(encoder.image_size)
    )
    if spawn.observation_history is not None:
        history.restore(dict(spawn.observation_history))
    else:
        history.append(observation)
    expert = None if evaluation else cached_teacher(env, settings, stage.action_delay)
    if (
        expert is not None
        and bool(settings.get("require_acados_teacher", False))
        and "acados" not in expert.source
    ):
        raise RuntimeError(f"DAgger requires an ACADOS teacher, got {expert.source!r}")
    start_passed = env.tracker.passed_count
    target_gates = 2 if evaluation or source == "procedural" else 1
    teacher_beta = schedule(
        float(settings.get("teacher_beta_start", 0.75)),
        float(settings.get("teacher_beta_end", 0.10)), round_index, rounds,
    )
    disagreement_threshold = schedule(
        float(settings.get("disagreement_threshold_start", 0.55)),
        float(settings.get("disagreement_threshold_end", 0.30)), round_index, rounds,
    )
    uncertainty_threshold = schedule(
        float(settings.get("uncertainty_threshold_start", 0.18)),
        float(settings.get("uncertainty_threshold_end", 0.10)), round_index, rounds,
    )
    raw_steps = gates_passed = interventions = decisions = 0
    uncertainty_sum = disagreement_sum = 0.0
    teacher_queries = teacher_invalid = teacher_solver_failures = teacher_recovery = 0
    crashed = success = False
    after_gate_steps = -1
    amp = bool(settings.get("amp", True))
    max_steps = int(settings.get("max_steps", stage.max_steps))
    try:
        while raw_steps < max_steps and not crashed and not success:
            observation_tokens, action_tokens, previous = encode_history(
                history, encoder, action_tokenizer, actor, device=device, amp=amp
            )
            student, uncertainty = learner_chunk(
                actor, observation_tokens, action_tokens, previous, settings,
                deterministic=evaluation,
            )
            labels: list[np.ndarray] = []
            teacher_first = None
            first_label_valid = False
            strict_solver = bool(settings.get("require_solver_clean_labels", True))
            if expert is not None:
                first_command = expert(observation)
                teacher_queries += 1
                teacher_first, first_label_valid, solver_failed = expert_label(
                    first_command, strict_solver=strict_solver
                )
                teacher_invalid += int(not first_label_valid)
                teacher_solver_failures += int(solver_failed)
                teacher_recovery += int(
                    int(np.asarray(first_command.diagnostics.get("mode", 0)).item()) == 1
                )
                if first_label_valid and teacher_first is not None:
                    disagreement = float(
                        np.sqrt(np.mean((student[0] - teacher_first) ** 2))
                    )
                    robot_gate = (
                        uncertainty > uncertainty_threshold
                        or disagreement > disagreement_threshold
                    )
                    intervention = bool(rng.random() < teacher_beta or robot_gate)
                else:
                    disagreement = 0.0
                    intervention = False
                uncertainty_sum += uncertainty
                disagreement_sum += disagreement
            else:
                disagreement = 0.0
                intervention = False
            decisions += 1
            interventions += int(intervention)
            gate_before = env.tracker.passed_count
            labels_clean = first_label_valid
            for offset in range(actor.action_horizon):
                if expert is not None:
                    if offset == 0:
                        teacher = teacher_first
                        label_valid = first_label_valid
                    else:
                        command = expert(observation)
                        teacher_queries += 1
                        teacher, label_valid, solver_failed = expert_label(
                            command, strict_solver=strict_solver
                        )
                        teacher_invalid += int(not label_valid)
                        teacher_solver_failures += int(solver_failed)
                        teacher_recovery += int(
                            int(np.asarray(command.diagnostics.get("mode", 0)).item()) == 1
                        )
                    labels_clean = labels_clean and label_valid
                    if label_valid and teacher is not None:
                        labels.append(teacher)
                    executed = (
                        teacher.copy()
                        if intervention and label_valid and teacher is not None
                        else student[offset].copy()
                    )
                    if intervention and label_valid and teacher is not None:
                        noise = float(settings.get("teacher_execution_noise", 0.025))
                        executed = np.clip(executed + rng.normal(0.0, noise, 4), -1.0, 1.0)
                else:
                    executed = student[offset]
                executed_ctbr = normalized_to_ctbr(executed)
                observation, _, terminated, _, info = env.step(executed_ctbr)
                if expert is not None:
                    expert.observe_executed_action(executed_ctbr)
                history.append(observation)
                raw_steps += 1
                gates_passed = env.tracker.passed_count - start_passed
                if bool(info.get("gate_passed")):
                    after_gate_steps = 0
                elif after_gate_steps >= 0:
                    after_gate_steps += 1
                crashed = bool(
                    info.get("ground_contact") or info.get("unity_collision") or terminated
                )
                success = gates_passed >= target_gates
                if crashed or success or raw_steps >= max_steps:
                    break
            if expert is not None and labels_clean and len(labels) == actor.action_horizon:
                while len(labels) < actor.action_horizon:
                    labels.append(labels[-1].copy())
                transition = bool(
                    source == "transition_post" or gate_before != env.tracker.passed_count
                    or 0 <= after_gate_steps <= int(settings.get("transition_priority_steps", 90))
                )
                priority = 1.0
                priority *= float(settings.get("post_priority", 4.0)) if transition else 1.0
                priority *= float(settings.get("intervention_priority", 2.0)) if intervention else 1.0
                priority *= 1.0 + min(2.0, uncertainty + disagreement)
                replay.add(DAggerSample(
                    observation_tokens=observation_tokens[0].half().cpu(),
                    action_tokens=action_tokens[0].half().cpu(),
                    previous_action=previous[0].float().cpu(),
                    target=torch.from_numpy(np.stack(labels)).float(),
                    priority=float(priority), source=source,
                ))
    finally:
        env.close()
    return {
        "success": float(success), "gates_passed": float(gates_passed),
        "at_least_one_gate": float(gates_passed >= 1),
        "at_least_two_gates": float(gates_passed >= 2),
        "crash": float(crashed), "raw_steps": float(raw_steps),
        "post_source": float(source == "transition_post"),
        "teacher_fraction": interventions / max(decisions, 1),
        "teacher_label_valid_fraction": 1.0 - teacher_invalid / max(teacher_queries, 1),
        "teacher_solver_failure_fraction": teacher_solver_failures / max(teacher_queries, 1),
        "teacher_recovery_fraction": teacher_recovery / max(teacher_queries, 1),
        "uncertainty": uncertainty_sum / max(decisions, 1),
        "disagreement": disagreement_sum / max(decisions, 1),
        "teacher_beta": float(teacher_beta),
    }


def mean_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    keys = results[0].keys()
    return {key: float(np.mean([result[key] for result in results])) for key in keys}


def stabilization_loss(
    actor: TokenizerFlowPolicy, batch: dict[str, torch.Tensor], settings: dict[str, Any]
) -> torch.Tensor:
    target = batch["target"]
    sigma = float(settings.get("stabilization_noise", 0.08))
    remaining = float(settings.get("stabilization_remaining_time", 0.25))
    perturbed = (target + sigma * torch.randn_like(target)).clamp(-1.0, 1.0)
    flow_time = target.new_full((len(target),), 1.0 - remaining)
    step_size = target.new_full((len(target),), remaining)
    context = actor.encode_context(batch["observation_tokens"], batch["action_tokens"])
    velocity = actor.velocity_from_context(perturbed, flow_time, step_size, context)
    correction = (target - perturbed) / remaining
    return (velocity.float() - correction.float()).square().mean()


def train_updates(
    actor: TokenizerFlowPolicy, optimizer: torch.optim.Optimizer,
    replay: PrioritizedReplay, settings: dict[str, Any], device: str,
) -> dict[str, float]:
    actor.train()
    totals: dict[str, list[float]] = {}
    amp = bool(settings.get("amp", True))
    updates = int(settings.get("updates_per_round", 750))
    for _ in range(updates):
        batch = replay.sample(int(settings.get("batch_size", 64)), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=device.startswith("cuda") and amp,
        ):
            flow, metrics = actor.shortcut_training_loss(
                batch["target"], batch["observation_tokens"], batch["action_tokens"],
                previous_action=batch["previous_action"],
                direct_weight=float(settings.get("shortcut_direct_weight", 0.5)),
                bootstrap_weight=float(settings.get("shortcut_bootstrap_weight", 1.0)),
            )
            stabilization = stabilization_loss(actor, batch, settings)
            loss = flow + float(settings.get("stabilization_weight", 0.15)) * stabilization
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            actor.parameters(), float(settings.get("grad_clip", 2.0))
        )
        optimizer.step()
        values = {
            **metrics, "dagger_loss": loss.detach(),
            "stabilization_loss": stabilization.detach(),
            "gradient_norm": gradient.detach(),
        }
        for key, value in values.items():
            totals.setdefault(key, []).append(float(value))
    return {key: float(np.mean(value)) for key, value in totals.items()}


def main() -> None:
    args, config = arguments()
    settings = config["flow_dagger"]
    device = str(args.device)
    seed = int(settings.get("seed", 20260850))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("high")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

    initial_path = Path(settings["actor"])
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    actor_config = dict(initial["model_config"])
    actor_config["source_mode"] = "previous_action"
    actor_config["source_noise"] = float(settings.get("source_noise", 0.25))
    actor = TokenizerFlowPolicy(**actor_config).to(device)
    actor.load_state_dict(initial["model"])
    loaded = load_frozen_actor_stack(
        None, Path(initial["observation_tokenizer"]),
        Path(initial["action_tokenizer"]), device,
    )
    _, encoder, action_tokenizer, _, observation_path, action_path, _ = loaded
    encoder.eval().requires_grad_(False).to(device)
    action_tokenizer.eval().requires_grad_(False).to(device)
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=float(settings.get("learning_rate", 5e-5)),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
        fused=device.startswith("cuda"),
    )
    replay = PrioritizedReplay(int(settings.get("replay_capacity", 60000)), seed)
    archive = load_archive(settings)
    stage = RacingCurriculumStage(**settings["stage"])
    manager = CheckpointManager.from_config(config)
    logger = init_wandb(config)
    rounds = int(settings.get("rounds", 12))
    episodes = int(settings.get("episodes_per_round", 24))
    eval_episodes = int(settings.get("evaluation_episodes", 24))

    print(
        f"flow_dagger actor={initial_path} parameters="
        f"{sum(parameter.numel() for parameter in actor.parameters()):,} "
        f"source=previous_action/{actor.source_noise:.3f} rounds={rounds} "
        f"episodes_per_round={episodes} archive={settings.get('archive_checkpoint')}",
        flush=True,
    )
    global_step = 0
    for round_index in range(rounds):
        started = time.perf_counter()
        actor.eval()
        collection = [
            collect_episode(
                actor, encoder, action_tokenizer, replay, archive, stage, settings,
                round_index=round_index, episode_index=index, device=device,
                evaluation=False,
            )
            for index in range(episodes)
        ]
        collection_metrics = mean_metrics(collection)
        update_metrics = train_updates(actor, optimizer, replay, settings, device)
        actor.eval()
        evaluation = [
            collect_episode(
                actor, encoder, action_tokenizer, replay, None, stage, settings,
                round_index=round_index, episode_index=100000 + index,
                device=device, evaluation=True,
            )
            for index in range(eval_episodes)
        ]
        evaluation_metrics = mean_metrics(evaluation)
        global_step += episodes + int(settings.get("updates_per_round", 750))
        selection = (
            evaluation_metrics["at_least_two_gates"]
            + 0.10 * evaluation_metrics["at_least_one_gate"]
            + 0.05 * collection_metrics["success"]
        )
        elapsed = time.perf_counter() - started
        train_logged = {
            **{f"collect_{key}": value for key, value in collection_metrics.items()},
            **update_metrics, **replay.metrics(), "round_seconds": elapsed,
        }
        eval_logged = {
            **evaluation_metrics, "selection_score": selection,
            "round": float(round_index + 1),
        }
        logger.log_train(train_logged, global_step)
        logger.log_eval(eval_logged, global_step)
        print(
            f"round={round_index + 1}/{rounds} replay={len(replay.samples)} "
            f"collect_success={collection_metrics['success']:.3f} "
            f"teacher={collection_metrics['teacher_fraction']:.3f} "
            f"teacher_valid={collection_metrics['teacher_label_valid_fraction']:.4f} "
            f"teacher_solver_fail={collection_metrics['teacher_solver_failure_fraction']:.4f} "
            f"uncertainty={collection_metrics['uncertainty']:.3f} "
            f"disagreement={collection_metrics['disagreement']:.3f} "
            f"eval_gate1={evaluation_metrics['at_least_one_gate']:.3f} "
            f"eval_gate2={evaluation_metrics['at_least_two_gates']:.3f} "
            f"loss={update_metrics['dagger_loss']:.4f} seconds={elapsed:.1f}",
            flush=True,
        )
        manager.save_eval(
            eval_logged, step=global_step,
            summary="Robot-gated DAgger with MPCC labels and stabilizing shortcut flow.",
        )
        manager.save({
            "model": actor.state_dict(), "optimizer": optimizer.state_dict(),
            "model_config": actor_config, "training_config": config,
            "observation_tokenizer": str(observation_path),
            "action_tokenizer": str(action_path), "rng_state": capture_rng_state(),
            "round": round_index + 1, "step": global_step,
            "metrics": eval_logged,
            "input_contract": {
                **dict(initial.get("input_contract", {})),
                "imitation_algorithm": "robot_gated_dagger_stabilizing_flow",
                "flow_source": "narrow_previous_action_gaussian",
            },
            "teacher_contract": {
                "backend": str(settings.get("mpcc_backend", "predictive")),
                "require_acados": bool(settings.get("require_acados_teacher", False)),
                "require_solver_clean_labels": bool(
                    settings.get("require_solver_clean_labels", True)
                ),
                "nominal_speed": float(settings.get("mpcc_nominal_speed", 6.0)),
                "racing_line_aperture_fraction": float(
                    settings.get("racing_line_aperture_fraction", 0.20)
                ),
            },
        }, step=global_step, metrics=eval_logged, rank=True)
    logger.finish()


if __name__ == "__main__":
    main()
