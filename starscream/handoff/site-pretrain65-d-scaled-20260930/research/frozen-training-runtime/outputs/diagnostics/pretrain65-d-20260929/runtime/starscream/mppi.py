"""Policy-guided MPPI planning over the frozen Starscream world model.

The planner follows the TD-MPC2 inference pattern: shift the previous Gaussian
plan, mix policy rollouts with Gaussian candidates, score short model rollouts
with a terminal learned value, and refit the distribution to weighted elites.
Starscream's PMPO checkpoint currently provides V(s), not Q(s, a), so the
terminal bootstrap is deliberately named and reported as a value bootstrap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from dreamerv4 import TwoHot
from starscream.actor_critic import (
    ActionValueEnsemble,
    CriticWithTarget,
    DistributionalActionValueCritic,
    DistributionalCritic,
    GaussianActor,
    build_actor_state_from_latents,
)
from starscream.dynamics import RaceDynamics
from starscream.tokenizer import ActionTokenizer, MultiModalTokenizer


@dataclass(frozen=True)
class MPPIConfig:
    horizon: int = 3
    population: int = 48
    iterations: int = 2
    particles: int = 2
    policy_fraction: float = 0.125
    elite_fraction: float = 0.20
    temperature: float = 0.50
    momentum: float = 0.10
    initial_std: float = 0.35
    min_std: float = 0.05
    max_std: float = 0.70
    policy_noise_scale: float = 0.50
    discount: float = 0.997
    continuation_floor: float = 0.01
    continuation_ceiling: float = 0.995
    terminal_value_weight: float = 1.0
    risk_penalty: float = 0.25
    prior_penalty: float = 0.04
    slew_penalty: float = 0.03
    saturation_penalty: float = 0.01
    stochastic_dynamics: bool = True
    action_timing_steps: tuple[float, float, float] = (1.0, 1.0, 1.0)
    seed: int = 20261201

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "MPPIConfig":
        raw = dict(values or {})
        if "action_timing_steps" in raw:
            raw["action_timing_steps"] = tuple(raw["action_timing_steps"])
        result = cls(**raw)
        if result.horizon < 1 or result.population < 2 or result.iterations < 1:
            raise ValueError("MPPI horizon/iterations must be positive and population >= 2")
        if result.particles < 1:
            raise ValueError("MPPI particles must be positive")
        if not 0.0 <= result.policy_fraction <= 1.0:
            raise ValueError("MPPI policy_fraction must be in [0,1]")
        if not 0.0 < result.elite_fraction <= 1.0:
            raise ValueError("MPPI elite_fraction must be in (0,1]")
        if result.temperature <= 0.0:
            raise ValueError("MPPI temperature must be positive")
        if not 0.0 <= result.momentum < 1.0:
            raise ValueError("MPPI momentum must be in [0,1)")
        if not 0.0 < result.min_std <= result.initial_std <= result.max_std:
            raise ValueError("MPPI std bounds must contain initial_std")
        if len(result.action_timing_steps) != 3:
            raise ValueError("action_timing_steps must have [dt, delay, valid]")
        return result


@dataclass
class BeliefState:
    latents: torch.Tensor
    action_history: torch.Tensor
    timing_history: torch.Tensor
    current_packed: torch.Tensor

    def repeat(self, count: int) -> "BeliefState":
        return BeliefState(
            latents=self.latents.repeat_interleave(count, dim=0),
            action_history=self.action_history.repeat_interleave(count, dim=0),
            timing_history=self.timing_history.repeat_interleave(count, dim=0),
            current_packed=self.current_packed.repeat_interleave(count, dim=0),
        )


def weighted_elite_update(
    candidates: torch.Tensor,
    scores: torch.Tensor,
    *,
    elite_fraction: float,
    temperature: float,
    minimum_std: float,
    maximum_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return softmax-weighted elite mean/std and their original indices."""

    if candidates.ndim < 2 or scores.shape != (candidates.shape[0],):
        raise ValueError("candidates and scores must share their leading population axis")
    elite_count = max(2, min(len(scores), int(math.ceil(len(scores) * elite_fraction))))
    elite_scores, indices = scores.topk(elite_count, sorted=False)
    elite = candidates.index_select(0, indices)
    weights = torch.softmax(
        (elite_scores - elite_scores.max()) / float(temperature), dim=0
    )
    view = (elite_count,) + (1,) * (elite.ndim - 1)
    weights = weights.view(view)
    mean = (weights * elite).sum(dim=0)
    variance = (weights * (elite - mean).square()).sum(dim=0)
    std = variance.clamp_min(float(minimum_std) ** 2).sqrt().clamp_max(maximum_std)
    return mean, std, indices


class MPPIPlanner:
    """Vectorized short-horizon MPPI using the exact PMPO imagination contract."""

    def __init__(
        self,
        *,
        actor: GaussianActor,
        encoder: MultiModalTokenizer,
        action_tokenizer: ActionTokenizer,
        dynamics: RaceDynamics,
        reward_coder: TwoHot,
        critic: CriticWithTarget | ActionValueEnsemble | None,
        config: MPPIConfig,
        device: str,
        amp: bool,
    ) -> None:
        self.actor = actor
        self.encoder = encoder
        self.action_tokenizer = action_tokenizer
        self.dynamics = dynamics
        self.reward_coder = reward_coder
        self.critic = critic
        self.config = config
        self.device = device
        self.amp = bool(amp)
        self.context_length = int(actor.temporal_context)
        self.patch = int(actor.action_horizon)
        self.action_dim = int(actor.action_dim)
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._calls = 0
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(int(config.seed))

    def reset(self) -> None:
        self._mean = None
        self._std = None
        self._calls = 0
        self._generator.manual_seed(int(self.config.seed))

    def _randn_like(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=self._generator,
        )

    @property
    def settings(self) -> dict[str, Any]:
        return asdict(self.config)

    def _future_timing(self, batch: int, reference: torch.Tensor) -> torch.Tensor:
        timing = reference.new_tensor(self.config.action_timing_steps)
        return timing.view(1, 1, 3).expand(batch, self.patch, 3)

    def _posterior_belief(self, batch: dict[str, torch.Tensor]) -> BeliefState:
        history = self.context_length * self.encoder.temporal_patch_size
        observation = {
            key: batch[key][:, :history] for key in ("mask", "proprio", "route", "timing")
        }
        latents = self.encoder.encode(observation)
        batch_size = len(latents)
        action_history = batch["previous_action"][:, :history].reshape(
            batch_size, self.context_length, self.patch, self.action_dim
        )
        timing_history = batch["timing"][:, :history].reshape(
            batch_size, self.context_length, self.patch, self.encoder.timing_dim
        )
        return BeliefState(
            latents=latents,
            action_history=action_history,
            timing_history=timing_history,
            current_packed=self.dynamics.pack(latents[:, -1:]),
        )

    def _transition(
        self, belief: BeliefState, action: torch.Tensor
    ) -> tuple[BeliefState, torch.Tensor, torch.Tensor]:
        batch_size = len(action)
        future_action_timing = self._future_timing(batch_size, action)
        encoded_action = self.action_tokenizer(action, future_action_timing)
        macro_action = encoded_action.mean.mean(dim=2)
        source = (
            self._randn_like(belief.current_packed)
            if self.config.stochastic_dynamics
            else torch.zeros_like(belief.current_packed)
        )
        one_jump = int(math.log2(self.dynamics.k_max))
        step_indices = torch.full(
            (batch_size, 1), one_jump, device=action.device, dtype=torch.long
        )
        signal_indices = torch.zeros_like(step_indices)
        output = self.dynamics(
            source,
            macro_action,
            step_indices,
            signal_indices,
            initial_context=belief.current_packed,
            action_context=encoded_action.context,
            causal_transition=True,
        )
        next_packed = output.predicted_packed_latents.clone()
        next_latent = self.dynamics.unpack(next_packed)
        future_observation_timing = belief.timing_history[:, -1:].clone()
        next_belief = BeliefState(
            latents=torch.cat([belief.latents, next_latent], dim=1)[
                :, -self.context_length:
            ],
            action_history=torch.cat(
                [belief.action_history, action[:, None]], dim=1
            )[:, -self.context_length:],
            timing_history=torch.cat(
                [belief.timing_history, future_observation_timing], dim=1
            )[:, -self.context_length:],
            current_packed=next_packed,
        )
        reward = self.reward_coder.mean(output.reward_logits.float()).squeeze(1)
        continuation = output.continue_logits.float().sigmoid().squeeze(1).clamp(
            self.config.continuation_floor, self.config.continuation_ceiling
        )
        return next_belief, reward, continuation

    def _actor_state(self, belief: BeliefState) -> torch.Tensor:
        return build_actor_state_from_latents(
            self.encoder,
            belief.latents,
            belief.action_history,
            belief.timing_history,
        )[:, -self.context_length:]

    def _policy_trajectories(
        self, initial: BeliefState, count: int
    ) -> torch.Tensor:
        if count < 1:
            return initial.latents.new_empty(
                0, self.config.horizon, self.patch, self.action_dim
            )
        belief = initial.repeat(count)
        chunks = []
        for horizon_index in range(self.config.horizon):
            output = self.actor(self._actor_state(belief))
            noise = self._randn_like(output.mean) * output.std
            action = (
                output.mean
                + float(self.config.policy_noise_scale) * noise
            ).clamp(-1.0, 1.0)
            if horizon_index == 0:
                action[0] = output.mean[0]
            chunks.append(action)
            belief, _, _ = self._transition(belief, action)
        return torch.stack(chunks, dim=1)

    def _score(
        self,
        initial: BeliefState,
        candidates: torch.Tensor,
        prior: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        population = len(candidates)
        particles = self.config.particles
        belief = initial.repeat(population * particles)
        actions = candidates[:, None].expand(
            population, particles, *candidates.shape[1:]
        ).reshape(population * particles, *candidates.shape[1:])
        returns = actions.new_zeros(population * particles)
        survival = actions.new_ones(population * particles)
        discount = 1.0
        for horizon_index in range(self.config.horizon):
            belief, reward, continuation = self._transition(
                belief, actions[:, horizon_index]
            )
            returns.add_(float(discount) * survival * reward)
            survival = survival * continuation
            discount *= self.config.discount
        reward_returns = returns.clone()
        terminal_bootstrap = torch.zeros_like(returns)
        if self.critic is not None and self.config.terminal_value_weight:
            terminal_state = self._actor_state(belief)
            if isinstance(self.critic, ActionValueEnsemble):
                terminal_action = self.actor(terminal_state).mode()
                terminal_value = self.critic.values(
                    terminal_state, terminal_action, target=True
                ).min(dim=0).values.float()
            else:
                terminal_value = self.critic.target.value(terminal_state).float()
            terminal_bootstrap = float(discount) * survival * terminal_value
            returns.add_(float(self.config.terminal_value_weight) * terminal_bootstrap)

        particle_returns = returns.view(population, particles)
        model_mean = particle_returns.mean(dim=1)
        model_std = (
            particle_returns.std(dim=1, unbiased=False)
            if particles > 1 else torch.zeros_like(model_mean)
        )
        reward_return_mean = reward_returns.view(population, particles).mean(dim=1)
        terminal_bootstrap_mean = terminal_bootstrap.view(
            population, particles
        ).mean(dim=1)
        prior_cost = (candidates - prior).square().mean(dim=(1, 2, 3))
        previous = initial.action_history[0, -1, -1]
        flattened = candidates.flatten(1, 2)
        first_slew = (flattened[:, 0] - previous).square().mean(dim=1)
        internal_slew = (
            (flattened[:, 1:] - flattened[:, :-1]).square().mean(dim=(1, 2))
            if flattened.shape[1] > 1 else torch.zeros_like(first_slew)
        )
        slew_cost = first_slew + internal_slew
        saturation_cost = (candidates.abs() >= 0.98).float().mean(dim=(1, 2, 3))
        score = (
            model_mean
            - self.config.risk_penalty * model_std
            - self.config.prior_penalty * prior_cost
            - self.config.slew_penalty * slew_cost
            - self.config.saturation_penalty * saturation_cost
        )
        return score, {
            "model_return_mean": model_mean,
            "model_return_std": model_std,
            "reward_return_mean": reward_return_mean,
            "terminal_bootstrap_mean": terminal_bootstrap_mean,
            "prior_cost": prior_cost,
            "slew_cost": slew_cost,
            "saturation_cost": saturation_cost,
        }

    @torch.no_grad()
    def plan(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        if len(batch["mask"]) != 1:
            raise ValueError("online MPPI currently plans one environment at a time")
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda") and self.amp,
        ):
            initial = self._posterior_belief(batch)
            policy_count = int(round(
                self.config.population * self.config.policy_fraction
            ))
            policy_rollouts = self._policy_trajectories(
                initial, max(1, policy_count)
            )
            prior = policy_rollouts[0].float()
            policy = policy_rollouts[:policy_count]
            if self._mean is None:
                mean = prior.clone()
                std = torch.full_like(mean, self.config.initial_std)
            else:
                mean = torch.cat([self._mean[1:], prior[-1:]], dim=0)
                std = torch.cat(
                    [self._std[1:], torch.full_like(self._std[-1:], self.config.initial_std)],
                    dim=0,
                )

            last_scores = None
            last_details = None
            for _ in range(self.config.iterations):
                gaussian_count = self.config.population - policy_count
                gaussian = (
                    mean[None]
                    + std[None] * torch.randn(
                        gaussian_count,
                        *mean.shape,
                        device=mean.device,
                        dtype=mean.dtype,
                        generator=self._generator,
                    )
                ).clamp(-1.0, 1.0)
                if gaussian_count:
                    gaussian[0] = mean
                candidates = torch.cat([policy.float(), gaussian], dim=0)
                scores, details = self._score(initial, candidates, prior)
                fitted_mean, fitted_std, _ = weighted_elite_update(
                    candidates,
                    scores,
                    elite_fraction=self.config.elite_fraction,
                    temperature=self.config.temperature,
                    minimum_std=self.config.min_std,
                    maximum_std=self.config.max_std,
                )
                mean = self.config.momentum * mean + (1.0 - self.config.momentum) * fitted_mean
                std = self.config.momentum * std + (1.0 - self.config.momentum) * fitted_std
                mean.clamp_(-1.0, 1.0)
                std.clamp_(self.config.min_std, self.config.max_std)
                last_scores, last_details = scores, details

        self._mean = mean.detach()
        self._std = std.detach()
        self._calls += 1
        assert last_scores is not None and last_details is not None
        best = int(last_scores.argmax())
        reward_ranking = last_details["reward_return_mean"].float()
        value_ranking = last_details["terminal_bootstrap_mean"].float()
        reward_centered = reward_ranking - reward_ranking.mean()
        value_centered = value_ranking - value_ranking.mean()
        rank_correlation = (
            (reward_centered * value_centered).mean()
            / (
                reward_centered.square().mean().sqrt()
                * value_centered.square().mean().sqrt()
            ).clamp_min(1e-8)
            if value_ranking.abs().max() > 0 else value_ranking.new_zeros(())
        )
        diagnostics = {
            "planner_calls": float(self._calls),
            "predicted_score": float(last_scores[best]),
            "predicted_return": float(last_details["model_return_mean"][best]),
            "predicted_reward_return": float(
                last_details["reward_return_mean"][best]
            ),
            "terminal_bootstrap": float(
                last_details["terminal_bootstrap_mean"][best]
            ),
            "reward_value_candidate_correlation": float(rank_correlation),
            "rollout_std": float(last_details["model_return_std"][best]),
            "prior_cost": float(last_details["prior_cost"][best]),
            "slew_cost": float(last_details["slew_cost"][best]),
            "plan_std": float(std[0].mean()),
        }
        return mean[0].float(), std[0].float(), diagnostics


@dataclass
class MPPIFlightPolicy:
    path: Path
    name: str
    step: int
    actor: GaussianActor
    encoder: MultiModalTokenizer
    history: int
    patch: int
    device: str
    amp: bool
    contract: dict[str, Any]
    planner: MPPIPlanner
    last_diagnostics: dict[str, float] | None = None

    def reset(self) -> None:
        self.planner.reset()
        self.last_diagnostics = None

    @torch.no_grad()
    def action_chunk(self, history, *, sample: bool) -> tuple[np.ndarray, np.ndarray, float]:
        del sample  # MPPI returns its refitted mean; exploration occurs inside planning.
        batch = history.batch(self.device)
        started = time.perf_counter()
        action, std, diagnostics = self.planner.plan(batch)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.last_diagnostics = diagnostics
        return action.cpu().numpy(), std.cpu().numpy(), latency_ms


def build_mppi_flight_policy(
    base_policy,
    *,
    config_path: Path,
    settings: dict[str, Any],
    world_model_path: Path,
    action_value_path: Path | None = None,
) -> MPPIFlightPolicy:
    """Attach MPPI components to an already validated actor flight policy."""

    actor_checkpoint = torch.load(base_policy.path, map_location="cpu", weights_only=False)
    recorded_world_model = actor_checkpoint.get("world_model_checkpoint")
    if recorded_world_model and Path(recorded_world_model) != world_model_path:
        raise ValueError(
            "MPPI world model does not match the actor's training contract: "
            f"{world_model_path} != {recorded_world_model}"
        )
    dynamics_checkpoint = torch.load(world_model_path, map_location="cpu", weights_only=False)
    dynamics = RaceDynamics(**dynamics_checkpoint["model_config"])
    dynamics.load_state_dict(dynamics_checkpoint["model"])
    dynamics.eval().requires_grad_(False).to(base_policy.device)
    if not dynamics.causal_residual or not dynamics.pmpo_auxiliary_heads:
        raise ValueError("MPPI requires causal residual dynamics with PMPO outcome heads")
    if dynamics.n_latents != base_policy.encoder.n_latents:
        raise ValueError("world model and observation tokenizer latent counts differ")

    action_path = Path(actor_checkpoint["action_tokenizer"])
    action_checkpoint = torch.load(action_path, map_location="cpu", weights_only=False)
    raw_action_config = action_checkpoint.get("model_config", {})
    action_config = dict(raw_action_config.get("tokenizer", raw_action_config))
    action_tokenizer = ActionTokenizer(**action_config)
    action_tokenizer.load_state_dict(
        action_checkpoint.get("tokenizer", action_checkpoint["model"])
    )
    action_tokenizer.eval().requires_grad_(False).to(base_policy.device)

    planner_config = MPPIConfig.from_dict(settings)
    critic = None
    if planner_config.terminal_value_weight:
        if action_value_path is not None:
            action_value_checkpoint = torch.load(
                action_value_path, map_location="cpu", weights_only=False
            )
            recorded_actor = action_value_checkpoint.get("actor_checkpoint")
            if recorded_actor and Path(recorded_actor) != base_policy.path:
                raise ValueError("action-value checkpoint was trained for another actor")
            critic_config = dict(action_value_checkpoint["model_config"])
            critic = ActionValueEnsemble(
                [
                    DistributionalActionValueCritic(**critic_config)
                    for _ in range(int(action_value_checkpoint["ensemble_size"]))
                ]
            )
            critic.load_state_dict(action_value_checkpoint["model"])
        else:
            if "critic" not in actor_checkpoint or "critic_config" not in actor_checkpoint:
                raise ValueError(
                    "terminal bootstrap requested but no action-value or PMPO critic exists"
                )
            critic = CriticWithTarget(
                DistributionalCritic(**actor_checkpoint["critic_config"])
            )
            critic.load_state_dict(actor_checkpoint["critic"])
        critic.eval().requires_grad_(False).to(base_policy.device)
    reward_coder = TwoHot(
        dynamics_checkpoint["model_config"].get("reward_bins", 255)
    ).to(base_policy.device)
    planner = MPPIPlanner(
        actor=base_policy.actor,
        encoder=base_policy.encoder,
        action_tokenizer=action_tokenizer,
        dynamics=dynamics,
        reward_coder=reward_coder,
        critic=critic,
        config=planner_config,
        device=base_policy.device,
        amp=base_policy.amp,
    )
    contract = dict(base_policy.contract)
    contract.update(
        planner="tdmpc2_style_policy_guided_mppi",
        world_model_checkpoint=str(world_model_path),
        terminal_bootstrap=(
            f"action_value_ensemble:{action_value_path}"
            if action_value_path is not None
            else (
                "legacy_pmpo_target_state_value"
                if planner_config.terminal_value_weight else "none"
            )
        ),
        mppi=planner.settings,
    )
    return MPPIFlightPolicy(
        path=base_policy.path,
        name=f"tdmpc2-mppi/{config_path.stem}",
        step=base_policy.step,
        actor=base_policy.actor,
        encoder=base_policy.encoder,
        history=base_policy.history,
        patch=base_policy.patch,
        device=base_policy.device,
        amp=base_policy.amp,
        contract=contract,
        planner=planner,
    )
