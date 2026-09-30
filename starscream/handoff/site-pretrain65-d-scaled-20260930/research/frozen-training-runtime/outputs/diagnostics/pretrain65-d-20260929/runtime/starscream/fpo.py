"""Flow Policy Optimization++ primitives for Starscream action chunks.

The per-sample ratio, loss clamps, straight-through log-ratio clamp, and ASPO
trust region follow the released FPO++ implementation from amazon-far/
fpo-control.  Starscream's adaptation keeps one environment decision equal to
one CTBR action chunk and masks unexecuted tail actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


def clamp_ste(value: torch.Tensor, minimum: float | None = None, maximum: float | None = None) -> torch.Tensor:
    """Clamp in the forward pass and retain the identity backward gradient."""

    clamped = value.clamp(min=minimum, max=maximum)
    return value + (clamped - value).detach()


def sample_cfm_conditions(
    batch: int,
    samples: int,
    horizon: int,
    action_dim: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    time_beta: float = 1.0,
    step_size: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if min(batch, samples, horizon, action_dim) < 1 or time_beta <= 0 or step_size <= 0:
        raise ValueError("CFM condition dimensions and scales must be positive")
    epsilon = torch.randn(batch, samples, horizon, action_dim, device=device, dtype=dtype)
    uniform = torch.rand(batch, samples, device=device, dtype=dtype)
    flow_time = 0.005 + 0.99 * (1.0 - (1.0 - uniform).pow(1.0 / time_beta))
    shortcut_step = torch.full_like(flow_time, float(step_size))
    return epsilon, flow_time, shortcut_step


def conditional_flow_loss(
    actor: nn.Module,
    observation_tokens: torch.Tensor,
    action_tokens: torch.Tensor,
    actions: torch.Tensor,
    epsilon: torch.Tensor,
    flow_time: torch.Tensor,
    shortcut_step: torch.Tensor,
    *,
    valid_steps: torch.Tensor | None = None,
    huber_delta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one CFM loss per action and Monte-Carlo condition.

    Losses average CTBR components and sum executed chunk timesteps, matching
    FPO++ action-chunk ratios without assigning credit to an unexecuted tail.
    """

    if actions.ndim != 3:
        raise ValueError("actions must have shape (B,H,A)")
    batch, horizon, action_dim = actions.shape
    if epsilon.ndim != 4 or epsilon.shape[0] != batch or epsilon.shape[2:] != (horizon, action_dim):
        raise ValueError("epsilon must have shape (B,N,H,A)")
    samples = epsilon.shape[1]
    if flow_time.shape != (batch, samples) or shortcut_step.shape != (batch, samples):
        raise ValueError("flow_time and shortcut_step must have shape (B,N)")
    if valid_steps is None:
        valid_steps = torch.ones(batch, horizon, device=actions.device, dtype=torch.bool)
    if valid_steps.shape != (batch, horizon) or not torch.all(valid_steps.any(dim=1)):
        raise ValueError("each action chunk needs at least one valid executed step")

    context = actor.encode_context(observation_tokens, action_tokens)
    noisy = epsilon + flow_time[..., None, None] * (actions[:, None] - epsilon)
    target = actions[:, None] - epsilon
    expanded_context = context[:, None].expand(batch, samples, *context.shape[1:])
    predicted = actor.velocity_from_context(
        noisy.flatten(0, 1),
        flow_time.flatten(),
        shortcut_step.flatten(),
        expanded_context.flatten(0, 1),
    ).reshape(batch, samples, horizon, action_dim)
    difference = predicted.float() - target.float()
    if huber_delta is None:
        element = difference.square()
    else:
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        absolute = difference.abs()
        element = torch.where(
            absolute <= huber_delta,
            0.5 * difference.square(),
            huber_delta * (absolute - 0.5 * huber_delta),
        )
    per_step = element.mean(dim=-1)
    loss = (per_step * valid_steps[:, None].float()).sum(dim=-1)
    return loss, predicted


@dataclass(frozen=True, slots=True)
class FPOMetrics:
    loss: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_std: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor
    old_cfm_loss: torch.Tensor
    current_cfm_loss: torch.Tensor


def fpo_plus_plus_loss(
    old_cfm_loss: torch.Tensor,
    current_cfm_loss: torch.Tensor,
    advantage: torch.Tensor,
    *,
    clip_epsilon: float = 0.02,
    trust_region: str = "aspo",
    cfm_loss_clamp: float = 20.0,
    negative_cfm_clamp: float = 20.0,
    log_ratio_clamp: float = 10.0,
    log_ratio_gain: float = 1.0,
) -> FPOMetrics:
    """FPO++ per-sample ratio with PPO/SPO/asymmetric trust regions."""

    if old_cfm_loss.shape != current_cfm_loss.shape or old_cfm_loss.ndim != 2:
        raise ValueError("old/current CFM loss must share shape (B,N)")
    if advantage.shape not in {(old_cfm_loss.shape[0],), (old_cfm_loss.shape[0], 1)}:
        raise ValueError("advantage must have shape (B,) or (B,1)")
    if (
        clip_epsilon <= 0 or log_ratio_gain <= 0
        or trust_region not in {"ppo", "spo", "aspo"}
    ):
        raise ValueError("invalid FPO++ trust-region settings")
    advantage = advantage.reshape(-1, 1).float()
    old = old_cfm_loss.float()
    current = current_cfm_loss.float()
    if cfm_loss_clamp > 0:
        old = old.clamp(max=cfm_loss_clamp)
        current = current.clamp(max=cfm_loss_clamp)
    if negative_cfm_clamp > 0:
        current = torch.where(advantage < 0, current.clamp(max=negative_cfm_clamp), current)
    # CFM regression loss is not measured in log-density units.  Keep the
    # released estimator as gain=1, while allowing an empirically calibrated
    # conversion whose value is recorded in every experiment contract.
    log_ratio = clamp_ste(
        float(log_ratio_gain) * (old - current), maximum=log_ratio_clamp
    )
    ratio = torch.exp(log_ratio)
    ppo_unclipped = ratio * advantage
    ppo_clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    ppo_objective = torch.minimum(ppo_unclipped, ppo_clipped)
    spo_objective = (
        ratio * advantage
        - advantage.abs() / (2.0 * clip_epsilon) * (ratio - 1.0).square()
    )
    if trust_region == "ppo":
        objective = ppo_objective
    elif trust_region == "spo":
        objective = spo_objective
    else:
        objective = torch.where(advantage > 0, ppo_objective, spo_objective)
    clipped = (ratio - 1.0).abs() > clip_epsilon
    return FPOMetrics(
        loss=-objective.mean(),
        ratio_mean=ratio.mean().detach(),
        ratio_std=ratio.std(unbiased=False).detach(),
        approximate_kl=((ratio - 1.0) - log_ratio).mean().detach(),
        clip_fraction=clipped.float().mean().detach(),
        old_cfm_loss=old.mean().detach(),
        current_cfm_loss=current.mean().detach(),
    )


def variable_discount_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discounts: torch.Tensor,
    *,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE for action chunks with variable executed lengths and terminal masks."""

    if not (rewards.shape == values.shape == next_values.shape == discounts.shape):
        raise ValueError("GAE tensors must have identical one-dimensional shapes")
    if rewards.ndim != 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("GAE expects vectors and lambda in [0,1]")
    advantages = torch.zeros_like(rewards)
    running = rewards.new_zeros(())
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + discounts[index] * next_values[index] - values[index]
        running = delta + discounts[index] * gae_lambda * running
        advantages[index] = running
    return advantages, advantages + values


class FlowValueCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.input_dim:
            raise ValueError("critic input contract mismatch")
        return self.network(value).squeeze(-1)


def configure_flow_actor_scope(actor: nn.Module, scope: str) -> tuple[int, int]:
    """Apply the monotonic curriculum unfreeze schedule."""

    if scope not in {"flow_tail", "flow", "flow_context_tail", "all"}:
        raise ValueError(f"unknown flow actor scope {scope!r}")
    for parameter in actor.parameters():
        parameter.requires_grad_(scope == "all")
    if scope != "all":
        prefixes = ["final_norm.", "final_modulation.", "velocity."]
        if scope == "flow_tail":
            tail = max(0, len(actor.flow_blocks) - 2)
            prefixes.extend(f"flow_blocks.{index}." for index in range(tail, len(actor.flow_blocks)))
        else:
            prefixes.extend(["noisy_action_projection.", "condition_mlp.", "action_slot"])
            prefixes.extend(f"flow_blocks.{index}." for index in range(len(actor.flow_blocks)))
            if scope == "flow_context_tail":
                prefixes.extend([
                    f"context_blocks.{len(actor.context_blocks) - 1}.", "context_norm.",
                    "observation_projection.", "action_token_projection.",
                    "token_slot", "modality_embedding",
                ])
        for name, parameter in actor.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                parameter.requires_grad_(True)
    trainable = sum(parameter.numel() for parameter in actor.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in actor.parameters())
    if not trainable:
        raise RuntimeError("flow actor scope selected no parameters")
    return trainable, total
