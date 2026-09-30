"""SkyDreamer-style state rewards and flow-world-model rollout helpers."""

from __future__ import annotations

import torch


TASK_STATE_POSITION_SCALE = 20.0
TASK_STATE_RATE_SCALE = 6.0


def decode_structured_task_state(
    previous: torch.Tensor, prediction: torch.Tensor
) -> torch.Tensor:
    """Decode the RaceDynamics structured-delta task-state contract."""

    if previous.shape != prediction.shape or previous.shape[-1] != 19:
        raise ValueError("task states must have matching (..., 19) shapes")
    following = prediction.clone()
    following[..., 0:6] = previous[..., 0:6] + prediction[..., 0:6]
    following[..., 12:19] = previous[..., 12:19] + prediction[..., 12:19]
    return following


def state_estimate_reward(
    previous: torch.Tensor,
    following: torch.Tensor,
    *,
    progress_weight: float = 5.0,
    gate_weight: float = 30.0,
    gate_clearance_m: float = 0.8,
    control_frequency_hz: float = 90.0,
    rate_clip: float = 17.0,
    rate_denominator: float = 1.0e5,
    raw_steps: int = 3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the published reward from predicted normalized task states.

    A macro transition spans ``raw_steps`` controller ticks. Progress and gate
    crossing are computed once over the macro; the per-tick angular-rate cost
    is integrated over those ticks.
    """

    if previous.shape != following.shape or previous.shape[-1] != 19:
        raise ValueError("task states must have matching (..., 19) shapes")
    if gate_clearance_m <= 0 or control_frequency_hz <= 0 or raw_steps <= 0:
        raise ValueError("reward scales and raw_steps must be positive")
    previous_position = previous[..., 0:3].float() * TASK_STATE_POSITION_SCALE
    following_position = following[..., 0:3].float() * TASK_STATE_POSITION_SCALE
    progress_delta = previous_position.norm(dim=-1) - following_position.norm(dim=-1)

    previous_x = previous_position[..., 0]
    following_x = following_position[..., 0]
    crossed = (previous_x < 0.0) & (following_x >= 0.0)
    alpha = (-previous_x / (following_x - previous_x).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    crossing_position = previous_position + alpha.unsqueeze(-1) * (
        following_position - previous_position
    )
    gate_quality = (
        1.0
        - crossing_position[..., 1:3].abs().amax(dim=-1) / float(gate_clearance_m)
    ).clamp(0.0, 1.0) * crossed.float()

    body_rate_l1 = (following[..., 12:15].float() * TASK_STATE_RATE_SCALE).abs().sum(-1)
    rate_penalty = (
        torch.exp(body_rate_l1.clamp(max=float(rate_clip))) - 1.0
    ) / (2.0 * float(control_frequency_hz) * float(rate_denominator))
    rate_penalty = rate_penalty * float(raw_steps)
    progress = float(progress_weight) * progress_delta
    gate = float(gate_weight) * gate_quality
    reward = progress - rate_penalty + gate
    return reward, {
        "progress_delta": progress_delta,
        "progress_reward": progress,
        "body_rate_l1": body_rate_l1,
        "rate_penalty": rate_penalty,
        "gate_crossing": crossed.float(),
        "gate_quality": gate_quality,
        "gate_reward": gate,
    }


def replay_skydreamer_reward(
    batch: dict[str, torch.Tensor],
    *,
    progress_weight: float = 5.0,
    gate_weight: float = 30.0,
    control_frequency_hz: float = 90.0,
    rate_clip: float = 17.0,
    rate_denominator: float = 1.0e5,
) -> torch.Tensor:
    """Reconstruct raw-step SkyDreamer rewards for old and new replay."""

    required = ("progress_delta", "gate_quality", "task_state")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"replay is missing SkyDreamer fields: {missing}")
    rate_l1 = (batch["task_state"][:, 1:, 12:15].float() * TASK_STATE_RATE_SCALE).abs().sum(-1)
    rate_penalty = (
        torch.exp(rate_l1.clamp(max=float(rate_clip))) - 1.0
    ) / (2.0 * float(control_frequency_hz) * float(rate_denominator))
    return (
        float(progress_weight) * batch["progress_delta"].float()
        - rate_penalty
        + float(gate_weight) * batch["gate_quality"].float()
    )
