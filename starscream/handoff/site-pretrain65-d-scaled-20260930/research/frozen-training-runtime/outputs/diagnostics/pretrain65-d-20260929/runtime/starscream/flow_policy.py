"""Causal frozen-token inputs for the direct shortcut-flow policy."""

from __future__ import annotations

import torch

from .DiT import repeat_first_history
from .wam import causal_action_timing


@torch.no_grad()
def tokenizer_policy_tokens(
    observation_tokenizer,
    action_tokenizer,
    batch: dict[str, torch.Tensor],
    *,
    history_raw_steps: int,
    valid_history_steps: torch.Tensor | int | None = None,
    amp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode three aligned macro-steps without any expert target leakage."""

    patch = int(observation_tokenizer.temporal_patch_size)
    if history_raw_steps % patch:
        raise ValueError("flow-policy history must form complete tokenizer patches")
    observation = {
        key: batch[key][:, :history_raw_steps]
        for key in ("mask", "proprio", "route", "timing")
    }
    device_type = batch["mask"].device.type
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=device_type == "cuda" and amp,
    ):
        observation_tokens = observation_tokenizer.encode(observation)
        action_tokens, _ = action_tokenizer.encode(
            batch["previous_action"][:, :history_raw_steps],
            causal_action_timing(batch)[:, :history_raw_steps],
        )
    if observation_tokens.shape[:2] != action_tokens.shape[:2]:
        raise RuntimeError("observation and applied-action token histories are misaligned")
    if valid_history_steps is not None:
        observation_tokens = repeat_first_history(
            observation_tokens, valid_history_steps
        )
        action_tokens = repeat_first_history(action_tokens, valid_history_steps)
    return observation_tokens.detach(), action_tokens.detach()
