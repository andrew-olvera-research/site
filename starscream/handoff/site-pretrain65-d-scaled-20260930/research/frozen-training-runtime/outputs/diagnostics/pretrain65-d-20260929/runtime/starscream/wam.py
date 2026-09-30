"""Causal world-backbone feature extraction for WAM-style policies."""

from __future__ import annotations

import math

import torch


def causal_action_timing(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Reconstruct action-tokenizer timing using online-observable signals."""

    timing = batch["timing"]
    if timing.shape[-1] < 10:
        raise ValueError("observation timing contract requires ten channels")
    # transition dt, age of the applied command, and applied-command validity.
    return torch.stack([timing[..., 9], timing[..., 3], timing[..., 7]], dim=-1)


def world_backbone_tokens_from_latents(
    action_tokenizer,
    dynamics,
    latents: torch.Tensor,
    applied_actions: torch.Tensor,
    observation_timing: torch.Tensor,
    *,
    signal_index: int | None = None,
    step_index: int = 0,
    detach: bool = True,
) -> torch.Tensor:
    """Run the native causal backbone on aligned latent/action histories.

    ``applied_actions`` and ``observation_timing`` may be raw-step tensors or
    macro tensors with an explicit patch dimension.  No current/future policy
    target is accepted by this interface.
    """

    if latents.ndim != 4:
        raise ValueError("latents must have shape (B,T,N,D)")
    if applied_actions.ndim == 4:
        applied_actions = applied_actions.flatten(1, 2)
    if observation_timing.ndim == 4:
        observation_timing = observation_timing.flatten(1, 2)
    if applied_actions.ndim != 3 or observation_timing.ndim != 3:
        raise ValueError("action/timing histories must be raw or macro-aligned sequences")
    action_timing = torch.stack(
        [observation_timing[..., 9], observation_timing[..., 3], observation_timing[..., 7]],
        dim=-1,
    )
    encoded_action = action_tokenizer(applied_actions, action_timing)
    if encoded_action.mean.shape[1] != latents.shape[1]:
        raise ValueError("latent and action histories contain different macro counts")
    macro_action = encoded_action.mean.mean(dim=2)
    packed = dynamics.pack(latents)
    batch_size, macros = packed.shape[:2]
    step = torch.full(
        (batch_size, macros), int(step_index), device=packed.device, dtype=torch.long
    )
    clean_signal = dynamics.k_max if signal_index is None else int(signal_index)
    signal = torch.full_like(step, clean_signal)
    output = dynamics(
        packed, macro_action, step, signal,
        action_context=encoded_action.context,
        return_backbone_tokens=True,
    )
    if output.backbone_tokens is None:
        raise RuntimeError("world model did not expose backbone tokens")
    return output.backbone_tokens.detach() if detach else output.backbone_tokens


@torch.no_grad()
def frozen_world_backbone_tokens(
    encoder,
    action_tokenizer,
    dynamics,
    batch: dict[str, torch.Tensor],
    *,
    history: int,
    device: str,
    amp: bool,
    signal_index: int | None = None,
    step_index: int = 0,
) -> torch.Tensor:
    """Return clean, causal dynamics-transformer tokens without target actions."""

    patch = int(encoder.temporal_patch_size)
    if history % patch:
        raise ValueError("WAM history must form complete tokenizer patches")
    observation = {
        key: batch[key][:, :history] for key in ("mask", "proprio", "route", "timing")
    }
    with torch.autocast(
        "cuda", dtype=torch.bfloat16,
        enabled=device.startswith("cuda") and amp,
    ):
        latents = encoder.encode(observation)
        tokens = world_backbone_tokens_from_latents(
            action_tokenizer, dynamics, latents,
            batch["previous_action"][:, :history], batch["timing"][:, :history],
            signal_index=signal_index, step_index=step_index, detach=True,
        )
    if tokens.shape[1] != history // patch:
        raise RuntimeError("world token history is not aligned to macro observations")
    return tokens
