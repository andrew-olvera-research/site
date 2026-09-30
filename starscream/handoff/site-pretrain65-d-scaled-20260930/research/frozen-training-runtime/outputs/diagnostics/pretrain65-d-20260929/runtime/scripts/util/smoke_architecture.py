#!/usr/bin/env python3
"""One-batch CUDA smoke for tokenizer, dynamics, Gaussian actor, and critic math."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from starscream.actor_critic import DistributionalCritic, GaussianActor
from starscream.dataloader import DreamerSequenceDataset
from starscream.dynamics import RaceDynamics
from starscream.loss import (
    action_tokenizer_loss,
    actor_bc_gnll_loss,
    critic_twohot_loss,
    observation_jepa_loss,
    shortcut_forcing_objective,
)
from starscream.tokenizer import ActionJEPA, ActionTokenizer, MultiModalTokenizer, ObservationJEPA


def finite_gradients(module: torch.nn.Module) -> bool:
    return all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/data/starscream-champion-debug-v4")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = args.device
    observation_data = DreamerSequenceDataset(args.data, sequence_length=24, stride=3, mode="tokenizer")
    action_data = DreamerSequenceDataset(args.data, sequence_length=48, stride=3, mode="action_tokenizer")
    observation_batch = next(iter(DataLoader(observation_data, batch_size=2)))
    action_batch = next(iter(DataLoader(action_data, batch_size=4)))
    observation_batch = {key: value.to(device) for key, value in observation_batch.items()}
    action_batch = {key: value.to(device) for key, value in action_batch.items()}

    observation = ObservationJEPA(MultiModalTokenizer(), predictor_depth=2).to(device)
    action = ActionJEPA(ActionTokenizer(), predictor_depth=2).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        observation_output = observation(observation_batch)
        observation_loss, observation_metrics = observation_jepa_loss(observation_output, observation_batch)
        action_output = action(action_batch)
        action_loss, action_metrics = action_tokenizer_loss(action_output, action_batch)
    observation_loss.backward()
    action_loss.backward()
    assert finite_gradients(observation) and finite_gradients(action)

    dynamics = RaceDynamics(
        d_model=128, d_bottleneck=128, n_latents=4, n_spatial=2,
        n_heads=4, depth=2, n_register=1,
    ).to(device)
    encoded = observation_output.reconstruction.latents.detach()
    packed = dynamics.pack(encoded)
    clean, initial = packed[:, 1:], packed[:, :1]
    macro_actions = action_output.reconstruction.mean.detach()[:, : clean.shape[1]].mean(dim=2)
    macro_actions = macro_actions[: clean.shape[0]]
    if macro_actions.shape[0] < clean.shape[0]:
        macro_actions = macro_actions.expand(clean.shape[0], -1, -1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        dynamics_output, dynamics_objective = shortcut_forcing_objective(
            dynamics, clean, macro_actions, initial_context=initial
        )
    dynamics_objective.backward()
    assert dynamics_output.predicted_packed_latents.shape == clean.shape
    assert finite_gradients(dynamics)

    actor = GaussianActor(input_dim=128, hidden_dim=128, action_horizon=3).to(device)
    critic = DistributionalCritic(input_dim=128, hidden_dim=128).to(device)
    state = dynamics_output.agent_state.detach()
    policy = actor(state)
    target_action = torch.zeros_like(policy.mean)
    actor_loss, _ = actor_bc_gnll_loss(policy, target_action)
    critic_loss, _ = critic_twohot_loss(critic, state, torch.zeros(state.shape[:-1], device=device))
    (actor_loss + critic_loss).backward()
    assert finite_gradients(actor) and finite_gradients(critic)
    print(
        {
            "observation_loss": float(observation_loss.detach()),
            "action_loss": float(action_loss.detach()),
            "dynamics_loss": float(dynamics_objective.detach()),
            "observation_metrics": sorted(observation_metrics),
            "action_metrics": sorted(action_metrics),
            "observation_trainable": sum(p.numel() for p in observation.parameters() if p.requires_grad),
            "action_trainable": sum(p.numel() for p in action.parameters() if p.requires_grad),
        }
    )


if __name__ == "__main__":
    main()
