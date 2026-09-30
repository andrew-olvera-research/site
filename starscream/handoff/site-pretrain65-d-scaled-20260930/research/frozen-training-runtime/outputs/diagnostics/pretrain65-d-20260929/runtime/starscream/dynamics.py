"""Action-conditioned Dreamer-4 latent dynamics and control heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from dreamerv4 import Dynamics, pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck


@dataclass(frozen=True)
class DynamicsOutput:
    predicted_packed_latents: torch.Tensor
    agent_state: torch.Tensor
    reward_logits: torch.Tensor
    reward_value: torch.Tensor | None
    continue_logits: torch.Tensor
    task_state_mean: torch.Tensor
    task_state_log_scale: torch.Tensor
    privileged_state: torch.Tensor | None = None
    estimator_state: torch.Tensor | None = None
    estimator_std: torch.Tensor | None = None
    backbone_tokens: torch.Tensor | None = None


class RaceDynamics(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 384,
        d_bottleneck: int = 64,
        n_latents: int = 16,
        n_spatial: int = 4,
        n_heads: int = 6,
        depth: int = 8,
        n_register: int = 4,
        k_max: int = 8,
        reward_bins: int = 255,
        action_context_dim: int = 256,
        privileged_state_dim: int = 0,
        estimator_state_dim: int = 0,
        estimator_std_dim: int = 0,
        task_state_uncertainty: bool = True,
        separate_control_tokens: bool = False,
        control_backbone_gradient_scale: float = 1.0,
        causal_residual: bool = False,
        direct_reward_head: bool = False,
        pmpo_auxiliary_heads: bool = False,
        pmpo_head_hidden: int = 512,
        pmpo_head_dropout: float = 0.0,
        pmpo_head_backbone_gradient_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if n_latents % n_spatial:
            raise ValueError("n_latents must be divisible by n_spatial")
        self.n_latents = int(n_latents)
        self.n_spatial = int(n_spatial)
        self.packing_factor = n_latents // n_spatial
        self.d_bottleneck = int(d_bottleneck)
        self.d_spatial = d_bottleneck * self.packing_factor
        self.k_max = int(k_max)
        self.task_state_uncertainty = bool(task_state_uncertainty)
        self.separate_control_tokens = bool(separate_control_tokens)
        self.control_backbone_gradient_scale = float(control_backbone_gradient_scale)
        self.causal_residual = bool(causal_residual)
        self.direct_reward_head = bool(direct_reward_head)
        self.pmpo_auxiliary_heads = bool(pmpo_auxiliary_heads)
        self.pmpo_head_dropout = float(pmpo_head_dropout)
        self.pmpo_head_backbone_gradient_scale = float(
            pmpo_head_backbone_gradient_scale
        )
        if not 0.0 <= self.control_backbone_gradient_scale <= 1.0:
            raise ValueError("control_backbone_gradient_scale must be in [0,1]")
        if not 0.0 <= self.pmpo_head_backbone_gradient_scale <= 1.0:
            raise ValueError("pmpo_head_backbone_gradient_scale must be in [0,1]")
        if not 0.0 <= self.pmpo_head_dropout < 1.0:
            raise ValueError("pmpo_head_dropout must be in [0,1)")
        self.action_context_projection = nn.Linear(action_context_dim, d_model)
        self.core = Dynamics(
            d_model=d_model,
            d_bottleneck=d_bottleneck,
            d_spatial=self.d_spatial,
            n_spatial=n_spatial,
            n_register=n_register,
            n_agent=3 if self.separate_control_tokens else 1,
            n_heads=n_heads,
            depth=depth,
            k_max=k_max,
            space_mode="wm_agent",
        )
        self.head_norm = nn.RMSNorm(d_model)
        self.reward_head = nn.Linear(d_model, reward_bins)
        self.reward_value_head = nn.Linear(d_model, 1) if self.direct_reward_head else None
        self.continue_head = nn.Linear(d_model, 1)
        self.pmpo_reward_head = None
        self.pmpo_reward_value_head = None
        self.pmpo_continue_head = None
        if self.pmpo_auxiliary_heads:
            # PMPO consumes imagined rewards and continuations from recursive
            # transitions. Make those heads explicitly action- and
            # outcome-conditioned instead of asking a pre-transition agent
            # token to infer the residual head's result implicitly.
            feature_dim = d_model + 2 * self.d_spatial + 16

            def outcome_head(output_dim: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.RMSNorm(feature_dim),
                    nn.Linear(feature_dim, int(pmpo_head_hidden)),
                    nn.Sequential(
                        nn.SiLU(), nn.Dropout(self.pmpo_head_dropout)
                    ),
                    nn.Linear(int(pmpo_head_hidden), output_dim),
                )

            self.pmpo_reward_head = outcome_head(reward_bins)
            self.pmpo_continue_head = outcome_head(1)
            if self.direct_reward_head:
                self.pmpo_reward_value_head = outcome_head(1)
            # Retain the legacy modules in the state dict so v3 checkpoints can
            # initialize v3.1 cleanly, but do not optimize unused parameters.
            self.reward_head.requires_grad_(False)
            self.continue_head.requires_grad_(False)
            if self.reward_value_head is not None:
                self.reward_value_head.requires_grad_(False)
        self.state_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, 38 if self.task_state_uncertainty else 19),
        )
        self.privileged_state_head = (
            nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, privileged_state_dim))
            if privileged_state_dim else None
        )
        self.estimator_state_head = (
            nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, estimator_state_dim))
            if estimator_state_dim else None
        )
        self.estimator_std_head = (
            nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, estimator_std_dim))
            if estimator_std_dim else None
        )
        self.causal_residual_head = None
        if self.causal_residual:
            self.causal_residual_head = nn.Sequential(
                nn.RMSNorm(2 * self.d_spatial + d_model),
                nn.Linear(2 * self.d_spatial + d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, self.d_spatial),
            )
            # Start at the persistence baseline. The relative transition loss
            # then learns only the action-conditioned change that beats it.
            nn.init.zeros_(self.causal_residual_head[-1].weight)
            nn.init.zeros_(self.causal_residual_head[-1].bias)

    def pack(self, latents: torch.Tensor) -> torch.Tensor:
        return pack_bottleneck_to_spatial(
            latents, n_spatial=self.n_spatial, k=self.packing_factor
        )

    def unpack(self, packed: torch.Tensor) -> torch.Tensor:
        return unpack_spatial_to_bottleneck(packed, k=self.packing_factor)

    def forward(
        self,
        noisy_packed_latents: torch.Tensor,
        actions: torch.Tensor,
        step_indices: torch.Tensor,
        signal_indices: torch.Tensor,
        initial_context: torch.Tensor | None = None,
        action_context: torch.Tensor | None = None,
        causal_transition: bool = False,
        return_backbone_tokens: bool = False,
    ) -> DynamicsOutput:
        if actions.shape[-1] == 4:
            padded = actions.new_zeros(*actions.shape[:-1], 16)
            padded[..., :4] = actions
            actions = padded
        transition_actions = actions
        has_context = initial_context is not None
        if has_context:
            if initial_context.shape[:1] != noisy_packed_latents.shape[:1] or initial_context.shape[1] != 1:
                raise ValueError("initial_context must have shape (B,1,S,D)")
            noisy_packed_latents = torch.cat([initial_context, noisy_packed_latents], dim=1)
            actions = torch.cat([torch.zeros_like(actions[:, :1]), actions], dim=1)
            step_indices = torch.cat([torch.zeros_like(step_indices[:, :1]), step_indices], dim=1)
            signal_indices = torch.cat([torch.zeros_like(signal_indices[:, :1]), signal_indices], dim=1)
            if action_context is not None:
                action_context = torch.cat([torch.zeros_like(action_context[:, :1]), action_context], dim=1)
        action_embeddings = (
            self.action_context_projection(action_context) if action_context is not None else None
        )
        core_output = self.core(
            actions, step_indices, signal_indices, noisy_packed_latents,
            action_embeddings=action_embeddings,
            return_hidden=return_backbone_tokens,
        )
        if return_backbone_tokens:
            predicted, agent, backbone_tokens = core_output
        else:
            predicted, agent = core_output
            backbone_tokens = None
        if agent is None:
            raise RuntimeError("RaceDynamics requires an agent token")
        if has_context:
            predicted = predicted[:, 1:]
            agent = agent[:, 1:]
            if backbone_tokens is not None:
                backbone_tokens = backbone_tokens[:, 1:]
        states = self.head_norm(agent)
        state = states[:, :, 0]
        reward_state = states[:, :, 1] if self.separate_control_tokens else state
        continue_state = states[:, :, 2] if self.separate_control_tokens else state

        def scale_backbone_gradient(value: torch.Tensor) -> torch.Tensor:
            scale = self.control_backbone_gradient_scale
            return value.detach() + scale * (value - value.detach())

        reward_state = scale_backbone_gradient(reward_state)
        continue_state = scale_backbone_gradient(continue_state)
        if causal_transition:
            if not self.causal_residual or self.causal_residual_head is None:
                raise ValueError("causal_transition requires causal_residual=True")
            if initial_context is None or initial_context.shape[1] != predicted.shape[1]:
                raise ValueError("causal residual transitions require one context per prediction")
            spatial_state = state.unsqueeze(2).expand(-1, -1, self.n_spatial, -1)
            residual_features = torch.cat(
                [predicted, initial_context, spatial_state], dim=-1
            )
            predicted = initial_context + self.causal_residual_head(residual_features)
        if self.pmpo_auxiliary_heads:
            context = (
                initial_context
                if initial_context is not None
                else torch.zeros_like(predicted[:, :1])
            )
            next_summary = predicted.mean(dim=2)
            delta_summary = (predicted - context).mean(dim=2)

            def pmpo_features(control_state: torch.Tensor) -> torch.Tensor:
                features = torch.cat(
                    [control_state, next_summary, delta_summary, transition_actions],
                    dim=-1,
                )
                scale = self.pmpo_head_backbone_gradient_scale
                return features.detach() + scale * (features - features.detach())

            reward_features = pmpo_features(reward_state)
            continue_features = pmpo_features(continue_state)
            reward_logits = self.pmpo_reward_head(reward_features)
            continue_logits = self.pmpo_continue_head(continue_features).squeeze(-1)
            reward_value = (
                self.pmpo_reward_value_head(reward_features).squeeze(-1)
                if self.pmpo_reward_value_head is not None else None
            )
        else:
            reward_logits = self.reward_head(reward_state)
            continue_logits = self.continue_head(continue_state).squeeze(-1)
            reward_value = (
                self.reward_value_head(reward_state).squeeze(-1)
                if self.reward_value_head is not None else None
            )
        task = self.state_head(state)
        if self.task_state_uncertainty:
            task_mean, task_log_scale = task.chunk(2, -1)
            task_log_scale = task_log_scale.clamp(-3.0, 2.0)
        else:
            task_mean = task
            task_log_scale = torch.zeros_like(task_mean)
        return DynamicsOutput(
            predicted_packed_latents=predicted,
            agent_state=state,
            reward_logits=reward_logits,
            reward_value=reward_value,
            continue_logits=continue_logits,
            task_state_mean=task_mean,
            # A physical-state uncertainty below exp(-3) in normalized units is
            # not supported by this dataset and caused catastrophic held-out NLL.
            task_state_log_scale=task_log_scale,
            privileged_state=(
                self.privileged_state_head(state)
                if self.privileged_state_head is not None else None
            ),
            estimator_state=(
                self.estimator_state_head(state)
                if self.estimator_state_head is not None else None
            ),
            estimator_std=(
                self.estimator_std_head(state)
                if self.estimator_std_head is not None else None
            ),
            backbone_tokens=backbone_tokens,
        )
