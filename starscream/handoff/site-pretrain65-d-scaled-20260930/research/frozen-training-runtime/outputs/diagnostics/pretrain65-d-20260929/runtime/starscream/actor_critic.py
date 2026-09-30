"""Distributional Dreamer critic and target-network mechanics."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn

from dreamerv4 import TwoHot


def actor_state_dimension(encoder) -> int:
    """Feature width of the shared BC/closed-loop/PMPO actor contract."""

    latent_dim = encoder.n_latents * encoder.d_bottleneck
    state_dim = int(
        encoder.state_head.out_features
        // (2 if encoder.predict_state_uncertainty else 1)
    )
    patch = encoder.temporal_patch_size
    return (
        2 * latent_dim
        + 2 * encoder.proprio_dim
        + patch * 4
        + patch * encoder.timing_dim
        + 3 * state_dim
    )


def _temporal_difference(value: torch.Tensor) -> torch.Tensor:
    difference = torch.zeros_like(value)
    difference[:, 1:] = value[:, 1:] - value[:, :-1]
    return difference


def build_actor_state(
    encoder,
    batch: dict[str, torch.Tensor],
    history: int,
    device: str,
    amp: bool,
    *,
    encoder_grad: bool = False,
) -> torch.Tensor:
    """Build the exact causal actor feature sequence used in BC and flight."""

    observation_history = {
        "mask": batch["mask"][:, :history],
        "proprio": batch["proprio"][:, :history],
        "route": batch["route"][:, :history],
        "timing": batch["timing"][:, :history],
    }
    if "estimate" in batch:
        observation_history["estimate"] = batch["estimate"][:, :history]
    gradient_context = nullcontext() if encoder_grad else torch.no_grad()
    with gradient_context, torch.autocast(
        "cuda", dtype=torch.bfloat16,
        enabled=device.startswith("cuda") and amp,
    ):
        # Every feature is available before selecting action a_t. The dynamics
        # agent_state is action-conditioned and would leak the target.
        latents = encoder.encode(observation_history)
        patch = encoder.temporal_patch_size
        batch_size, raw_steps = observation_history["proprio"].shape[:2]
        if raw_steps % patch:
            raise ValueError("actor history must form complete tokenizer patches")
        macros = raw_steps // patch
        applied_action_history = batch["previous_action"][:, :history].reshape(
            batch_size, macros, patch * 4
        )
        raw_timing = observation_history["timing"].reshape(
            batch_size, macros, patch * encoder.timing_dim
        )
        features = build_actor_state_from_latents(
            encoder, latents, applied_action_history, raw_timing
        )
    expected = actor_state_dimension(encoder)
    if features.shape[-1] != expected:
        raise RuntimeError(
            f"actor state contract produced {features.shape[-1]} features, expected {expected}"
        )
    return features


def build_actor_state_from_latents(
    encoder,
    latents: torch.Tensor,
    applied_action_history: torch.Tensor,
    timing_history: torch.Tensor,
) -> torch.Tensor:
    """Build the shared actor state from posterior or recursively predicted latents.

    ``applied_action_history`` and ``timing_history`` contain one flattened raw
    patch per latent macro-step. Keeping this operation separate from image
    encoding makes the BC, closed-loop, and imagination contracts identical.
    """

    if latents.ndim != 4:
        raise ValueError("latents must have shape (B,T,N,D)")
    if applied_action_history.ndim == 4:
        applied_action_history = applied_action_history.flatten(2)
    if timing_history.ndim == 4:
        timing_history = timing_history.flatten(2)
    if applied_action_history.shape[:2] != latents.shape[:2]:
        raise ValueError("applied action history must align with latent macro-steps")
    if timing_history.shape[:2] != latents.shape[:2]:
        raise ValueError("timing history must align with latent macro-steps")

    state_mean, state_log_scale = encoder.estimate_state(latents)
    decoded_proprio = encoder.estimate_proprio(latents)
    latent_features = latents.flatten(2)
    features = torch.cat(
        [
            latent_features,
            _temporal_difference(latent_features),
            decoded_proprio,
            _temporal_difference(decoded_proprio),
            applied_action_history,
            timing_history,
            state_mean,
            _temporal_difference(state_mean),
            state_log_scale,
        ],
        dim=-1,
    )
    expected = actor_state_dimension(encoder)
    if features.shape[-1] != expected:
        raise RuntimeError(
            f"imagined actor state produced {features.shape[-1]} features, expected {expected}"
        )
    return features


@dataclass(frozen=True)
class GaussianPolicyOutput:
    """Diagonal Gaussian over a normalized CTBR action chunk."""

    mean: torch.Tensor
    log_std: torch.Tensor
    action_tokens: torch.Tensor | None = None

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp()

    def rsample(self) -> torch.Tensor:
        return (self.mean + self.std * torch.randn_like(self.mean)).clamp(-1.0, 1.0)

    def mode(self) -> torch.Tensor:
        return self.mean

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        log_two_pi = torch.log(action.new_tensor(2.0 * torch.pi))
        elementwise = -0.5 * (action - self.mean).square() * torch.exp(-2.0 * self.log_std)
        elementwise = elementwise - self.log_std - 0.5 * log_two_pi
        return elementwise.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        constant = 0.5 * (1.0 + torch.log(self.mean.new_tensor(2.0 * torch.pi)))
        return (self.log_std + constant).sum(dim=-1)


class GaussianActor(nn.Module):
    """Gaussian action-chunk policy with a configurable causal decoder."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 512,
        action_dim: int = 4,
        action_horizon: int = 8,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
        temporal_depth: int = 0,
        temporal_heads: int = 8,
        temporal_context: int = 8,
        temporal_dropout: float = 0.0,
        separate_std_head: bool = False,
        std_hidden_dim: int = 256,
        std_backbone_gradient_scale: float = 1.0,
        initial_log_std: float | None = None,
        decoder_type: str = "monolithic",
        action_query_heads: int | None = None,
        action_query_depth: int = 1,
        action_query_dropout: float = 0.0,
        action_token_count: int = 0,
        action_token_dim: int = 0,
        head_hidden_dim: int | None = None,
        head_depth: int = 2,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.temporal_depth = int(temporal_depth)
        self.temporal_context = int(temporal_context)
        self.separate_std_head = bool(separate_std_head)
        self.std_backbone_gradient_scale = float(std_backbone_gradient_scale)
        self.decoder_type = str(decoder_type)
        self.action_token_count = int(action_token_count)
        self.action_token_dim = int(action_token_dim)
        self.uses_action_tokenizer = self.action_token_count > 0
        self.head_hidden_dim = int(head_hidden_dim or hidden_dim)
        self.head_depth = int(head_depth)
        if self.decoder_type not in {"monolithic", "action_query"}:
            raise ValueError("decoder_type must be monolithic or action_query")
        if not 0.0 <= self.std_backbone_gradient_scale <= 1.0:
            raise ValueError("std_backbone_gradient_scale must be in [0,1]")
        if self.uses_action_tokenizer and (
            self.action_token_dim < 1
            or not self.separate_std_head
            or self.decoder_type != "monolithic"
        ):
            raise ValueError(
                "action-token decoding requires positive token dimensions, "
                "separate_std_head=true, and decoder_type=monolithic"
            )
        if self.head_depth < 1:
            raise ValueError("head_depth must be positive")
        if self.temporal_depth > 0:
            if hidden_dim % int(temporal_heads):
                raise ValueError("hidden_dim must be divisible by temporal_heads")
            self.input_projection = nn.Sequential(
                nn.RMSNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.SiLU()
            )
            self.temporal_position = nn.Parameter(
                torch.randn(self.temporal_context, hidden_dim) * 0.02
            )
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=int(temporal_heads),
                dim_feedforward=4 * hidden_dim,
                dropout=float(temporal_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(
                layer, num_layers=self.temporal_depth, norm=nn.RMSNorm(hidden_dim)
            )
            network_input_dim = hidden_dim
        else:
            self.input_projection = None
            self.temporal_position = None
            self.temporal = None
            network_input_dim = input_dim
        output_dim = action_horizon * action_dim
        self.action_queries = None
        self.action_query_decoder = None
        if self.decoder_type == "action_query":
            if self.temporal is None:
                raise ValueError("action_query decoder requires temporal_depth > 0")
            query_heads = int(action_query_heads or temporal_heads)
            if hidden_dim % query_heads:
                raise ValueError("hidden_dim must be divisible by action_query_heads")
            if int(action_query_depth) < 1:
                raise ValueError("action_query_depth must be positive")
            self.action_queries = nn.Parameter(
                torch.randn(self.action_horizon, hidden_dim) * 0.02
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=query_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=float(action_query_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.action_query_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=int(action_query_depth),
                norm=nn.RMSNorm(hidden_dim),
            )
            network_output_dim = action_dim if self.separate_std_head else 2 * action_dim
        else:
            network_output_dim = (
                self.action_token_count * self.action_token_dim
                if self.uses_action_tokenizer
                else output_dim if self.separate_std_head else 2 * output_dim
            )
        mean_layers: list[nn.Module] = [
            nn.RMSNorm(network_input_dim),
            nn.Linear(network_input_dim, self.head_hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(self.head_depth - 1):
            mean_layers.extend(
                [nn.Linear(self.head_hidden_dim, self.head_hidden_dim), nn.SiLU()]
            )
        mean_layers.append(nn.Linear(self.head_hidden_dim, network_output_dim))
        self.network = nn.Sequential(*mean_layers)
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.std_network = None
        if self.separate_std_head:
            self.std_network = nn.Sequential(
                nn.RMSNorm(network_input_dim),
                nn.Linear(network_input_dim, int(std_hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(std_hidden_dim), int(std_hidden_dim)),
                nn.SiLU(),
                nn.Linear(
                    int(std_hidden_dim),
                    action_dim if self.decoder_type == "action_query" else output_dim,
                ),
            )
            nn.init.zeros_(self.std_network[-1].weight)
            requested = (
                0.5 * (self.min_log_std + self.max_log_std)
                if initial_log_std is None else float(initial_log_std)
            )
            if not self.min_log_std < requested < self.max_log_std:
                raise ValueError("initial_log_std must lie strictly between its bounds")
            unit = 2.0 * (requested - self.min_log_std) / (
                self.max_log_std - self.min_log_std
            ) - 1.0
            raw_initial = torch.atanh(torch.tensor(unit)).item()
            nn.init.constant_(self.std_network[-1].bias, raw_initial)

    def forward(self, state: torch.Tensor, action_tokenizer=None) -> GaussianPolicyOutput:
        if state.ndim not in {2, 3}:
            raise ValueError("actor state must have shape (B,D) or (B,T,D)")
        if self.temporal is not None:
            if state.ndim == 2:
                state = state[:, None]
            if state.shape[1] > self.temporal_context:
                raise ValueError("actor state sequence exceeds temporal_context")
            state = self.input_projection(state)
            state = state + self.temporal_position[-state.shape[1] :].unsqueeze(0)
            length = state.shape[1]
            causal_mask = torch.triu(
                torch.ones(length, length, device=state.device, dtype=torch.bool),
                diagonal=1,
            )
            memory = self.temporal(state, mask=causal_mask)
            if self.decoder_type == "action_query":
                assert self.action_queries is not None
                assert self.action_query_decoder is not None
                queries = self.action_queries.unsqueeze(0).expand(state.shape[0], -1, -1)
                state = self.action_query_decoder(queries, memory)
            else:
                state = memory[:, -1]
        elif state.ndim == 3:
            state = state[:, -1]
        batch = state.shape[0]
        action_tokens = None
        if self.std_network is None:
            decoded = self.network(state)
            if self.decoder_type == "action_query":
                mean, raw_log_std = decoded.view(
                    batch, self.action_horizon, self.action_dim, 2
                ).unbind(dim=-1)
            else:
                mean, raw_log_std = decoded.view(
                    batch, self.action_horizon, self.action_dim, 2
                ).unbind(dim=-1)
        else:
            if self.uses_action_tokenizer:
                if action_tokenizer is None:
                    raise ValueError(
                        "this actor requires its frozen action tokenizer decoder"
                    )
                if (
                    action_tokenizer.n_tokens != self.action_token_count
                    or action_tokenizer.d_latent != self.action_token_dim
                    or action_tokenizer.temporal_patch_size != self.action_horizon
                ):
                    raise ValueError("actor and action tokenizer contracts do not match")
                action_tokens = self.network(state).tanh().view(
                    batch, 1, self.action_token_count, self.action_token_dim
                )
                mean, _ = action_tokenizer.decode(action_tokens)
                mean = mean[:, 0]
            else:
                mean = self.network(state).view(
                    batch, self.action_horizon, self.action_dim
                )
            scale = self.std_backbone_gradient_scale
            std_state = state.detach() + scale * (state - state.detach())
            raw_log_std = self.std_network(std_state).view(
                batch, self.action_horizon, self.action_dim
            )
        # Smooth bounded parameterization keeps gradients alive at both limits.
        unit = raw_log_std.tanh()
        log_std = self.min_log_std + 0.5 * (unit + 1.0) * (
            self.max_log_std - self.min_log_std
        )
        return GaussianPolicyOutput(
            mean if self.uses_action_tokenizer else mean.tanh(),
            log_std,
            action_tokens,
        )


class WAMGaussianActor(nn.Module):
    """Small action head over frozen causal world-backbone token sequences."""

    def __init__(
        self,
        *,
        backbone_dim: int = 512,
        hidden_dim: int = 512,
        action_dim: int = 4,
        action_horizon: int = 3,
        cross_attention_heads: int = 8,
        cross_attention_dropout: float = 0.0,
        head_depth: int = 2,
        min_log_std: float = -2.3,
        max_log_std: float = 0.5,
        initial_log_std: float = -1.5,
        separate_std_head: bool = True,
    ) -> None:
        super().__init__()
        if backbone_dim % int(cross_attention_heads):
            raise ValueError("backbone_dim must be divisible by cross_attention_heads")
        if head_depth < 1:
            raise ValueError("head_depth must be positive")
        self.backbone_dim = int(backbone_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.separate_std_head = bool(separate_std_head)
        self.action_queries = nn.Parameter(
            torch.randn(self.action_horizon, self.backbone_dim) * 0.02
        )
        self.query_norm = nn.RMSNorm(self.backbone_dim)
        self.token_norm = nn.RMSNorm(self.backbone_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.backbone_dim,
            int(cross_attention_heads),
            dropout=float(cross_attention_dropout),
            batch_first=True,
        )

        def mlp(output_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.RMSNorm(self.backbone_dim),
                nn.Linear(self.backbone_dim, hidden_dim),
                nn.SiLU(),
            ]
            for _ in range(head_depth - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
            layers.append(nn.Linear(hidden_dim, output_dim))
            return nn.Sequential(*layers)

        self.mean_head = mlp(self.action_dim)
        self.std_head = mlp(self.action_dim) if self.separate_std_head else None
        self.log_std = (
            None if self.separate_std_head else nn.Parameter(
                torch.full((self.action_horizon, self.action_dim), initial_log_std)
            )
        )
        if self.std_head is not None:
            nn.init.zeros_(self.std_head[-1].weight)
            initial_unit = 2.0 * (
                (initial_log_std - self.min_log_std)
                / (self.max_log_std - self.min_log_std)
            ) - 1.0
            initial_unit = min(max(initial_unit, -0.999), 0.999)
            nn.init.constant_(self.std_head[-1].bias, float(torch.atanh(torch.tensor(initial_unit))))

    def features(self, backbone_tokens: torch.Tensor) -> torch.Tensor:
        """Fuse a causal world-token bank into one feature per action offset."""
        if backbone_tokens.ndim != 4:
            raise ValueError("backbone tokens must have shape (B,T,S,D)")
        batch, _, _, width = backbone_tokens.shape
        if width != self.backbone_dim:
            raise ValueError(
                f"world token width {width} does not match actor {self.backbone_dim}"
            )
        context = self.token_norm(backbone_tokens.flatten(1, 2))
        queries = self.query_norm(self.action_queries).unsqueeze(0).expand(batch, -1, -1)
        fused, _ = self.cross_attention(
            queries, context, context, need_weights=False
        )
        return fused + queries

    def forward(self, backbone_tokens: torch.Tensor) -> GaussianPolicyOutput:
        fused = self.features(backbone_tokens)
        batch = fused.shape[0]
        mean = self.mean_head(fused).tanh()
        if self.std_head is not None:
            raw_log_std = self.std_head(fused)
        else:
            raw_log_std = self.log_std.unsqueeze(0).expand(batch, -1, -1)
        unit = raw_log_std.tanh()
        log_std = self.min_log_std + 0.5 * (unit + 1.0) * (
            self.max_log_std - self.min_log_std
        )
        return GaussianPolicyOutput(mean, log_std)


class DistributionalCritic(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 512,
        bins: int = 255,
        temporal_depth: int = 0,
        temporal_heads: int = 8,
        temporal_context: int = 8,
        temporal_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.temporal_depth = int(temporal_depth)
        self.temporal_context = int(temporal_context)
        self.input_projection = None
        self.temporal_position = None
        self.temporal = None
        network_input = input_dim
        if self.temporal_depth > 0:
            if hidden_dim % int(temporal_heads):
                raise ValueError("critic hidden_dim must be divisible by temporal_heads")
            self.input_projection = nn.Sequential(
                nn.RMSNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.SiLU()
            )
            self.temporal_position = nn.Parameter(
                torch.randn(self.temporal_context, hidden_dim) * 0.02
            )
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=int(temporal_heads),
                dim_feedforward=4 * hidden_dim,
                dropout=float(temporal_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(
                layer, num_layers=self.temporal_depth, norm=nn.RMSNorm(hidden_dim)
            )
            network_input = hidden_dim
        self.network = nn.Sequential(
            nn.RMSNorm(network_input),
            nn.Linear(network_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, bins),
        )
        self.coder = TwoHot(bins)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.temporal is not None:
            if state.ndim == 2:
                state = state[:, None]
            if state.ndim != 3 or state.shape[1] > self.temporal_context:
                raise ValueError("temporal critic state must have shape (B,T,D) within context")
            state = self.input_projection(state)
            state = state + self.temporal_position[-state.shape[1] :].unsqueeze(0)
            length = state.shape[1]
            causal_mask = torch.triu(
                torch.ones(length, length, device=state.device, dtype=torch.bool),
                diagonal=1,
            )
            state = self.temporal(state, mask=causal_mask)[:, -1]
        return self.network(state)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.coder.mean(self(state))


class CriticWithTarget(nn.Module):
    def __init__(self, critic: DistributionalCritic) -> None:
        super().__init__()
        self.online = critic
        self.target = copy.deepcopy(critic).requires_grad_(False)

    @torch.no_grad()
    def update_target(self, rate: float = 0.01) -> None:
        for target, online in zip(self.target.parameters(), self.online.parameters()):
            target.lerp_(online, rate)


class DistributionalActionValueCritic(nn.Module):
    """Distributional Q over a causal state history and one action chunk."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_horizon: int = 3,
        action_dim: int = 4,
        hidden_dim: int = 512,
        bins: int = 255,
        temporal_depth: int = 2,
        temporal_heads: int = 8,
        temporal_context: int = 8,
        temporal_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % int(temporal_heads):
            raise ValueError("Q hidden_dim must be divisible by temporal_heads")
        self.state_dim = int(state_dim)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.temporal_context = int(temporal_context)
        self.state_projection = nn.Sequential(
            nn.RMSNorm(state_dim), nn.Linear(state_dim, hidden_dim), nn.SiLU()
        )
        self.temporal_position = nn.Parameter(
            torch.randn(self.temporal_context, hidden_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(temporal_heads),
            dim_feedforward=4 * hidden_dim,
            dropout=float(temporal_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer, num_layers=int(temporal_depth), norm=nn.RMSNorm(hidden_dim)
        )
        self.action_projection = nn.Sequential(
            nn.RMSNorm(self.action_horizon * self.action_dim),
            nn.Linear(self.action_horizon * self.action_dim, hidden_dim),
            nn.SiLU(),
        )
        self.network = nn.Sequential(
            nn.RMSNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, bins),
        )
        self.coder = TwoHot(bins)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.ndim == 2:
            state = state[:, None]
        if state.ndim != 3 or state.shape[1] > self.temporal_context:
            raise ValueError("Q state must have shape (B,T,D) within temporal context")
        expected_action = (len(state), self.action_horizon, self.action_dim)
        if action.shape != expected_action:
            raise ValueError(f"Q action must have shape {expected_action}")
        encoded = self.state_projection(state)
        encoded = encoded + self.temporal_position[-state.shape[1]:].unsqueeze(0)
        length = state.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=state.device, dtype=torch.bool),
            diagonal=1,
        )
        state_feature = self.temporal(encoded, mask=causal_mask)[:, -1]
        action_feature = self.action_projection(action.flatten(1))
        return self.network(torch.cat([state_feature, action_feature], dim=-1))

    def value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.coder.mean(self(state, action))


class ActionValueEnsemble(nn.Module):
    """Independent distributional Q critics and Polyak target copies."""

    def __init__(self, critics: list[DistributionalActionValueCritic]) -> None:
        super().__init__()
        if len(critics) < 2:
            raise ValueError("action-value ensemble requires at least two critics")
        self.online = nn.ModuleList(critics)
        self.target = copy.deepcopy(self.online).requires_grad_(False)

    def values(
        self, state: torch.Tensor, action: torch.Tensor, *, target: bool = False
    ) -> torch.Tensor:
        critics = self.target if target else self.online
        return torch.stack([critic.value(state, action) for critic in critics], dim=0)

    @torch.no_grad()
    def update_target(self, rate: float = 0.01) -> None:
        for target, online in zip(self.target.parameters(), self.online.parameters()):
            target.lerp_(online, rate)


def lambda_returns(
    reward: torch.Tensor,
    continuation: torch.Tensor,
    next_value: torch.Tensor,
    *,
    discount: float = 0.997,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Standard TD(lambda) targets for transition-aligned imagined rewards."""

    if reward.shape != continuation.shape:
        raise ValueError("reward and continuation must have identical shapes")
    if next_value.shape != reward.shape:
        raise ValueError("next_value must provide one bootstrap value per transition")
    result = torch.empty_like(reward)
    bootstrap = next_value[:, -1]
    for index in range(reward.shape[1] - 1, -1, -1):
        mixed = (1.0 - float(lambda_)) * next_value[:, index] + float(lambda_) * bootstrap
        bootstrap = reward[:, index] + float(discount) * continuation[:, index] * mixed
        result[:, index] = bootstrap
    return result


def diagonal_gaussian_kl(
    policy: GaussianPolicyOutput, prior: GaussianPolicyOutput
) -> torch.Tensor:
    """Reverse KL ``KL(policy || prior)`` summed over one action chunk."""

    variance_ratio = torch.exp(2.0 * (policy.log_std - prior.log_std))
    mean_term = (policy.mean - prior.mean).square() * torch.exp(-2.0 * prior.log_std)
    elementwise = (
        prior.log_std - policy.log_std
        + 0.5 * (variance_ratio + mean_term)
        - 0.5
    )
    return elementwise.sum(dim=(-1, -2))


def pmpo_policy_loss(
    policy: GaussianPolicyOutput,
    prior: GaussianPolicyOutput,
    action: torch.Tensor,
    advantage: torch.Tensor,
    *,
    positive_weight: float = 0.5,
    prior_kl_weight: float = 0.3,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dreamer-4 PMPO: balanced sign feedback plus reverse behavioral KL.

    Positive samples maximize their likelihood, negative samples minimize it,
    and both sets are normalized independently so reward scale and class balance
    cannot dominate the update.
    """

    if not 0.0 <= positive_weight <= 1.0:
        raise ValueError("positive_weight must lie in [0,1]")
    log_probability = policy.log_prob(action).sum(dim=-1)
    if log_probability.shape != advantage.shape:
        raise ValueError("advantage must provide one value per sampled action chunk")
    weight = (
        torch.ones_like(advantage)
        if sample_weight is None else sample_weight.to(advantage.dtype)
    )
    positive = advantage >= 0
    negative = ~positive

    def weighted_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = weight * mask.to(weight.dtype)
        return (value * selected).sum() / selected.sum().clamp_min(1.0)

    positive_loss = -weighted_mean(log_probability, positive)
    negative_loss = weighted_mean(log_probability, negative)
    reverse_kl = diagonal_gaussian_kl(policy, prior)
    kl_loss = (reverse_kl * weight).sum() / weight.sum().clamp_min(1.0)
    loss = (
        float(positive_weight) * positive_loss
        + (1.0 - float(positive_weight)) * negative_loss
        + float(prior_kl_weight) * kl_loss
    )
    return loss, {
        "loss": loss.detach(),
        "positive_loss": positive_loss.detach(),
        "negative_loss": negative_loss.detach(),
        "reverse_kl": kl_loss.detach(),
        "positive_fraction": positive.float().mean().detach(),
        "log_probability": log_probability.mean().detach(),
    }
