"""Compact privileged-state racing policy and shared training utilities.

This module intentionally implements the smallest control contract needed to
establish a BC -> DAgger -> exact PPO racing baseline.  The actor receives only
continuous state/route/action features available before the next command and
emits one normalized CTBR command at 90 Hz.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starscream.DiT import TokenizerFlowPolicy
from starscream.plant_privileged import PLANT_OBSERVATION_CONTRACT, PLANT_SETTINGS_DIM
from starscream.world_privileged import WORLD_OBSERVATION_CONTRACT, WORLD_OBSERVATION_CONTRACTS, WORLD_DISPLACEMENT_CONTRACT, world_observation_features
from starscream.sequence_architecture import (
    ModernTemporalEncoder,
    RouteSemanticStepTokenizer,
    SwiGLUActionMLP,
    UnifiedRouteTransformer,
)
from starscream.agile_generalization import (
    GREEN2026_BODY_RATES,
    GREEN2026_FEATURE_DIM,
    GREEN2026_GATE_COUNT,
    GREEN2026_GATE_TO_GATE_CORNERS,
    GREEN2026_LINEAR_VELOCITY,
    GREEN2026_NEXT_GATE_CORNERS,
    GREEN2026_OBSERVATION_CONTRACT,
    GREEN2026_PREVIOUS_ACTION,
    GREEN2026_ROTATION,
    GREEN2026_ROUTE_CHAIN_OBSERVATION_CONTRACT,
    LEGACY_OBSERVATION_CONTRACT,
    green2026_batch_features,
    green2026_observation_features,
    green2026_route_chain_batch_features,
    green2026_route_chain_observation_features,
    observation_contract_feature_dim,
)


ACTION_LOW = np.asarray([0.0, -6.0, -6.0, -6.0], np.float32)
# Collective thrust uses the project-wide [0, 30] m/s^2 CTBR contract.  Hover
# is therefore near normalized zero, not at the upper action boundary.
ACTION_HIGH = np.asarray([30.0, 6.0, 6.0, 6.0], np.float32)
TASK_DIM = 19
ROUTE_GATES = 3
ROUTE_RECORD_DIM = 13
FEATURE_DIM = TASK_DIM + ROUTE_GATES * ROUTE_RECORD_DIM + 4 + 2


def privileged_feature_dim(
    route_gates: int = ROUTE_GATES,
    observation_contract: str = LEGACY_OBSERVATION_CONTRACT,
) -> int:
    """Return the flat actor feature width for a route lookahead contract."""

    route_gates = int(route_gates)
    if route_gates < 1:
        raise ValueError("route_gates must be positive")
    legacy = TASK_DIM + route_gates * ROUTE_RECORD_DIM + 4 + 2
    return observation_contract_feature_dim(
        observation_contract, legacy_feature_dim=legacy, route_gates=route_gates,
    )


def route_gates_from_feature_dim(width: int) -> int:
    """Invert :func:`privileged_feature_dim` with strict schema validation."""

    payload = int(width) - TASK_DIM - 4 - 2
    if payload < ROUTE_RECORD_DIM or payload % ROUTE_RECORD_DIM:
        raise ValueError(f"feature width {width} is not a privileged route schema")
    return payload // ROUTE_RECORD_DIM


def ctbr_to_normalized(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, np.float32)
    return (2.0 * (action - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW) - 1.0).clip(
        -1.0, 1.0
    ).astype(np.float32)


def normalized_to_ctbr(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, np.float32)
    return (ACTION_LOW + 0.5 * (action.clip(-1.0, 1.0) + 1.0) * (
        ACTION_HIGH - ACTION_LOW
    )).astype(np.float32)


def observation_features(
    observation: Mapping[str, Any], *, route_gates: int | None = None,
    observation_contract: str = LEGACY_OBSERVATION_CONTRACT,
    action_encoder: Any = ctbr_to_normalized,
) -> np.ndarray:
    """Return the causal privileged actor observation for N future gates."""

    if str(observation_contract) == PLANT_OBSERVATION_CONTRACT:
        base = observation_features(observation, route_gates=route_gates,
            observation_contract=LEGACY_OBSERVATION_CONTRACT, action_encoder=action_encoder)
        settings = np.asarray(observation['privileged']['plant_settings'], np.float32)
        if settings.shape != (PLANT_SETTINGS_DIM,) or not np.isfinite(settings).all():
            raise ValueError('Missing or invalid applied plant descriptor')
        return np.concatenate([base, settings])

    if str(observation_contract) in WORLD_OBSERVATION_CONTRACTS:
        count = len(observation['privileged']['world_route_records']) if route_gates is None else int(route_gates)
        return world_observation_features(observation, count, action_encoder,
            displacement=str(observation_contract) == WORLD_DISPLACEMENT_CONTRACT)

    if str(observation_contract) == GREEN2026_OBSERVATION_CONTRACT:
        return green2026_observation_features(
            observation, action_encoder=action_encoder,
        )
    if str(observation_contract) == GREEN2026_ROUTE_CHAIN_OBSERVATION_CONTRACT:
        expected_gates = (
            len(observation["gates"]["position"])
            if route_gates is None else int(route_gates)
        )
        return green2026_route_chain_observation_features(
            observation, route_gates=expected_gates, action_encoder=action_encoder,
        )
    if str(observation_contract) != LEGACY_OBSERVATION_CONTRACT:
        raise ValueError(
            f"unknown privileged observation contract {observation_contract!r}"
        )

    task = np.asarray(observation["task_state"], np.float32)
    route = np.asarray(observation["flight_plan"]["records"], np.float32)
    previous = np.asarray(
        action_encoder(np.asarray(observation["previous_action"], np.float32)),
        np.float32,
    )
    if previous.shape != (4,) or not np.all(np.isfinite(previous)):
        raise ValueError("encoded previous action must be finite CTBR[4]")
    age = float(observation["age"]["previous_action"])
    valid = float(observation["valid"]["previous_action"])
    expected_gates = int(route.shape[0]) if route_gates is None else int(route_gates)
    if task.shape != (TASK_DIM,) or route.shape != (expected_gates, ROUTE_RECORD_DIM):
        raise ValueError("privileged racing observation has an incompatible shape")
    result = np.concatenate(
        [task, route.reshape(-1), previous, np.asarray([age, valid], np.float32)]
    ).astype(np.float32)
    expected_dim = privileged_feature_dim(expected_gates)
    if result.shape != (expected_dim,) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"privileged racing features must be finite and {expected_dim}-dimensional"
        )
    return result


class Green2026StepTokenizer(nn.Module):
    """Semantic within-transition attention before Starscream temporal attention.

    The paper used a two-layer MLP and one observation at a time.  Starscream's
    controlled extension keeps the exact 40 values but represents them as a
    dynamics token, previous-command token, and eight corner tokens.  A learned
    readout attends within each transition; the existing causal Transformer then
    attends across the resulting history of transition embeddings.
    """

    def __init__(
        self, dimension: int, *, layers: int, heads: int, dropout: float,
    ) -> None:
        super().__init__()
        if layers < 1 or heads < 1 or dimension % heads:
            raise ValueError("invalid Green-2026 semantic tokenizer dimensions")
        self.dimension = int(dimension)
        self.dynamics_projection = nn.Linear(12, dimension)
        self.previous_action_projection = nn.Linear(4, dimension)
        self.corner_projection = nn.Linear(3, dimension)
        # readout + dynamics + previous command + 4 next corners + 4 deltas
        self.slot_embedding = nn.Parameter(torch.empty(1, 11, dimension))
        self.readout = nn.Parameter(torch.empty(1, 1, dimension))
        nn.init.trunc_normal_(self.slot_embedding, std=0.02)
        nn.init.trunc_normal_(self.readout, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=4 * dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(dimension)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != GREEN2026_FEATURE_DIM:
            raise ValueError("semantic tokenizer requires [B,T,40] features")
        batch, steps, _ = features.shape
        dynamics = torch.cat([
            features[..., GREEN2026_ROTATION],
            features[..., GREEN2026_LINEAR_VELOCITY],
            features[..., GREEN2026_BODY_RATES],
        ], dim=-1)
        previous = features[..., GREEN2026_PREVIOUS_ACTION]
        next_corners = features[..., GREEN2026_NEXT_GATE_CORNERS].reshape(
            batch, steps, 4, 3
        )
        gate_deltas = features[..., GREEN2026_GATE_TO_GATE_CORNERS].reshape(
            batch, steps, 4, 3
        )
        tokens = torch.cat([
            self.dynamics_projection(dynamics)[:, :, None],
            self.previous_action_projection(previous)[:, :, None],
            self.corner_projection(next_corners),
            self.corner_projection(gate_deltas),
        ], dim=2).reshape(batch * steps, 10, self.dimension)
        readout = self.readout.expand(batch * steps, -1, -1)
        tokens = torch.cat([readout, tokens], dim=1) + self.slot_embedding
        return self.norm(self.encoder(tokens)[:, 0]).reshape(
            batch, steps, self.dimension
        )


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, np.float32)
        std = np.asarray(self.std, np.float32)
        if mean.ndim != 1 or std.shape != mean.shape or not len(mean):
            raise ValueError("feature normalizer must contain aligned 1D statistics")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("feature normalizer must be finite with positive scales")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureNormalizer":
        values = np.asarray(values, np.float32)
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError("normalizer data must have shape (N,D)")
        return cls(values.mean(0), np.maximum(values.std(0), 1.0e-4))

    def numpy(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, np.float32) - self.mean) / self.std).astype(np.float32)

    def tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=values.device, dtype=values.dtype)
        std = torch.as_tensor(self.std, device=values.device, dtype=values.dtype)
        return (values - mean) / std

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"mean": self.mean.copy(), "std": self.std.copy()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FeatureNormalizer":
        return cls(np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32))


class SquashedGaussian:
    def __init__(self, location: torch.Tensor, log_std: torch.Tensor) -> None:
        self.location = location
        self.log_std = log_std.expand_as(location)

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp()

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.location)

    def rsample(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.location + self.std * torch.randn_like(self.location)
        return torch.tanh(raw), raw

    def log_prob(self, action: torch.Tensor, raw: torch.Tensor | None = None) -> torch.Tensor:
        clipped = action.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        raw = torch.atanh(clipped) if raw is None else raw
        base = -0.5 * ((raw - self.location) / self.std).square()
        base = base - self.log_std - 0.5 * np.log(2.0 * np.pi)
        jacobian = torch.log(1.0 - clipped.square() + 1.0e-6)
        return (base - jacobian).sum(-1)

    def base_entropy(self) -> torch.Tensor:
        return (self.log_std + 0.5 * (1.0 + np.log(2.0 * np.pi))).sum(-1)


class PrivilegedMLPPolicy(nn.Module):
    """Causal dynamics encoder with an exact-likelihood MLP CTBR head.

    The action remains a single 90 Hz command so PPO has an exact, local credit
    assignment contract.  The policy state is not one-step: a recurrent encoder
    integrates a configurable continuous history before the MLP control head.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 256,
        depth: int = 3,
        recurrent_dim: int = 256,
        recurrent_depth: int = 2,
        context_steps: int = 18,
        encoder_type: str = "gru",
        attention_heads: int = 8,
        transformer_feedforward_dim: int | None = None,
        transformer_dropout: float = 0.0,
        observation_contract: str = LEGACY_OBSERVATION_CONTRACT,
        structured_observation_tokens: bool = False,
        route_semantic_tokens: bool = False,
        unified_route_transformer: bool = False,
        unified_readout_mode: str = "final_route",
        unified_readout_decoder_depth: int = 2,
        unified_readout_decoder_queries: int = 2,
        unified_readout_initialization_std: float = 0.02,
        continuous_input_rms_norm: bool = True,
        unified_bidirectional: bool = False,
        route_feature_mode: str = "none",
        route_head_modulation: bool = False,
        separate_route_encoder: bool = False,
        observation_token_layers: int = 2,
        observation_token_heads: int | None = None,
        observation_embedding_dropout: float = 0.0,
        continuous_projection_fourier_features: int = 0,
        continuous_projection_fourier_scale: float = 2.0,
        modern_transformer_blocks: bool = False,
        modern_adaln_zero: bool = True,
        temporal_alibi: bool = False,
        history_condition_dropout_probability: float = 0.0,
        history_guidance_scale: float = 1.0,
        swiglu_action_head: bool = False,
        output_initialization_scale: float = 1.0e-3,
        gate_contrastive_dim: int = 0,
        initial_log_std: float = -1.2,
        minimum_log_std: float = -3.5,
        maximum_log_std: float = 0.0,
        speed_conditioning: bool = False,
        speed_conditioning_scale: float = 20.0,
        default_speed_command: float = 0.0,
        topology_target_dim: int = 0,
        action_chunk_steps: int = 1,
        action_chunk_decoder_layers: int = 0,
        action_chunk_decoder_heads: int = 8,
        action_chunk_decoder_feedforward_dim: int | None = None,
        action_conditioned_dynamics_dim: int = 0,
        reward_aux_dim: int = 0,
        action_head_type: str = "mlp",
        action_mixture_centers: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
        action_mixture_cluster_weights: list[float] | tuple[float, ...] | None = None,
        action_mixture_residual_logit_scale: float = 2.0,
        flow_depth: int = 3,
        flow_heads: int = 8,
        flow_mlp_ratio: int = 4,
        flow_sampling_steps: int = 3,
        flow_sampling_method: str = "euler",
        flow_source_noise: float = 0.25,
        flow_deterministic_source: str = "fixed_gaussian",
        flow_context_mode: str = "encoded_history",
        flow_context_depth: int = 0,
        flow_shortcut_step_sizes: tuple[float, ...] | list[float] = (
            1.0 / 12.0, 1.0 / 6.0, 1.0 / 3.0,
        ),
    ) -> None:
        super().__init__()
        if (
            input_dim < 1 or hidden_dim < 16 or depth < 1
            or recurrent_dim < 16 or recurrent_depth < 1 or context_steps < 1
        ):
            raise ValueError("invalid compact policy dimensions")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.recurrent_dim = int(recurrent_dim)
        self.recurrent_depth = int(recurrent_depth)
        self.context_steps = int(context_steps)
        self.encoder_type = str(encoder_type).lower()
        self.attention_heads = int(attention_heads)
        self.transformer_feedforward_dim = int(
            transformer_feedforward_dim or 4 * self.recurrent_dim
        )
        self.transformer_dropout = float(transformer_dropout)
        self.observation_contract = str(observation_contract)
        self.plant_settings_dim = PLANT_SETTINGS_DIM if self.observation_contract == PLANT_OBSERVATION_CONTRACT else 0
        if self.plant_settings_dim and (not unified_route_transformer or flow_context_mode != 'encoded_history'):
            raise ValueError('Plant contract requires the canonical unified route actor')
        self.continuous_input_rms_norm = bool(continuous_input_rms_norm)
        self.unified_bidirectional = bool(unified_bidirectional)
        self.route_feature_mode = str(route_feature_mode)
        self.route_head_modulation = bool(route_head_modulation)
        self.separate_route_encoder = bool(separate_route_encoder)
        if (self.route_feature_mode != 'none' or self.route_head_modulation or self.separate_route_encoder) and (
            not unified_route_transformer or observation_contract not in {LEGACY_OBSERVATION_CONTRACT, PLANT_OBSERVATION_CONTRACT}
        ):
            raise ValueError('route ablations require the legacy privileged unified transformer')
        if self.unified_bidirectional and not unified_route_transformer:
            raise ValueError('bidirectional option requires unified route encoder')
        if not self.continuous_input_rms_norm and not unified_route_transformer:
            raise ValueError('input normalization ablation requires unified route encoder')
        self.structured_observation_tokens = bool(structured_observation_tokens)
        self.route_semantic_tokens = bool(route_semantic_tokens)
        self.unified_route_transformer = bool(unified_route_transformer)
        self.unified_readout_mode = str(unified_readout_mode)
        self.unified_readout_decoder_depth = int(unified_readout_decoder_depth)
        self.unified_readout_decoder_queries = int(unified_readout_decoder_queries)
        self.unified_readout_initialization_std = float(
            unified_readout_initialization_std
        )
        self.observation_token_layers = int(observation_token_layers)
        self.observation_token_heads = int(
            observation_token_heads or self.attention_heads
        )
        self.observation_embedding_dropout = float(observation_embedding_dropout)
        self.continuous_projection_fourier_features = int(
            continuous_projection_fourier_features
        )
        self.continuous_projection_fourier_scale = float(
            continuous_projection_fourier_scale
        )
        self.modern_transformer_blocks = bool(modern_transformer_blocks)
        self.modern_adaln_zero = bool(modern_adaln_zero)
        self.temporal_alibi = bool(temporal_alibi)
        self.history_condition_dropout_probability = float(
            history_condition_dropout_probability
        )
        self.history_guidance_scale = float(history_guidance_scale)
        self.swiglu_action_head = bool(swiglu_action_head)
        self.output_initialization_scale = float(output_initialization_scale)
        self.gate_contrastive_dim = int(gate_contrastive_dim)
        if self.encoder_type not in {"gru", "transformer", "mlp"}:
            raise ValueError("encoder_type must be 'gru', 'transformer', or 'mlp'")
        if self.encoder_type == "mlp" and self.context_steps != 1:
            raise ValueError("the transition-MLP control requires context_steps=1")
        if self.encoder_type == "transformer" and (
            self.recurrent_dim % self.attention_heads != 0
            or self.attention_heads < 1
            or self.transformer_feedforward_dim < self.recurrent_dim
        ):
            raise ValueError("invalid privileged Transformer dimensions")
        if (
            not 0.0 <= self.observation_embedding_dropout < 1.0
            or self.continuous_projection_fourier_features < 0
            or not np.isfinite(self.continuous_projection_fourier_scale)
            or self.continuous_projection_fourier_scale <= 0.0
            or not 0.0 <= self.history_condition_dropout_probability < 1.0
            or not np.isfinite(self.history_guidance_scale)
            or self.history_guidance_scale < 0.0
            or not np.isfinite(self.output_initialization_scale)
            or self.output_initialization_scale <= 0.0
            or self.gate_contrastive_dim < 0
            or self.unified_readout_decoder_depth < 1
            or self.unified_readout_decoder_queries < 1
            or not np.isfinite(self.unified_readout_initialization_std)
            or self.unified_readout_initialization_std <= 0.0
        ):
            raise ValueError("invalid modern sequence architecture controls")
        self.minimum_log_std = float(minimum_log_std)
        self.maximum_log_std = float(maximum_log_std)
        self.speed_conditioning = bool(speed_conditioning)
        self.speed_conditioning_scale = float(speed_conditioning_scale)
        self.default_speed_command = float(default_speed_command)
        self.topology_target_dim = int(topology_target_dim)
        self.action_chunk_steps = int(action_chunk_steps)
        self.action_chunk_decoder_layers = int(action_chunk_decoder_layers)
        self.action_chunk_decoder_heads = int(action_chunk_decoder_heads)
        self.action_chunk_decoder_feedforward_dim = int(
            action_chunk_decoder_feedforward_dim or 4 * self.recurrent_dim
        )
        self.action_conditioned_dynamics_dim = int(
            action_conditioned_dynamics_dim
        )
        self.reward_aux_dim = int(reward_aux_dim)
        self.action_head_type = str(action_head_type)
        self.action_mixture_residual_logit_scale = float(
            action_mixture_residual_logit_scale
        )
        self.flow_depth = int(flow_depth)
        self.flow_heads = int(flow_heads)
        self.flow_mlp_ratio = int(flow_mlp_ratio)
        self.flow_sampling_steps = int(flow_sampling_steps)
        self.flow_sampling_method = str(flow_sampling_method)
        self.flow_source_noise = float(flow_source_noise)
        self.flow_deterministic_source = str(flow_deterministic_source)
        if self.flow_deterministic_source not in {'fixed_gaussian','zero'}:
            raise ValueError('invalid deterministic flow source')
        self.flow_context_mode = str(flow_context_mode)
        self.flow_context_depth = int(flow_context_depth)
        if self.observation_contract in {LEGACY_OBSERVATION_CONTRACT, PLANT_OBSERVATION_CONTRACT, *WORLD_OBSERVATION_CONTRACTS}:
            self.flow_route_gates = route_gates_from_feature_dim(self.input_dim - self.plant_settings_dim)
        elif self.observation_contract == GREEN2026_OBSERVATION_CONTRACT:
            if self.input_dim != GREEN2026_FEATURE_DIM:
                raise ValueError(
                    "Green-2026 observation contract requires input_dim=40"
                )
            self.flow_route_gates = GREEN2026_GATE_COUNT
        elif self.observation_contract == GREEN2026_ROUTE_CHAIN_OBSERVATION_CONTRACT:
            payload = self.input_dim - 16
            if payload < 12 or payload % 12:
                raise ValueError(
                    "Green route-chain observation requires input_dim=16+12*N"
                )
            self.flow_route_gates = payload // 12
        else:
            raise ValueError(
                f"unknown privileged observation contract {self.observation_contract!r}"
            )
        self.flow_shortcut_step_sizes = tuple(
            float(value) for value in flow_shortcut_step_sizes
        )
        if self.action_head_type not in {"mlp", "shortcut_flow"}:
            raise ValueError("action_head_type must be mlp or shortcut_flow")
        if self.action_mixture_residual_logit_scale <= 0.0:
            raise ValueError("action mixture residual logit scale must be positive")
        if action_mixture_centers is not None and self.action_head_type != "mlp":
            raise ValueError("action mixtures require the direct MLP action head")
        if self.flow_context_mode not in {
            "encoded_history", "structured_privileged_dit", "unified_route_dit"
        }:
            raise ValueError(
                "flow_context_mode must be encoded_history or "
                "structured_privileged_dit"
            )
        if self.flow_context_depth < 0:
            raise ValueError("flow_context_depth cannot be negative")
        if (
            self.flow_context_mode == "structured_privileged_dit"
            and self.encoder_type != "transformer"
        ):
            raise ValueError("structured privileged DiT requires encoder_type=transformer")
        if (
            self.flow_context_mode == "structured_privileged_dit"
            and self.observation_contract != LEGACY_OBSERVATION_CONTRACT
        ):
            raise ValueError(
                "legacy structured flow tokenization is incompatible with the "
                "Green-2026 contract; use structured_observation_tokens"
            )
        if self.structured_observation_tokens and (
            self.observation_contract != GREEN2026_OBSERVATION_CONTRACT
            or self.encoder_type != "transformer"
            or self.flow_context_mode == "structured_privileged_dit"
        ):
            raise ValueError(
                "structured_observation_tokens requires the Green-2026 contract, "
                "a Transformer encoder, and the standard history context"
            )
        if self.route_semantic_tokens and (
            self.observation_contract not in {LEGACY_OBSERVATION_CONTRACT, PLANT_OBSERVATION_CONTRACT, *WORLD_OBSERVATION_CONTRACTS}
            or self.encoder_type != "transformer"
            or self.flow_context_mode == "structured_privileged_dit"
        ):
            raise ValueError(
                "route_semantic_tokens requires the legacy route contract, "
                "a Transformer encoder, and the standard history context"
            )
        if self.route_semantic_tokens and self.structured_observation_tokens:
            raise ValueError("only one semantic observation tokenizer may be active")
        if self.unified_route_transformer and not self.route_semantic_tokens:
            raise ValueError("unified route Transformer requires route semantic tokens")
        if self.unified_route_transformer and not self.modern_transformer_blocks:
            raise ValueError("unified route Transformer requires modern blocks")
        if self.flow_context_mode == 'unified_route_dit' and not (
            self.unified_route_transformer and self.action_head_type == 'shortcut_flow'
            and self.action_chunk_steps == 1 and self.unified_readout_mode == 'final_route'
            and not self.history_condition_dropout_probability and not self.temporal_alibi
        ):
            raise ValueError('unified flow requires canonical unified single-action tokens without history/ALiBi ablations')
        if not self.unified_route_transformer and self.unified_readout_mode != "final_route":
            raise ValueError("unified readout modes require the unified route Transformer")
        if self.modern_transformer_blocks and self.encoder_type != "transformer":
            raise ValueError("modern Transformer blocks require encoder_type=transformer")
        if self.temporal_alibi and not self.modern_transformer_blocks:
            raise ValueError("temporal ALiBi requires modern Transformer blocks")
        if self.gate_contrastive_dim and not self.route_semantic_tokens:
            raise ValueError("gate contrastive learning requires route semantic tokens")
        if self.flow_sampling_steps < 1 or self.flow_sampling_method not in {
            "euler", "heun"
        }:
            raise ValueError("invalid shortcut-flow sampling contract")
        if self.speed_conditioning_scale <= 0 or self.default_speed_command < 0:
            raise ValueError("speed-conditioning scale must be positive and command non-negative")
        if (
            self.topology_target_dim < 0
            or self.action_conditioned_dynamics_dim < 0
            or self.reward_aux_dim < 0
        ):
            raise ValueError("auxiliary target dimensions cannot be negative")
        if self.action_chunk_steps < 1:
            raise ValueError("action_chunk_steps must be positive")
        if self.action_chunk_decoder_layers < 0:
            raise ValueError("action_chunk_decoder_layers cannot be negative")
        if self.action_chunk_decoder_layers and (
            self.action_chunk_steps < 2
            or self.action_chunk_decoder_heads < 1
            or self.recurrent_dim % self.action_chunk_decoder_heads != 0
            or self.action_chunk_decoder_feedforward_dim < self.recurrent_dim
        ):
            raise ValueError("invalid action-chunk Transformer decoder dimensions")
        if self.flow_context_mode == "structured_privileged_dit":
            # The structured flow policy owns temporal attention.  Keep the
            # legacy encoder path parameter-free so a checkpoint cannot appear
            # to be a unified DiT while silently optimizing an unused backbone.
            self.step_embedding: nn.Module = nn.Identity()
            self.recurrent: nn.Module = nn.Identity()
            self.register_parameter("temporal_position", None)
            self.temporal_norm: nn.Module = nn.Identity()
        elif self.unified_route_transformer:
            self.step_embedding = UnifiedRouteTransformer(
                self.input_dim,
                self.recurrent_dim,
                plant_settings_dim=self.plant_settings_dim,
                context_steps=self.context_steps,
                depth=self.recurrent_depth,
                heads=self.attention_heads,
                feedforward_dim=self.transformer_feedforward_dim,
                dropout=self.transformer_dropout,
                adaln_zero=self.modern_adaln_zero,
                alibi=self.temporal_alibi,
                readout_mode=self.unified_readout_mode,
                input_rms_norm=self.continuous_input_rms_norm,
                bidirectional=self.unified_bidirectional,
                route_feature_mode=self.route_feature_mode,
                route_head_modulation=self.route_head_modulation,
                separate_route_encoder=self.separate_route_encoder,
                readout_decoder_depth=self.unified_readout_decoder_depth,
                readout_decoder_queries=self.unified_readout_decoder_queries,
                readout_initialization_std=(
                    self.unified_readout_initialization_std
                ),
                fourier_features=self.continuous_projection_fourier_features,
                fourier_scale=self.continuous_projection_fourier_scale,
            )
            self.recurrent = nn.Identity()
            self.register_parameter("temporal_position", None)
            self.temporal_norm = nn.Identity()
        elif self.structured_observation_tokens:
            self.step_embedding = Green2026StepTokenizer(
                self.recurrent_dim,
                layers=self.observation_token_layers,
                heads=self.observation_token_heads,
                dropout=self.transformer_dropout,
            )
        elif self.route_semantic_tokens:
            self.step_embedding = RouteSemanticStepTokenizer(
                self.input_dim,
                self.recurrent_dim,
                layers=self.observation_token_layers,
                heads=self.observation_token_heads,
                feedforward_dim=self.transformer_feedforward_dim,
                dropout=self.transformer_dropout,
                adaln_zero=self.modern_adaln_zero,
                fourier_features=self.continuous_projection_fourier_features,
                fourier_scale=self.continuous_projection_fourier_scale,
            )
        else:
            self.step_embedding = nn.Sequential(
                nn.Linear(self.input_dim, self.recurrent_dim),
                nn.LayerNorm(self.recurrent_dim),
                nn.SiLU(),
                nn.Linear(self.recurrent_dim, self.recurrent_dim),
                nn.SiLU(),
            )
        if (
            not self.unified_route_transformer
            and
            self.flow_context_mode != "structured_privileged_dit"
            and self.encoder_type == "gru"
        ):
            self.recurrent = nn.GRU(
                self.recurrent_dim,
                self.recurrent_dim,
                num_layers=self.recurrent_depth,
                batch_first=True,
            )
            self.register_parameter("temporal_position", None)
            self.temporal_norm: nn.Module = nn.Identity()
        elif (
            not self.unified_route_transformer
            and
            self.flow_context_mode != "structured_privileged_dit"
            and self.encoder_type == "transformer"
        ):
            self.temporal_position = nn.Parameter(
                torch.zeros(1, self.context_steps, self.recurrent_dim)
            )
            nn.init.trunc_normal_(self.temporal_position, std=0.02)
            if self.modern_transformer_blocks:
                self.recurrent = ModernTemporalEncoder(
                    self.recurrent_dim,
                    depth=self.recurrent_depth,
                    heads=self.attention_heads,
                    feedforward_dim=self.transformer_feedforward_dim,
                    dropout=self.transformer_dropout,
                    adaln_zero=self.modern_adaln_zero,
                    alibi=self.temporal_alibi,
                )
                self.temporal_norm = nn.Identity()
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=self.recurrent_dim,
                    nhead=self.attention_heads,
                    dim_feedforward=self.transformer_feedforward_dim,
                    dropout=self.transformer_dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.recurrent = nn.TransformerEncoder(
                    layer, num_layers=self.recurrent_depth, enable_nested_tensor=False
                )
                self.temporal_norm = nn.LayerNorm(self.recurrent_dim)
        elif (
            not self.unified_route_transformer
            and self.flow_context_mode != "structured_privileged_dit"
        ):
            # A strict one-transition control: retain the identical nonlinear
            # observation projector and downstream CTBR/dynamics heads while
            # removing every temporal mixing layer.
            self.recurrent = nn.Identity()
            self.register_parameter("temporal_position", None)
            self.temporal_norm = nn.Identity()
        self.flow_task_projection: nn.Module | None = None
        self.flow_route_projection: nn.Module | None = None
        self.flow_previous_action_projection: nn.Module | None = None
        self.flow_timing_projection: nn.Module | None = None
        self.flow_speed_projection: nn.Module | None = None
        self.flow_structured_slot: nn.Parameter | None = None
        if self.flow_context_mode == "structured_privileged_dit":
            self.flow_task_projection = nn.Linear(TASK_DIM, self.recurrent_dim)
            self.flow_route_projection = nn.Linear(
                ROUTE_RECORD_DIM, self.recurrent_dim
            )
            self.flow_previous_action_projection = nn.Linear(
                4, self.recurrent_dim
            )
            self.flow_timing_projection = nn.Linear(2, self.recurrent_dim)
            self.flow_speed_projection = nn.Sequential(
                nn.Linear(1, self.recurrent_dim),
                nn.SiLU(),
                nn.Linear(self.recurrent_dim, self.recurrent_dim),
            )
            self.flow_structured_slot = nn.Parameter(torch.empty(
                1, 1, self.flow_route_gates + 4, self.recurrent_dim
            ))
            nn.init.trunc_normal_(self.flow_structured_slot, std=0.02)
        layers: list[nn.Module] = []
        width = self.recurrent_dim
        for _ in range(self.depth):
            layers.extend([nn.Linear(width, self.hidden_dim), nn.LeakyReLU(0.2)])
            width = self.hidden_dim
        # Receding-horizon action head.  The first CTBR command is the only one
        # executed (and the only one exposed to PPO); later commands are
        # training-time sequence targets that regularize the shared encoding.
        layers.append(nn.Linear(
            width,
            4 if (
                self.action_head_type == "shortcut_flow"
                or self.action_chunk_decoder_layers
            ) else 4 * self.action_chunk_steps,
        ))
        action_output_dim = (
            4 if (
                self.action_head_type == "shortcut_flow"
                or self.action_chunk_decoder_layers
            ) else 4 * self.action_chunk_steps
        )
        if self.swiglu_action_head:
            self.mean_network = SwiGLUActionMLP(
                self.recurrent_dim, self.hidden_dim, self.depth,
                action_output_dim, output_scale=self.output_initialization_scale,
            )
        else:
            self.mean_network = nn.Sequential(*layers)
            nn.init.uniform_(
                self.mean_network[-1].weight,
                -self.output_initialization_scale,
                self.output_initialization_scale,
            )
            nn.init.zeros_(self.mean_network[-1].bias)
        self.action_chunk_queries: nn.Parameter | None = None
        self.action_chunk_decoder: nn.Module | None = None
        self.action_chunk_output: nn.Module | None = None
        if self.action_chunk_decoder_layers:
            self.action_chunk_queries = nn.Parameter(torch.empty(
                self.action_chunk_steps, self.recurrent_dim
            ))
            nn.init.trunc_normal_(self.action_chunk_queries, std=0.02)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.recurrent_dim,
                nhead=self.action_chunk_decoder_heads,
                dim_feedforward=self.action_chunk_decoder_feedforward_dim,
                dropout=self.transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.action_chunk_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=self.action_chunk_decoder_layers,
                norm=nn.LayerNorm(self.recurrent_dim),
            )
            self.action_chunk_output = nn.Linear(self.recurrent_dim, 4)
            nn.init.uniform_(self.action_chunk_output.weight, -1.0e-3, 1.0e-3)
            nn.init.zeros_(self.action_chunk_output.bias)
        self.flow_action_head: TokenizerFlowPolicy | None = None
        if self.action_head_type == "shortcut_flow":
            self._create_shortcut_flow_action_head()
        self.register_buffer("action_mixture_centers", None)
        self.register_buffer("action_mixture_cluster_weights", None)
        self.action_mixture_mode_head: nn.Module | None = None
        self.action_mixture_residual_head: nn.Module | None = None
        if action_mixture_centers is not None:
            self._create_action_mixture_modules(
                torch.as_tensor(action_mixture_centers, dtype=torch.float32),
                (
                    None if action_mixture_cluster_weights is None
                    else torch.as_tensor(
                        action_mixture_cluster_weights, dtype=torch.float32
                    )
                ),
            )
        self.log_std_parameter = nn.Parameter(torch.full((4,), float(initial_log_std)))
        self.dynamics_head = nn.Sequential(
            nn.Linear(self.recurrent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, TASK_DIM),
        )
        self.action_conditioned_dynamics_head: nn.Module | None = None
        if self.action_conditioned_dynamics_dim:
            self._create_action_conditioned_dynamics_head()
        self.reward_aux_head: nn.Module | None = None
        if self.reward_aux_dim:
            # Reward is action-conditioned: under mixed-policy DAgger the same
            # state can receive a different immediate reward depending on the
            # command that actually entered the plant.  Conditioning avoids
            # asking the shared representation to average those outcomes.
            self.reward_aux_head = nn.Sequential(
                nn.Linear(self.recurrent_dim + 4, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.reward_aux_dim),
            )
        self.gate_contrastive_query: nn.Module | None = None
        self.gate_contrastive_target: nn.Module | None = None
        if self.gate_contrastive_dim:
            self.gate_contrastive_query = nn.Sequential(
                nn.RMSNorm(self.recurrent_dim),
                nn.Linear(self.recurrent_dim, self.gate_contrastive_dim),
            )
            self.gate_contrastive_target = nn.Sequential(
                nn.RMSNorm(ROUTE_RECORD_DIM),
                nn.Linear(ROUTE_RECORD_DIM, self.gate_contrastive_dim),
            )
        self.observation_embedding_dropout_layer = nn.Dropout(
            self.observation_embedding_dropout
        )
        # Supervised representation regularizers are deliberately runtime
        # state, not checkpoint architecture. PPO enables exact-likelihood
        # mode so rollout and optimizer forwards are identical even if the
        # DAgger checkpoint used embedding/history dropout.
        self.exact_likelihood_mode = False
        self.null_history_embedding: nn.Parameter | None = None
        if (
            not self.unified_route_transformer
            and (
                self.history_condition_dropout_probability > 0.0
                or self.history_guidance_scale != 1.0
            )
        ):
            self.null_history_embedding = nn.Parameter(torch.zeros(
                1, 1, self.recurrent_dim
            ))
        self.speed_conditioner: nn.Module | None = None
        if self.speed_conditioning:
            self._create_speed_conditioner()
        self.topology_head: nn.Module | None = None
        self.topology_adapter: nn.Module | None = None
        if self.topology_target_dim:
            self._create_topology_modules()

    def set_exact_likelihood_mode(self, enabled: bool = True) -> None:
        """Disable stochastic supervised masks while retaining CFG guidance."""

        self.exact_likelihood_mode = bool(enabled)

    def _create_speed_conditioner(self) -> None:
        """Create a behavior-preserving command adapter.

        The final projection starts at zero, so enabling speed conditioning on
        an existing checkpoint leaves every action exactly unchanged until PPO
        learns to use the command.
        """

        conditioner = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.recurrent_dim),
        )
        nn.init.zeros_(conditioner[-1].weight)
        nn.init.zeros_(conditioner[-1].bias)
        self.speed_conditioner = conditioner

    def _create_shortcut_flow_action_head(self) -> None:
        if self.action_chunk_steps < 1:
            raise ValueError("shortcut flow requires a positive action horizon")
        if self.flow_context_mode == 'unified_route_dit':
            from starscream.unified_actor_flow import UnifiedActorFlow
            # The existing module now owns only the exact v5.5 tokenizer.
            # One shared DiT below owns ALL attention, including dynamics.
            self.step_embedding.blocks = nn.ModuleList()
            self.step_embedding.output_norm = nn.Identity()
            self.mean_network = nn.Identity()
            self.flow_action_head = UnifiedActorFlow(self.recurrent_dim,self.attention_heads,
                self.recurrent_depth,self.transformer_feedforward_dim,self.flow_source_noise,
                self.flow_deterministic_source)
            return
        structured = self.flow_context_mode == "structured_privileged_dit"
        self.flow_action_head = TokenizerFlowPolicy(
            observation_token_dim=self.recurrent_dim,
            action_token_dim=self.recurrent_dim,
            observation_tokens=(
                self.flow_route_gates + 4 if structured else self.context_steps
            ),
            visual_tokens=0,
            action_tokens=0,
            history_steps=(self.context_steps if structured else 1),
            action_horizon=self.action_chunk_steps,
            action_dim=4,
            d_model=self.recurrent_dim,
            n_heads=self.flow_heads,
            context_depth=(self.flow_context_depth if structured else 0),
            flow_depth=self.flow_depth,
            mlp_ratio=self.flow_mlp_ratio,
            k_max=8,
            source_mode="gaussian",
            source_noise=self.flow_source_noise,
            shortcut_step_sizes=self.flow_shortcut_step_sizes,
        )

    def enable_shortcut_flow_action_head(
        self,
        *,
        action_horizon: int,
        depth: int = 3,
        heads: int = 8,
        mlp_ratio: int = 4,
        sampling_steps: int = 3,
        sampling_method: str = "euler",
        source_noise: float = 0.25,
        shortcut_step_sizes: tuple[float, ...] | list[float] = (
            1.0 / 12.0, 1.0 / 6.0, 1.0 / 3.0,
        ),
    ) -> None:
        """Attach a joint CTBR-chunk flow decoder to a pretrained backbone."""

        if self.action_head_type == "shortcut_flow":
            if self.action_chunk_steps != int(action_horizon):
                raise ValueError("cannot change an existing flow action horizon")
            return
        if self.action_chunk_steps not in {1, int(action_horizon)}:
            raise ValueError("checkpoint action horizon is incompatible with flow upgrade")
        self.action_chunk_steps = int(action_horizon)
        self.action_head_type = "shortcut_flow"
        self.flow_depth = int(depth)
        self.flow_heads = int(heads)
        self.flow_mlp_ratio = int(mlp_ratio)
        self.flow_sampling_steps = int(sampling_steps)
        self.flow_sampling_method = str(sampling_method)
        self.flow_source_noise = float(source_noise)
        self.flow_shortcut_step_sizes = tuple(
            float(value) for value in shortcut_step_sizes
        )
        device = next(self.parameters()).device
        self._create_shortcut_flow_action_head()
        assert self.flow_action_head is not None
        self.flow_action_head.to(device)

    def _create_action_mixture_modules(
        self,
        centers: torch.Tensor,
        cluster_weights: torch.Tensor | None,
    ) -> None:
        """Create behavior-preserving joint-CTBR mode and residual heads.

        The existing direct-action logit is shared by every component.  Both
        new projections start at zero, so every candidate—and therefore the
        deployed argmax candidate—is exactly the incoming policy action before
        the first update.  A fixed codebook is used only to assign supervised
        joint-action modes; it is not an observation tokenizer.
        """

        if self.action_head_type != "mlp" or self.action_chunk_steps != 1:
            raise ValueError(
                "joint action mixtures require a single-step direct MLP head"
            )
        centers = torch.as_tensor(centers, dtype=torch.float32)
        if centers.ndim != 2 or centers.shape[1] != 4 or centers.shape[0] < 2:
            raise ValueError("action mixture centers must have shape (K,4), K>=2")
        if not torch.isfinite(centers).all() or torch.any(centers.abs() > 1.0001):
            raise ValueError("action mixture centers must be finite normalized CTBR")
        components = int(centers.shape[0])
        if cluster_weights is None:
            weights = torch.ones(components, dtype=torch.float32)
        else:
            weights = torch.as_tensor(cluster_weights, dtype=torch.float32)
            if weights.shape != (components,):
                raise ValueError("action mixture cluster weights must have shape (K,)")
            if not torch.isfinite(weights).all() or torch.any(weights <= 0.0):
                raise ValueError("action mixture cluster weights must be finite and positive")
            weights = weights / weights.mean()
        device = next(self.parameters()).device
        self.action_mixture_centers = centers.detach().clone().to(device)
        self.action_mixture_cluster_weights = weights.detach().clone().to(device)
        self.action_mixture_mode_head = nn.Linear(self.recurrent_dim, components)
        self.action_mixture_residual_head = nn.Linear(
            self.recurrent_dim, components * 4
        )
        # Exact behavior preservation matters for continuation experiments.
        nn.init.zeros_(self.action_mixture_mode_head.weight)
        nn.init.zeros_(self.action_mixture_mode_head.bias)
        nn.init.zeros_(self.action_mixture_residual_head.weight)
        nn.init.zeros_(self.action_mixture_residual_head.bias)
        self.action_mixture_mode_head.to(device)
        self.action_mixture_residual_head.to(device)

    def enable_action_mixture_head(
        self,
        centers: torch.Tensor | np.ndarray,
        cluster_weights: torch.Tensor | np.ndarray | None = None,
        *,
        residual_logit_scale: float = 2.0,
    ) -> None:
        """Attach the multimodal direct-action objective to a checkpoint."""

        residual_logit_scale = float(residual_logit_scale)
        if not np.isfinite(residual_logit_scale) or residual_logit_scale <= 0.0:
            raise ValueError("action mixture residual logit scale must be positive")
        incoming = torch.as_tensor(centers, dtype=torch.float32)
        if self.action_mixture_mode_head is not None:
            assert self.action_mixture_centers is not None
            if (
                tuple(incoming.shape) != tuple(self.action_mixture_centers.shape)
                or not torch.allclose(
                    incoming.cpu(), self.action_mixture_centers.detach().cpu()
                )
            ):
                raise ValueError("cannot replace an existing action mixture codebook")
            return
        self.action_mixture_residual_logit_scale = residual_logit_scale
        self._create_action_mixture_modules(
            incoming,
            None if cluster_weights is None else torch.as_tensor(cluster_weights),
        )

    @property
    def action_mixture_components(self) -> int:
        return (
            0 if self.action_mixture_centers is None
            else int(self.action_mixture_centers.shape[0])
        )

    def _action_mixture_outputs(
        self, control: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return deployed logits/action, mode logits, and all candidates."""

        if (
            self.action_mixture_mode_head is None
            or self.action_mixture_residual_head is None
            or self.action_mixture_centers is None
        ):
            raise RuntimeError("action mixture head is not enabled")
        base_logits = self._action_chunk_logits(control)[:, 0]
        mode_logits = self.action_mixture_mode_head(control)
        residual_logits = self.action_mixture_residual_logit_scale * torch.tanh(
            self.action_mixture_residual_head(control).reshape(
                len(control), self.action_mixture_components, 4
            )
        )
        candidate_logits = base_logits[:, None, :] + residual_logits
        candidates = torch.tanh(candidate_logits)
        selected_mode = mode_logits.argmax(dim=-1)
        batch = torch.arange(len(control), device=control.device)
        deployed_logits = candidate_logits[batch, selected_mode]
        deployed_action = candidates[batch, selected_mode]
        return deployed_logits, deployed_action, mode_logits, candidates

    def enable_speed_conditioning(
        self, *, scale: float = 20.0, default_command: float = 0.0,
    ) -> None:
        """Upgrade an unconditioned checkpoint without changing its policy."""

        if scale <= 0 or default_command < 0:
            raise ValueError("speed-conditioning scale must be positive and command non-negative")
        self.speed_conditioning = True
        self.speed_conditioning_scale = float(scale)
        self.default_speed_command = float(default_command)
        if self.speed_conditioner is None:
            device = next(self.parameters()).device
            self._create_speed_conditioner()
            assert self.speed_conditioner is not None
            self.speed_conditioner.to(device)

    def _create_topology_modules(self) -> None:
        """Create a route-plan readout and behavior-preserving control adapter.

        The plan predicts an MPCC trajectory in the active gate frame.  Its
        adapter is initialized to zero so an existing actor can be upgraded
        without changing even one control command before training.
        """

        if self.topology_target_dim <= 0:
            raise ValueError("topology target dimension must be positive")
        self.topology_head = nn.Sequential(
            nn.Linear(self.recurrent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.topology_target_dim),
        )
        # A newly attached plan head must begin as a neutral prediction. Large
        # pretrained encodings otherwise turn random projection weights into a
        # huge auxiliary loss before the head has learned its target units.
        nn.init.zeros_(self.topology_head[-1].weight)
        nn.init.zeros_(self.topology_head[-1].bias)
        self.topology_adapter = nn.Sequential(
            nn.LayerNorm(self.topology_target_dim),
            nn.Linear(self.topology_target_dim, self.recurrent_dim),
        )
        nn.init.zeros_(self.topology_adapter[-1].weight)
        nn.init.zeros_(self.topology_adapter[-1].bias)

    def _create_action_conditioned_dynamics_head(self) -> None:
        """Create a forward-model head conditioned on the proposed command.

        Unlike the legacy task-delta head, this module can pass a dynamics
        consistency gradient into the action mean. Its targets use a
        gate-invariant local frame, so gate-index changes do not invalidate the
        most important racing transitions.
        """

        if self.action_conditioned_dynamics_dim <= 0:
            raise ValueError("action-conditioned dynamics dimension must be positive")
        self.action_conditioned_dynamics_head = nn.Sequential(
            nn.Linear(self.recurrent_dim + 4, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.action_conditioned_dynamics_dim),
        )

    def enable_action_conditioned_dynamics(self, target_dim: int) -> None:
        """Upgrade an existing policy with a training-only forward model."""

        target_dim = int(target_dim)
        if target_dim <= 0:
            raise ValueError("action-conditioned dynamics dimension must be positive")
        if self.action_conditioned_dynamics_dim not in {0, target_dim}:
            raise ValueError(
                "cannot change action-conditioned dynamics dimension "
                f"from {self.action_conditioned_dynamics_dim} to {target_dim}"
            )
        if self.action_conditioned_dynamics_head is None:
            device = next(self.parameters()).device
            self.action_conditioned_dynamics_dim = target_dim
            self._create_action_conditioned_dynamics_head()
            assert self.action_conditioned_dynamics_head is not None
            self.action_conditioned_dynamics_head.to(device)

    def enable_topology_conditioning(self, target_dim: int) -> None:
        """Upgrade a checkpoint with a topology-plan head without policy drift."""

        target_dim = int(target_dim)
        if target_dim <= 0:
            raise ValueError("topology target dimension must be positive")
        if self.topology_target_dim not in {0, target_dim}:
            raise ValueError(
                "cannot change an existing topology target dimension "
                f"from {self.topology_target_dim} to {target_dim}"
            )
        if self.topology_head is None or self.topology_adapter is None:
            device = next(self.parameters()).device
            self.topology_target_dim = target_dim
            self._create_topology_modules()
            assert self.topology_head is not None
            assert self.topology_adapter is not None
            self.topology_head.to(device)
            self.topology_adapter.to(device)

    def _conditioned_encoding(
        self, encoding: torch.Tensor, speed_command: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.speed_conditioning:
            return encoding
        assert self.speed_conditioner is not None
        if speed_command is None:
            command = torch.full(
                (encoding.shape[0], 1), self.default_speed_command,
                device=encoding.device, dtype=encoding.dtype,
            )
        else:
            command = torch.as_tensor(
                speed_command, device=encoding.device, dtype=encoding.dtype,
            )
            if command.ndim == 0:
                command = command.expand(encoding.shape[0])
            if command.ndim == 1:
                command = command[:, None]
            if command.shape != (encoding.shape[0], 1):
                raise ValueError("speed command must be scalar or have shape (B,) or (B,1)")
        return encoding + self.speed_conditioner(command / self.speed_conditioning_scale)

    def _structured_flow_observations(
        self,
        normalized_features: torch.Tensor,
        speed_command: torch.Tensor | None,
    ) -> torch.Tensor:
        """Tokenize the privileged contract without flattening its semantics.

        Each causal step contributes one task token, one token per route gate,
        previous-command and timing tokens, and one speed-command token.  The
        resulting sequence is consumed directly by the flow policy's causal
        context Transformer; there is no separate recurrent policy backbone.
        """

        if self.flow_context_mode != "structured_privileged_dit":
            raise RuntimeError("structured flow observations are not enabled")
        if normalized_features.ndim == 2:
            normalized_features = normalized_features[:, None]
        if (
            normalized_features.ndim != 3
            or normalized_features.shape[-1] != self.input_dim
        ):
            raise ValueError(
                f"policy input must have shape (B,T,{self.input_dim})"
            )
        if normalized_features.shape[1] != self.context_steps:
            raise ValueError(
                "structured privileged DiT requires a fully padded causal "
                f"history of {self.context_steps} steps"
            )
        assert self.flow_task_projection is not None
        assert self.flow_route_projection is not None
        assert self.flow_previous_action_projection is not None
        assert self.flow_timing_projection is not None
        assert self.flow_speed_projection is not None
        assert self.flow_structured_slot is not None
        batch, steps, _ = normalized_features.shape
        route_end = TASK_DIM + self.flow_route_gates * ROUTE_RECORD_DIM
        task = self.flow_task_projection(normalized_features[..., :TASK_DIM])
        route = normalized_features[..., TASK_DIM:route_end].reshape(
            batch, steps, self.flow_route_gates, ROUTE_RECORD_DIM
        )
        route = self.flow_route_projection(route)
        previous = self.flow_previous_action_projection(
            normalized_features[..., route_end:route_end + 4]
        )
        timing = self.flow_timing_projection(
            normalized_features[..., route_end + 4:route_end + 6]
        )
        if speed_command is None:
            command = normalized_features.new_full(
                (batch,), self.default_speed_command
            )
        else:
            command = torch.as_tensor(
                speed_command,
                device=normalized_features.device,
                dtype=normalized_features.dtype,
            )
            if command.ndim == 0:
                command = command.expand(batch)
            if command.ndim == 2 and command.shape[-1] == 1:
                command = command[:, 0]
            if command.shape != (batch,):
                raise ValueError("speed command must be scalar or shaped [B]")
        speed = self.flow_speed_projection(
            (command / self.speed_conditioning_scale)[:, None]
        )
        tokens = torch.cat(
            (
                task[:, :, None], route, previous[:, :, None],
                timing[:, :, None], speed[:, None, None].expand(-1, steps, -1, -1),
            ),
            dim=2,
        )
        return tokens + self.flow_structured_slot

    def encode_tokens(self, normalized_features: torch.Tensor) -> torch.Tensor:
        if self.flow_context_mode == 'unified_route_dit':
            if normalized_features.ndim == 2:normalized_features=normalized_features[:,None]
            tokens,_,batch,steps=self.step_embedding._tokens(normalized_features[:,-self.context_steps:])
            hidden=self.flow_action_head.hidden(tokens)
            state=hidden[:,:2*steps].reshape(batch,steps,2,-1)[:,:,1]
            return torch.cat([state[:,:-1],hidden[:,-1:]],1)
        if self.flow_context_mode == "structured_privileged_dit":
            if self.flow_action_head is None:
                raise RuntimeError("structured privileged DiT flow head is disabled")
            observations = self._structured_flow_observations(
                normalized_features, None
            )
            empty_actions = observations.new_empty(
                observations.shape[0], self.context_steps, 0,
                self.recurrent_dim,
            )
            context = self.flow_action_head.encode_context(
                observations, empty_actions
            )
            return context.reshape(
                observations.shape[0], self.context_steps,
                self.flow_route_gates + 4, self.recurrent_dim,
            ).mean(dim=2)
        if normalized_features.ndim == 2:
            normalized_features = normalized_features[:, None]
        if normalized_features.ndim != 3 or normalized_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"policy input must have shape (B,T,{self.input_dim})"
            )
        if normalized_features.shape[1] > self.context_steps:
            normalized_features = normalized_features[:, -self.context_steps :]
        if self.unified_route_transformer:
            if not isinstance(self.step_embedding, UnifiedRouteTransformer):
                raise RuntimeError("unified route Transformer module is missing")
            history_mask: torch.Tensor | bool | None = None
            if (
                self.training
                and not self.exact_likelihood_mode
                and self.history_condition_dropout_probability > 0.0
                and normalized_features.shape[1] > 1
            ):
                history_mask = (
                    torch.rand(
                        normalized_features.shape[0],
                        device=normalized_features.device,
                    ) < self.history_condition_dropout_probability
                )
            embedding_dropout = (
                self.observation_embedding_dropout
                if self.training and not self.exact_likelihood_mode
                else 0.0
            )
            encoded = self.step_embedding(
                normalized_features,
                drop_old_history=history_mask,
                embedding_dropout=embedding_dropout,
            )
            if (
                (not self.training or self.exact_likelihood_mode)
                and self.history_guidance_scale != 1.0
                and normalized_features.shape[1] > 1
            ):
                unconditional = self.step_embedding(
                    normalized_features, drop_old_history=True,
                )
                encoded = unconditional + self.history_guidance_scale * (
                    encoded - unconditional
                )
            return encoded
        embedded = self.step_embedding(normalized_features)
        if not self.exact_likelihood_mode:
            embedded = self.observation_embedding_dropout_layer(embedded)
        if embedded.shape[1] > 1 and self.null_history_embedding is not None:
            if (
                self.training
                and not self.exact_likelihood_mode
                and self.history_condition_dropout_probability > 0.0
            ):
                drop = torch.rand(
                    embedded.shape[0], 1, 1, device=embedded.device
                ) < self.history_condition_dropout_probability
                null = self.null_history_embedding.expand(
                    embedded.shape[0], embedded.shape[1] - 1, -1
                )
                older = torch.where(drop, null, embedded[:, :-1])
                embedded = torch.cat([older, embedded[:, -1:]], dim=1)
        if self.encoder_type == "gru":
            encoded, _ = self.recurrent(embedded)
        elif self.encoder_type == "transformer":
            length = embedded.shape[1]
            raw_embedded = embedded
            if not self.temporal_alibi:
                embedded = embedded + self.temporal_position[:, -length:]
            if self.modern_transformer_blocks:
                encoded = self.recurrent(embedded)
                if (
                    (not self.training or self.exact_likelihood_mode)
                    and self.history_guidance_scale != 1.0
                    and length > 1
                    and self.null_history_embedding is not None
                ):
                    current_only = torch.cat([
                        self.null_history_embedding.expand(
                            raw_embedded.shape[0], length - 1, -1
                        ),
                        raw_embedded[:, -1:],
                    ], dim=1)
                    if not self.temporal_alibi:
                        current_only = (
                            current_only + self.temporal_position[:, -length:]
                        )
                    unconditional = self.recurrent(current_only)
                    encoded = unconditional + self.history_guidance_scale * (
                        encoded - unconditional
                    )
            else:
                causal_mask = torch.triu(
                    torch.ones(
                        length, length, device=embedded.device, dtype=torch.bool
                    ),
                    diagonal=1,
                )
                encoded = self.temporal_norm(
                    self.recurrent(embedded, mask=causal_mask, is_causal=True)
                )
        else:
            encoded = embedded
        return encoded

    def encode(self, normalized_features: torch.Tensor) -> torch.Tensor:
        return self.encode_tokens(normalized_features)[:, -1]

    def _flow_context(
        self,
        normalized_features: torch.Tensor,
        speed_command: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.flow_action_head is None:
            raise RuntimeError("shortcut-flow action head is not enabled")
        if self.flow_context_mode == 'unified_route_dit':
            if normalized_features.ndim == 2:normalized_features=normalized_features[:,None]
            tokens,_,_,_=self.step_embedding._tokens(normalized_features[:,-self.context_steps:])
            base=self.flow_action_head.hidden(tokens)[:,-1]
            # Same speed conditioning module/normalization as the direct actor.
            tokens=tokens+(self._conditioned_encoding(base,speed_command)-base)[:,None]
            return tokens,base,None
        if self.flow_context_mode == "structured_privileged_dit":
            observations = self._structured_flow_observations(
                normalized_features, speed_command
            )
            empty_actions = observations.new_empty(
                observations.shape[0], self.context_steps, 0,
                self.recurrent_dim,
            )
            context = self.flow_action_head.encode_context(
                observations, empty_actions
            )
            base = context.reshape(
                observations.shape[0], self.context_steps,
                self.flow_route_gates + 4, self.recurrent_dim,
            )[:, -1].mean(dim=1)
            return context, base, None
        tokens = self.encode_tokens(normalized_features)
        base = tokens[:, -1]
        conditioned = self._conditioned_encoding(base, speed_command)
        topology: torch.Tensor | None = None
        control = conditioned
        if self.topology_head is not None:
            assert self.topology_adapter is not None
            topology = self.topology_head(conditioned)
            control = conditioned + self.topology_adapter(topology)
        tokens = tokens + (control - base)[:, None]
        observation_tokens = tokens.unsqueeze(1)
        action_tokens = tokens.new_empty(
            tokens.shape[0], 1, 0, self.recurrent_dim
        )
        context = self.flow_action_head.encode_context(
            observation_tokens, action_tokens
        )
        return context, base, topology

    def flow_training_objective(
        self,
        normalized_features: torch.Tensor,
        target_chunk: torch.Tensor,
        speed_command: torch.Tensor | None = None,
        *,
        direct_weight: float = 0.5,
        bootstrap_weight: float = 1.0,
        stabilization_noise: float = 0.06,
        stabilization_remaining_time: float = 1.0 / 3.0,
        stabilization_enabled: bool = True,
        context: torch.Tensor | None = None,
        base: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Return per-sample shortcut loss, dynamics, metrics, and stabilization."""

        if self.flow_action_head is None:
            raise RuntimeError("shortcut-flow action head is not enabled")
        expected = (len(normalized_features), self.action_chunk_steps, 4)
        if target_chunk.shape != expected:
            raise ValueError(f"flow action target must have shape {expected}")
        if (context is None) != (base is None):
            raise ValueError("flow context and base must be supplied together")
        if context is None:
            context, base, _ = self._flow_context(
                normalized_features, speed_command
            )
        assert base is not None
        empty_observation = target_chunk.new_empty(
            len(target_chunk), 1, self.context_steps, self.recurrent_dim
        )
        empty_actions = target_chunk.new_empty(
            len(target_chunk), 1, 0, self.recurrent_dim
        )
        flow_per_sample, metrics = self.flow_action_head.shortcut_training_loss(
            target_chunk,
            empty_observation,
            empty_actions,
            direct_weight=direct_weight,
            bootstrap_weight=bootstrap_weight,
            context=context,
            reduction="none",
        )
        if stabilization_enabled:
            remaining = float(stabilization_remaining_time)
            if not 0.0 < remaining <= 1.0 or stabilization_noise < 0.0:
                raise ValueError("invalid shortcut-flow stabilization settings")
            perturbed = (
                target_chunk
                + float(stabilization_noise) * torch.randn_like(target_chunk)
            ).clamp(-1.0, 1.0)
            flow_time = target_chunk.new_full(
                (len(target_chunk),), 1.0 - remaining
            )
            step_size = target_chunk.new_full((len(target_chunk),), remaining)
            velocity = self.flow_action_head.velocity_from_context(
                perturbed, flow_time, step_size, context
            )
            correction = (target_chunk - perturbed) / remaining
            stabilization_per_sample = (
                velocity.float() - correction.float()
            ).square().mean(dim=(1, 2))
        else:
            stabilization_per_sample = target_chunk.new_zeros(
                len(target_chunk), dtype=torch.float32
            )
        dynamics = self.dynamics_head(base)
        return flow_per_sample, dynamics, metrics, stabilization_per_sample

    def flatten_backbone_parameters(self) -> None:
        """Compact GRU weights when available; Transformers need no equivalent."""

        flatten = getattr(self.recurrent, "flatten_parameters", None)
        if flatten is not None:
            flatten()

    def distribution(
        self, normalized_features: torch.Tensor,
        speed_command: torch.Tensor | None = None,
    ) -> SquashedGaussian:
        if self.action_head_type == "shortcut_flow":
            raise RuntimeError(
                "shortcut-flow policies do not define an exact Gaussian likelihood"
            )
        encoding = self._conditioned_encoding(
            self.encode(normalized_features), speed_command,
        )
        if self.topology_head is not None:
            assert self.topology_adapter is not None
            topology = self.topology_head(encoding)
            encoding = encoding + self.topology_adapter(topology)
        if self.action_mixture_mode_head is None:
            location = self._action_chunk_logits(encoding)[:, 0]
        else:
            location, _, _, _ = self._action_mixture_outputs(encoding)
        log_std = self.log_std_parameter.clamp(self.minimum_log_std, self.maximum_log_std)
        return SquashedGaussian(location, log_std)

    def forward(
        self, normalized_features: torch.Tensor,
        speed_command: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.action_head_type == "shortcut_flow":
            return self.predict_action_chunk(
                normalized_features, speed_command
            )[:, 0]
        return self.distribution(normalized_features, speed_command).mode()

    def predict_dynamics(
        self,
        normalized_features: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict legacy task deltas or invariant dynamics under ``action``."""

        base = self.encode(normalized_features)
        if self.action_conditioned_dynamics_head is None:
            return self.dynamics_head(base)
        if action is None:
            conditioned = self._conditioned_encoding(base, None)
            control = conditioned
            if self.topology_head is not None:
                assert self.topology_adapter is not None
                control = conditioned + self.topology_adapter(
                    self.topology_head(conditioned)
                )
            if self.action_mixture_mode_head is None:
                action = torch.tanh(self._action_chunk_logits(control)[:, 0])
            else:
                _, action, _, _ = self._action_mixture_outputs(control)
        if action.shape != (base.shape[0], 4):
            raise ValueError("action-conditioned dynamics requires actions shaped [B,4]")
        return self.action_conditioned_dynamics_head(
            torch.cat([base, action], dim=-1)
        )

    def predict_reward_components(
        self,
        normalized_features: torch.Tensor,
        executed_action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict dense RL reward components for an executed CTBR command."""

        if self.reward_aux_head is None:
            raise RuntimeError("reward auxiliary head is disabled")
        base = self.encode(normalized_features)
        if executed_action.shape != (base.shape[0], 4):
            raise ValueError("reward auxiliary actions must have shape [B,4]")
        return self.reward_aux_head(torch.cat([base, executed_action], dim=-1))

    def gate_contrastive_loss(
        self,
        normalized_features: torch.Tensor,
        *,
        temperature: float = 0.10,
        maximum_samples: int = 256,
    ) -> torch.Tensor:
        """InfoNCE alignment between policy state and the active route token."""

        if (
            self.gate_contrastive_query is None
            or self.gate_contrastive_target is None
            or not isinstance(
                self.step_embedding,
                (RouteSemanticStepTokenizer, UnifiedRouteTransformer),
            )
        ):
            raise RuntimeError("gate contrastive head is disabled")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("gate contrastive temperature must be positive")
        if maximum_samples < 2:
            raise ValueError("gate contrastive sample cap must be at least two")
        if normalized_features.ndim == 2:
            normalized_features = normalized_features[:, None]
        encoding = self.encode(normalized_features)
        return self.gate_contrastive_loss_from_encoding(
            encoding, normalized_features,
            temperature=temperature, maximum_samples=maximum_samples,
        )

    def gate_contrastive_loss_from_encoding(
        self,
        encoding: torch.Tensor,
        normalized_features: torch.Tensor,
        *,
        temperature: float = 0.10,
        maximum_samples: int = 256,
    ) -> torch.Tensor:
        """Compute gate InfoNCE while reusing an existing backbone forward."""

        if (
            self.gate_contrastive_query is None
            or self.gate_contrastive_target is None
            or not isinstance(
                self.step_embedding,
                (RouteSemanticStepTokenizer, UnifiedRouteTransformer),
            )
        ):
            raise RuntimeError("gate contrastive head is disabled")
        if normalized_features.ndim == 2:
            normalized_features = normalized_features[:, None]
        if encoding.shape != (len(normalized_features), self.recurrent_dim):
            raise ValueError("gate contrastive encoding is not batch aligned")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("gate contrastive temperature must be positive")
        if maximum_samples < 2:
            raise ValueError("gate contrastive sample cap must be at least two")
        if len(normalized_features) > maximum_samples:
            selected = torch.randperm(
                len(normalized_features), device=normalized_features.device
            )[:maximum_samples]
            normalized_features = normalized_features[selected]
            encoding = encoding[selected]
        query = F.normalize(
            self.gate_contrastive_query(encoding).float(), dim=-1
        )
        active_gate = self.step_embedding.active_gate_values(normalized_features)
        target = F.normalize(
            self.gate_contrastive_target(active_gate).float(), dim=-1
        )
        logits = query @ target.transpose(0, 1) / float(temperature)
        labels = torch.arange(len(query), device=query.device)
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.transpose(0, 1), labels)
        )

    def training_predictions(
        self,
        normalized_features: torch.Tensor,
        speed_command: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return action, dynamics, and topology outputs with one backbone pass."""

        action, dynamics, topology, _ = self.training_predictions_with_chunk(
            normalized_features, speed_command
        )
        return action, dynamics, topology

    def training_predictions_with_chunk(
        self,
        normalized_features: torch.Tensor,
        speed_command: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor
    ]:
        """Return current action, dynamics, topology, and the action horizon."""

        action, dynamics, topology, action_chunk, _ = (
            self.training_predictions_with_chunk_and_encoding(
                normalized_features, speed_command
            )
        )
        return action, dynamics, topology, action_chunk

    def training_predictions_with_chunk_and_encoding(
        self,
        normalized_features: torch.Tensor,
        speed_command: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor,
        torch.Tensor,
    ]:
        """Return supervised predictions and the shared causal encoding."""

        if self.action_head_type == "shortcut_flow":
            action_chunk = self.predict_action_chunk(
                normalized_features, speed_command
            )
            base = self.encode(normalized_features)
            dynamics = self.dynamics_head(base)
            return action_chunk[:, 0], dynamics, None, action_chunk, base
        base = self.encode(normalized_features)
        conditioned = self._conditioned_encoding(base, speed_command)
        topology: torch.Tensor | None = None
        control = conditioned
        if self.topology_head is not None:
            assert self.topology_adapter is not None
            topology = self.topology_head(conditioned)
            control = conditioned + self.topology_adapter(topology)
        if self.action_mixture_mode_head is None:
            action_chunk = torch.tanh(self._action_chunk_logits(control))
            action = action_chunk[:, 0]
        else:
            _, action, _, _ = self._action_mixture_outputs(control)
            action_chunk = action[:, None]
        if self.action_conditioned_dynamics_head is None:
            dynamics = self.dynamics_head(base)
        else:
            # Deliberately retain the action gradient. The expert-model target
            # describes the transition caused by its first planned command, so
            # this term penalizes commands whose physical consequence is wrong.
            dynamics = self.action_conditioned_dynamics_head(
                torch.cat([base, action], dim=-1)
            )
        return action, dynamics, topology, action_chunk, base

    def mixture_training_predictions(
        self,
        normalized_features: torch.Tensor,
        speed_command: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return deployed action, dynamics, topology, modes, and candidates.

        Supervised training assigns each expert command to a fixed joint CTBR
        mode, then regresses only that mode's candidate.  Deployment remains a
        single command selected by the categorical argmax.
        """

        if self.action_mixture_mode_head is None:
            raise RuntimeError("action mixture head is not enabled")
        base = self.encode(normalized_features)
        conditioned = self._conditioned_encoding(base, speed_command)
        topology: torch.Tensor | None = None
        control = conditioned
        if self.topology_head is not None:
            assert self.topology_adapter is not None
            topology = self.topology_head(conditioned)
            control = conditioned + self.topology_adapter(topology)
        _, deployed_action, mode_logits, candidates = (
            self._action_mixture_outputs(control)
        )
        if self.action_conditioned_dynamics_head is None:
            dynamics = self.dynamics_head(base)
        else:
            dynamics = self.action_conditioned_dynamics_head(
                torch.cat([base, deployed_action], dim=-1)
            )
        return deployed_action, dynamics, topology, mode_logits, candidates

    def predict_action_chunk(
        self,
        normalized_features: torch.Tensor,
        speed_command: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the normalized CTBR horizon used by receding-horizon BC.

        Only ``chunk[:, 0]`` is applied to the plant.  Keeping this method
        separate from ``distribution`` preserves the exact 4D PPO contract.
        """

        if self.action_head_type == "shortcut_flow":
            return self.sample_action_chunk(
                normalized_features, speed_command, deterministic=True
            )
        encoding = self._conditioned_encoding(
            self.encode(normalized_features), speed_command,
        )
        if self.topology_head is not None:
            assert self.topology_adapter is not None
            topology = self.topology_head(encoding)
            encoding = encoding + self.topology_adapter(topology)
        if self.action_mixture_mode_head is None:
            return torch.tanh(self._action_chunk_logits(encoding))
        _, action, _, _ = self._action_mixture_outputs(encoding)
        return action[:, None]

    def sample_action_chunk(
        self,
        normalized_features: torch.Tensor,
        speed_command: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
        source_noise: float | None = None,
    ) -> torch.Tensor:
        """Sample the shortcut-flow horizon under the deployment integrator.

        DAgger/evaluation use the deterministic zero-source endpoint. FPO++
        uses the same three Euler updates with the trained Gaussian source,
        while still executing only command zero in receding-horizon control.
        """

        if self.action_head_type != "shortcut_flow" or self.flow_action_head is None:
            raise RuntimeError("stochastic action chunks require a shortcut-flow policy")
        context, _, _ = self._flow_context(normalized_features, speed_command)
        batch = normalized_features.shape[0]
        observation_tokens = normalized_features.new_empty(
            batch, 1, self.context_steps, self.recurrent_dim
        )
        action_tokens = normalized_features.new_empty(
            batch, 1, 0, self.recurrent_dim
        )
        return self.flow_action_head.sample(
            observation_tokens,
            action_tokens,
            steps=self.flow_sampling_steps,
            method=self.flow_sampling_method,
            deterministic=deterministic,
            context=context,
            source_noise=source_noise,
        )

    def differentiable_action_chunk(
        self,
        normalized_features: torch.Tensor,
        speed_command: torch.Tensor | None = None,
        *,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the exact deterministic deployment endpoint with gradients."""

        if self.action_head_type != "shortcut_flow" or self.flow_action_head is None:
            raise RuntimeError("differentiable integration requires a shortcut-flow policy")
        if context is None:
            context, _, _ = self._flow_context(
                normalized_features, speed_command
            )
        batch = normalized_features.shape[0]
        observation_tokens = normalized_features.new_empty(
            batch, 1, self.context_steps, self.recurrent_dim
        )
        action_tokens = normalized_features.new_empty(
            batch, 1, 0, self.recurrent_dim
        )
        return self.flow_action_head.integrate(
            observation_tokens,
            action_tokens,
            steps=self.flow_sampling_steps,
            method=self.flow_sampling_method,
            deterministic=True,
            context=context,
        )

    def _action_chunk_logits(self, encoding: torch.Tensor) -> torch.Tensor:
        """Decode a dense action sequence jointly from one policy encoding."""

        if self.action_chunk_decoder is None:
            return self.mean_network(encoding).reshape(
                -1, self.action_chunk_steps, 4
            )
        assert self.action_chunk_queries is not None
        assert self.action_chunk_output is not None
        queries = self.action_chunk_queries.unsqueeze(0).expand(
            encoding.shape[0], -1, -1
        )
        decoded = self.action_chunk_decoder(
            queries, encoding.unsqueeze(1)
        )
        return self.action_chunk_output(decoded)


    def predict_topology(
        self,
        normalized_features: torch.Tensor,
        speed_command: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.topology_head is None:
            raise RuntimeError("topology conditioning is not enabled")
        encoding = self._conditioned_encoding(
            self.encode(normalized_features), speed_command,
        )
        return self.topology_head(encoding)

    def bind_feature_normalizer(self, normalizer: FeatureNormalizer, *, validate: bool = False) -> None:
        """Keep geometric derived features in physical units across train/eval/RL."""
        if self.route_feature_mode == 'none':
            return
        module = self.step_embedding
        for name, values in [('geometry_mean', normalizer.mean), ('geometry_std', normalizer.std)]:
            buffer = getattr(module, name)
            expected = torch.as_tensor(values, device=buffer.device, dtype=buffer.dtype)
            if validate and not torch.equal(buffer, expected):
                raise ValueError('checkpoint geometric statistics do not match its observation normalizer')
            with torch.no_grad():
                buffer.copy_(expected)

    def model_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "recurrent_dim": self.recurrent_dim,
            "recurrent_depth": self.recurrent_depth,
            "context_steps": self.context_steps,
            "encoder_type": self.encoder_type,
            "attention_heads": self.attention_heads,
            "transformer_feedforward_dim": self.transformer_feedforward_dim,
            "transformer_dropout": self.transformer_dropout,
            "observation_contract": self.observation_contract,
            "continuous_input_rms_norm": self.continuous_input_rms_norm,
            "unified_bidirectional": self.unified_bidirectional,
            "route_feature_mode": self.route_feature_mode,
            "route_head_modulation": self.route_head_modulation,
            "separate_route_encoder": self.separate_route_encoder,
            "structured_observation_tokens": self.structured_observation_tokens,
            "route_semantic_tokens": self.route_semantic_tokens,
            "unified_route_transformer": self.unified_route_transformer,
            "unified_readout_mode": self.unified_readout_mode,
            "unified_readout_decoder_depth": self.unified_readout_decoder_depth,
            "unified_readout_decoder_queries": self.unified_readout_decoder_queries,
            "unified_readout_initialization_std": (
                self.unified_readout_initialization_std
            ),
            "observation_token_layers": self.observation_token_layers,
            "observation_token_heads": self.observation_token_heads,
            "observation_embedding_dropout": self.observation_embedding_dropout,
            "continuous_projection_fourier_features": (
                self.continuous_projection_fourier_features
            ),
            "continuous_projection_fourier_scale": (
                self.continuous_projection_fourier_scale
            ),
            "modern_transformer_blocks": self.modern_transformer_blocks,
            "modern_adaln_zero": self.modern_adaln_zero,
            "temporal_alibi": self.temporal_alibi,
            "history_condition_dropout_probability": (
                self.history_condition_dropout_probability
            ),
            "history_guidance_scale": self.history_guidance_scale,
            "swiglu_action_head": self.swiglu_action_head,
            "output_initialization_scale": self.output_initialization_scale,
            "gate_contrastive_dim": self.gate_contrastive_dim,
            "initial_log_std": float(self.log_std_parameter.detach().mean()),
            "minimum_log_std": self.minimum_log_std,
            "maximum_log_std": self.maximum_log_std,
            "speed_conditioning": self.speed_conditioning,
            "speed_conditioning_scale": self.speed_conditioning_scale,
            "default_speed_command": self.default_speed_command,
            "topology_target_dim": self.topology_target_dim,
            "action_chunk_steps": self.action_chunk_steps,
            "action_chunk_decoder_layers": self.action_chunk_decoder_layers,
            "action_chunk_decoder_heads": self.action_chunk_decoder_heads,
            "action_chunk_decoder_feedforward_dim": (
                self.action_chunk_decoder_feedforward_dim
            ),
            "action_conditioned_dynamics_dim": (
                self.action_conditioned_dynamics_dim
            ),
            "reward_aux_dim": self.reward_aux_dim,
            "action_head_type": self.action_head_type,
            "action_mixture_centers": (
                None if self.action_mixture_centers is None
                else self.action_mixture_centers.detach().cpu().tolist()
            ),
            "action_mixture_cluster_weights": (
                None if self.action_mixture_cluster_weights is None
                else self.action_mixture_cluster_weights.detach().cpu().tolist()
            ),
            "action_mixture_residual_logit_scale": (
                self.action_mixture_residual_logit_scale
            ),
            "flow_depth": self.flow_depth,
            "flow_heads": self.flow_heads,
            "flow_mlp_ratio": self.flow_mlp_ratio,
            "flow_sampling_steps": self.flow_sampling_steps,
            "flow_sampling_method": self.flow_sampling_method,
            "flow_source_noise": self.flow_source_noise,
            "flow_deterministic_source": self.flow_deterministic_source,
            "flow_context_mode": self.flow_context_mode,
            "flow_context_depth": self.flow_context_depth,
            "flow_shortcut_step_sizes": list(self.flow_shortcut_step_sizes),
        }


def initialize_scratch_policy(
    model_config: Mapping[str, Any], device: str, *,
    context_steps: int | None = None,
) -> PrivilegedMLPPolicy:
    """Create matched scratch policies for a context-length ablation.

    The short model is always initialized first. A longer-context arm copies
    every shared parameter exactly and right-aligns the short temporal
    positions, while only newly exposed older positions retain fresh random
    initialization. No learned checkpoint state enters either arm.
    """

    source_config = dict(model_config)
    source_context = int(source_config.get("context_steps", 18))
    target_context = source_context if context_steps is None else int(context_steps)
    if target_context < source_context:
        raise ValueError(
            "scratch context may only be preserved or expanded; "
            f"source={source_context} requested={target_context}"
        )
    source = PrivilegedMLPPolicy(**source_config)
    if target_context == source_context:
        return source.to(device)
    if source.encoder_type != "transformer":
        raise ValueError("scratch context expansion is defined only for Transformers")
    target_config = dict(source_config)
    target_config["context_steps"] = target_context
    expanded = PrivilegedMLPPolicy(**target_config)
    state = dict(source.state_dict())
    temporal = expanded.temporal_position.detach().clone()
    temporal[:, -source_context:] = source.temporal_position.detach()
    state["temporal_position"] = temporal
    expanded.load_state_dict(state)
    return expanded.to(device)


class PrivilegedValue(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM + 4, hidden_dim: int = 384) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


class PrivilegedCourseValue(nn.Module):
    """Compact two-stream critic conditioned on a collision-free course ID.

    The state stream consumes the actor's normalized short history, which
    already contains truth state and six ordered future gates in the active
    recipe.  A separate task stream embeds rollout progress, global course
    geometry and the course's stable index in the active set.  FiLM-style
    fusion and a direct course residual make course-specific value offsets easy
    to learn without duplicating the actor's Transformer or attending over a
    second 24-gate token sequence.
    """

    def __init__(
        self, *, actor_input_dim: int, context_steps: int,
        episode_feature_dim: int = 4, course_feature_dim: int = 10,
        course_vocab_size: int = 512, hidden_dim: int = 256,
        course_embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        if (
            actor_input_dim < 1 or context_steps < 1
            or episode_feature_dim < 1 or course_feature_dim < 1
            or course_vocab_size < 2 or hidden_dim < 16
            or course_embedding_dim < 4
        ):
            raise ValueError("invalid privileged course critic dimensions")
        self.actor_input_dim = int(actor_input_dim)
        self.context_steps = int(context_steps)
        self.episode_feature_dim = int(episode_feature_dim)
        self.course_feature_dim = int(course_feature_dim)
        self.course_vocab_size = int(course_vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.course_embedding_dim = int(course_embedding_dim)
        self.state_dim = self.context_steps * self.actor_input_dim
        self.input_dim = (
            self.state_dim + self.episode_feature_dim
            + self.course_feature_dim + 1
        )

        self.state_encoder = nn.Sequential(
            nn.RMSNorm(self.state_dim),
            nn.Linear(self.state_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(
                self.episode_feature_dim + self.course_feature_dim,
                self.course_embedding_dim,
            ),
            nn.SiLU(),
        )
        self.course_embedding = nn.Embedding(
            self.course_vocab_size, self.course_embedding_dim,
        )
        self.modulation = nn.Linear(
            2 * self.course_embedding_dim, 2 * self.hidden_dim,
        )
        self.output = nn.Sequential(
            nn.RMSNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.course_value = nn.Linear(
            self.course_embedding_dim, 1, bias=False,
        )
        nn.init.uniform_(self.output[-1].weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.output[-1].bias)
        nn.init.zeros_(self.course_value.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or value.shape[-1] != self.input_dim:
            raise ValueError(
                "privileged course critic expected "
                f"[B,{self.input_dim}], got {tuple(value.shape)}"
            )
        state = value[:, :self.state_dim]
        context_end = self.input_dim - 1
        context = value[:, self.state_dim:context_end]
        raw_course_id = value[:, -1]
        course_id = raw_course_id.round().long()
        # The collection boundary validates IDs before storage.  Keep the
        # friendlier checks on CPU tests/debugging, but never introduce a host
        # synchronization inside CUDA graph capture or every critic minibatch.
        if value.device.type != "cuda":
            if not bool(torch.allclose(
                raw_course_id, course_id.to(raw_course_id.dtype),
            )):
                raise ValueError("privileged course critic IDs must be integers")
            if bool(torch.any(
                (course_id < 0) | (course_id >= self.course_vocab_size)
            )):
                raise ValueError("privileged course critic ID exceeds its vocabulary")
        course_id = course_id.clamp(0, self.course_vocab_size - 1)
        embedded_course = self.course_embedding(course_id)
        task = torch.cat([
            self.context_encoder(context), embedded_course,
        ], dim=-1)
        scale, shift = self.modulation(task).chunk(2, dim=-1)
        hidden = self.state_encoder(state)
        hidden = hidden * (1.0 + 0.1 * torch.tanh(scale)) + shift
        return (
            self.output(hidden).squeeze(-1)
            + self.course_value(embedded_course).squeeze(-1)
        )


class PrivilegedTransformerValue(nn.Module):
    """Asymmetric PPO value model with explicit temporal and course tokens.

    The flat transport tensor is only a storage format.  It is decoded into a
    short actor-state history, episode/course summaries, a longer privileged
    route preview, and a discrete course identity.  None of these additional
    course features are exposed to the actor.
    """

    def __init__(
        self, *, actor_input_dim: int, context_steps: int,
        route_tokens: int = 24, route_feature_dim: int = 14,
        episode_feature_dim: int = 4, course_feature_dim: int = 10,
        course_vocab_size: int = 4096, model_dim: int = 320,
        depth: int = 4, heads: int = 8, feedforward_dim: int = 640,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if (
            actor_input_dim < 1 or context_steps < 1 or route_tokens < 1
            or route_feature_dim < 1 or episode_feature_dim < 1
            or course_feature_dim < 1 or course_vocab_size < 2
            or model_dim < 16 or depth < 1 or heads < 1
            or model_dim % heads
        ):
            raise ValueError("invalid privileged Transformer critic dimensions")
        self.actor_input_dim = int(actor_input_dim)
        self.context_steps = int(context_steps)
        self.route_tokens = int(route_tokens)
        self.route_feature_dim = int(route_feature_dim)
        self.episode_feature_dim = int(episode_feature_dim)
        self.course_feature_dim = int(course_feature_dim)
        self.course_vocab_size = int(course_vocab_size)
        self.model_dim = int(model_dim)
        self.input_dim = (
            self.context_steps * self.actor_input_dim
            + self.episode_feature_dim + self.course_feature_dim
            + self.route_tokens * self.route_feature_dim + 1
        )

        self.state_projection = nn.Sequential(
            nn.RMSNorm(self.actor_input_dim),
            nn.Linear(self.actor_input_dim, self.model_dim),
        )
        self.episode_projection = nn.Linear(self.episode_feature_dim, self.model_dim)
        self.course_projection = nn.Linear(self.course_feature_dim, self.model_dim)
        self.route_projection = nn.Sequential(
            nn.RMSNorm(self.route_feature_dim),
            nn.Linear(self.route_feature_dim, self.model_dim),
        )
        self.course_embedding = nn.Embedding(self.course_vocab_size, self.model_dim)
        self.readout = nn.Parameter(torch.empty(1, 1, self.model_dim))
        self.history_position = nn.Parameter(torch.empty(
            1, self.context_steps, self.model_dim
        ))
        self.route_position = nn.Parameter(torch.empty(
            1, self.route_tokens, self.model_dim
        ))
        # readout, episode, global course, history, route, discrete course id
        self.token_type = nn.Parameter(torch.empty(1, 6, self.model_dim))
        for parameter in (
            self.readout, self.history_position, self.route_position,
            self.token_type,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(heads),
            dim_feedforward=int(feedforward_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(depth), enable_nested_tensor=False,
        )
        self.output = nn.Sequential(
            nn.RMSNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, 1),
        )
        nn.init.uniform_(self.output[-1].weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or value.shape[-1] != self.input_dim:
            raise ValueError(
                "privileged Transformer critic expected "
                f"[B,{self.input_dim}], got {tuple(value.shape)}"
            )
        batch = value.shape[0]
        cursor = 0
        history_width = self.context_steps * self.actor_input_dim
        history = value[:, cursor:cursor + history_width].reshape(
            batch, self.context_steps, self.actor_input_dim
        )
        cursor += history_width
        episode = value[:, cursor:cursor + self.episode_feature_dim]
        cursor += self.episode_feature_dim
        course = value[:, cursor:cursor + self.course_feature_dim]
        cursor += self.course_feature_dim
        route_width = self.route_tokens * self.route_feature_dim
        route = value[:, cursor:cursor + route_width].reshape(
            batch, self.route_tokens, self.route_feature_dim
        )
        cursor += route_width
        course_id = value[:, cursor].round().long().clamp(
            0, self.course_vocab_size - 1
        )

        readout = self.readout.expand(batch, -1, -1) + self.token_type[:, 0:1]
        episode_token = self.episode_projection(episode)[:, None] + self.token_type[:, 1:2]
        course_token = self.course_projection(course)[:, None] + self.token_type[:, 2:3]
        history_tokens = (
            self.state_projection(history) + self.history_position
            + self.token_type[:, 3:4]
        )
        route_tokens = (
            self.route_projection(route) + self.route_position
            + self.token_type[:, 4:5]
        )
        identity_token = (
            self.course_embedding(course_id)[:, None] + self.token_type[:, 5:6]
        )
        tokens = torch.cat([
            readout, episode_token, course_token, history_tokens,
            route_tokens, identity_token,
        ], dim=1)
        return self.output(self.encoder(tokens)[:, 0]).squeeze(-1)


@dataclass
class OneStepData:
    features: np.ndarray
    actions: np.ndarray
    previous_actions: np.ndarray
    next_task_deltas: np.ndarray
    episode_ids: np.ndarray
    paths: tuple[str, ...]
    action_chunks: np.ndarray | None = None
    action_chunk_offsets: tuple[int, ...] = ()
    states: np.ndarray | None = None
    next_states: np.ndarray | None = None
    track_ids: np.ndarray | None = None
    track_names: tuple[str, ...] = ()
    dynamics_valid: np.ndarray | None = None
    observation_contract: str = LEGACY_OBSERVATION_CONTRACT

    def __post_init__(self) -> None:
        count = len(self.features)
        if (
            self.features.ndim != 2
            or self.features.shape[0] != count
            or self.actions.shape != (count, 4)
            or self.previous_actions.shape != (count, 4)
            or self.next_task_deltas.shape != (count, TASK_DIM)
            or self.episode_ids.shape != (count,)
        ):
            raise ValueError("one-step dataset arrays are not aligned")
        if self.observation_contract == LEGACY_OBSERVATION_CONTRACT:
            route_gates_from_feature_dim(self.features.shape[1])
        elif self.observation_contract == GREEN2026_OBSERVATION_CONTRACT:
            if self.features.shape[1] != GREEN2026_FEATURE_DIM:
                raise ValueError("one-step feature width disagrees with its contract")
        elif self.observation_contract == GREEN2026_ROUTE_CHAIN_OBSERVATION_CONTRACT:
            payload = self.features.shape[1] - 16
            if payload < 12 or payload % 12:
                raise ValueError("one-step feature width disagrees with its contract")
        else:
            raise ValueError("one-step feature width disagrees with its contract")
        if self.track_ids is None:
            self.track_ids = np.zeros(count, np.int16)
        if self.track_ids.shape != (count,):
            raise ValueError("one-step track IDs are not aligned")
        if not self.track_names:
            self.track_names = ("unknown",)
        if self.dynamics_valid is None:
            self.dynamics_valid = np.ones(count, np.bool_)
        if self.dynamics_valid.shape != (count,):
            raise ValueError("one-step dynamics-valid mask is not aligned")
        if self.action_chunks is None:
            self.action_chunks = np.empty((count, 0, 4), np.float32)
        if self.action_chunks.shape != (count, len(self.action_chunk_offsets), 4):
            raise ValueError("future action chunks are not aligned")
        if self.states is None:
            self.states = np.empty((count, 0), np.float32)
        if self.next_states is None:
            self.next_states = np.empty((count, 0), np.float32)
        if self.states.shape != self.next_states.shape or self.states.shape[0] != count:
            raise ValueError("world-state transitions are not aligned")


def _decode_track(value: Any) -> str:
    raw = np.asarray(value)[()]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def load_one_step_data(
    root: str | Path,
    *,
    track: str | Sequence[str],
    require_controller_valid: bool = True,
    maximum_steps: int = 0,
    route_gates: int = ROUTE_GATES,
    action_chunk_offsets: Sequence[int] = (),
    observation_contract: str = LEGACY_OBSERVATION_CONTRACT,
) -> OneStepData:
    """Load compact one-step features into RAM to remove HDF5 training stalls."""

    requested_tracks = (track,) if isinstance(track, str) else tuple(track)
    if not requested_tracks:
        raise ValueError("at least one track is required")
    # Procedural training uses materialized YAML paths while episode metadata
    # stores each track's immutable internal name.  Resolve that alias here so
    # the rest of the loader and balanced sampler retain their old contract.
    canonical_tracks: list[str] = []
    track_geometry: dict[str, Any] = {}
    for requested in requested_tracks:
        candidate = Path(str(requested))
        if candidate.exists() and candidate.suffix.lower() in {".yaml", ".yml"}:
            from .env.tracks import load_track
            loaded_track = load_track(candidate)
            canonical_tracks.append(loaded_track.name)
            track_geometry[loaded_track.name] = loaded_track
        else:
            canonical_tracks.append(str(requested))
    if len(canonical_tracks) != len(set(canonical_tracks)):
        raise ValueError("requested tracks resolve to duplicate internal names")
    track_to_id = {name: index for index, name in enumerate(canonical_tracks)}
    chunk_offsets = tuple(int(item) for item in action_chunk_offsets)
    if (
        any(item < 0 for item in chunk_offsets)
        or len(set(chunk_offsets)) != len(chunk_offsets)
        or tuple(sorted(chunk_offsets)) != chunk_offsets
    ):
        raise ValueError(
            "action_chunk_offsets must be unique increasing non-negative steps"
        )
    paths = sorted(Path(root).rglob("*.h5"))
    feature_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    action_chunk_parts: list[np.ndarray] = []
    previous_parts: list[np.ndarray] = []
    next_delta_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    next_state_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    track_parts: list[np.ndarray] = []
    dynamics_valid_parts: list[np.ndarray] = []
    accepted_paths: list[str] = []
    total = 0
    for path in paths:
        with h5py.File(path, "r", swmr=True) as archive:
            archive_track = (
                _decode_track(archive["metadata/track"])
                if "metadata/track" in archive else ""
            )
            if archive_track not in track_to_id:
                continue
            length = int(archive["action/normalized"].shape[0])
            task = np.asarray(archive["observation/task_state"][:length], np.float32)
            next_task = np.asarray(
                archive["observation/task_state"][1 : length + 1], np.float32
            )
            states = np.asarray(archive["observation/state"][:length], np.float32)
            next_states = np.asarray(
                archive["observation/state"][1 : length + 1], np.float32
            )
            if (
                states.ndim != 2 or next_states.shape != states.shape
                or states.shape[0] != length or states.shape[1] < 13
            ):
                raise ValueError(f"incompatible world-state transitions in {path}")
            route_records = np.asarray(
                archive["observation/flight_plan/records"][:length], np.float32
            )
            if route_records.ndim != 3 or route_records.shape[2] != ROUTE_RECORD_DIM:
                raise ValueError(f"incompatible flight-plan records in {path}")
            requested_route_gates = int(route_gates)
            if requested_route_gates < 1:
                raise ValueError("route_gates must be positive")
            if route_records.shape[1] < requested_route_gates:
                geometry = track_geometry.get(archive_track)
                if (
                    geometry is not None
                    and "observation/flight_plan/index" in archive
                    and "transition/gate_passed" in archive
                ):
                    active_indices = np.asarray(
                        archive["observation/flight_plan/index"][:length, 0],
                        np.int64,
                    )
                    gate_passed = np.asarray(
                        archive["transition/gate_passed"][:length], np.bool_
                    )
                    passed_before = np.concatenate([
                        np.zeros(1, np.int64),
                        np.cumsum(gate_passed[:-1], dtype=np.int64),
                    ])
                    route_records = np.stack([
                        geometry.flight_plan(
                            int(active), requested_route_gates,
                            remaining=max(len(geometry.gates) - int(passed), 1),
                        )["records"]
                        for active, passed in zip(active_indices, passed_before)
                    ]).astype(np.float32)
                else:
                    padding = np.repeat(
                        route_records[:, -1:, :],
                        requested_route_gates - route_records.shape[1],
                        axis=1,
                    )
                    route_records = np.concatenate(
                        [route_records, padding], axis=1
                    )
            else:
                route_records = route_records[:, :requested_route_gates]
            route = route_records.reshape(length, -1)
            previous_ctbr = np.asarray(
                archive["observation/previous_action"][:length], np.float32
            )
            previous = np.stack([ctbr_to_normalized(item) for item in previous_ctbr])
            age = np.asarray(
                archive["observation/age/previous_action"][:length], np.float32
            ).reshape(length, 1)
            valid_previous = np.asarray(
                archive["observation/valid/previous_action"][:length], np.float32
            ).reshape(length, 1)
            if observation_contract == GREEN2026_OBSERVATION_CONTRACT:
                required = (
                    "observation/gates/position",
                    "observation/gates/normal",
                    "observation/gates/up",
                    "observation/gates/size",
                )
                missing = [name for name in required if name not in archive]
                if missing:
                    raise ValueError(
                        f"Green-2026 offline adapter is missing {missing} in {path}"
                    )
                features = green2026_batch_features(
                    states=states,
                    gate_positions_body=np.asarray(
                        archive[required[0]][:length], np.float32
                    ),
                    gate_normals_body=np.asarray(
                        archive[required[1]][:length], np.float32
                    ),
                    gate_up_body=np.asarray(
                        archive[required[2]][:length], np.float32
                    ),
                    gate_sizes=np.asarray(
                        archive[required[3]][:length], np.float32
                    ),
                    previous_actions_normalized=previous,
                )
            elif observation_contract == GREEN2026_ROUTE_CHAIN_OBSERVATION_CONTRACT:
                required = (
                    "observation/gates/position",
                    "observation/gates/normal",
                    "observation/gates/up",
                    "observation/gates/size",
                )
                missing = [name for name in required if name not in archive]
                if missing:
                    raise ValueError(
                        f"Green route-chain offline adapter is missing {missing} in {path}"
                    )
                gate_arrays = [
                    np.asarray(archive[name][:length], np.float32)
                    for name in required
                ]
                if any(value.shape[1] < requested_route_gates for value in gate_arrays):
                    # Older expert archives commonly stored only three relative
                    # gates. The already reconstructed route records contain the
                    # same body-frame geometry and preserve terminal repetition.
                    gate_arrays = [
                        route_records[..., 0:3],
                        route_records[..., 3:6],
                        route_records[..., 6:9],
                        route_records[..., 9:11],
                    ]
                features = green2026_route_chain_batch_features(
                    states=states,
                    gate_positions_body=gate_arrays[0],
                    gate_normals_body=gate_arrays[1],
                    gate_up_body=gate_arrays[2],
                    gate_sizes=gate_arrays[3],
                    previous_actions_normalized=previous,
                    route_gates=requested_route_gates,
                )
            elif observation_contract == LEGACY_OBSERVATION_CONTRACT:
                features = np.concatenate(
                    [task, route, previous, age, valid_previous], axis=1
                )
            else:
                raise ValueError(
                    f"unknown privileged observation contract {observation_contract!r}"
                )
            actions = np.asarray(archive["action/normalized"][:length], np.float32)
            dynamics_valid = ~np.asarray(
                archive["transition/gate_passed"][:length], np.bool_
            )
            legal = (
                np.all(np.isfinite(features), axis=1)
                & np.all(np.isfinite(actions), axis=1)
                & np.all(np.isfinite(states), axis=1)
                & np.all(np.isfinite(next_states), axis=1)
            )
            if require_controller_valid and "controller/valid" in archive:
                legal &= np.asarray(archive["controller/valid"][:length], bool)
            selected = np.flatnonzero(legal)
            if maximum_steps > 0:
                remaining = maximum_steps - total
                if remaining <= 0:
                    break
                selected = selected[:remaining]
            if not len(selected):
                continue
            # An invalid teacher step breaks causal history even inside one HDF5
            # episode.  Treat each contiguous legal run as its own sequence.
            runs = np.split(selected, np.flatnonzero(np.diff(selected) > 1) + 1)
            for run in runs:
                if not len(run):
                    continue
                sequence_id = len(episode_parts)
                feature_parts.append(features[run])
                action_parts.append(actions[run])
                if chunk_offsets:
                    positions = np.arange(len(run), dtype=np.int64)
                    future = np.stack([
                        np.minimum(positions + offset, len(run) - 1)
                        for offset in chunk_offsets
                    ], axis=1)
                    action_chunk_parts.append(actions[run[future]])
                else:
                    action_chunk_parts.append(
                        np.empty((len(run), 0, 4), np.float32)
                    )
                previous_parts.append(previous[run])
                next_delta_parts.append((next_task - task)[run])
                state_parts.append(states[run])
                next_state_parts.append(next_states[run])
                dynamics_valid_parts.append(dynamics_valid[run])
                episode_parts.append(np.full(len(run), sequence_id, np.int64))
                track_parts.append(np.full(len(run), track_to_id[archive_track], np.int16))
            accepted_paths.append(str(path))
            total += len(selected)
    if not feature_parts:
        raise ValueError(f"no valid one-step data for tracks {requested_tracks!r} under {root}")
    return OneStepData(
        features=np.concatenate(feature_parts).astype(np.float32),
        actions=np.concatenate(action_parts).clip(-1.0, 1.0).astype(np.float32),
        action_chunks=np.concatenate(action_chunk_parts).clip(-1.0, 1.0).astype(np.float32),
        action_chunk_offsets=chunk_offsets,
        previous_actions=np.concatenate(previous_parts).astype(np.float32),
        next_task_deltas=np.concatenate(next_delta_parts).astype(np.float32),
        states=np.concatenate(state_parts).astype(np.float32),
        next_states=np.concatenate(next_state_parts).astype(np.float32),
        episode_ids=np.concatenate(episode_parts),
        paths=tuple(accepted_paths),
        track_ids=np.concatenate(track_parts),
        track_names=tuple(canonical_tracks),
        dynamics_valid=np.concatenate(dynamics_valid_parts),
        observation_contract=str(observation_contract),
    )


def sequence_indices(
    episode_ids: np.ndarray, indices: np.ndarray, context_steps: int
) -> np.ndarray:
    """Build repeat-first padded causal history indices without crossing episodes."""

    episode_ids = np.asarray(episode_ids, np.int64)
    indices = np.asarray(indices, np.int64)
    starts = np.empty(len(episode_ids), np.int64)
    start = 0
    for index in range(len(episode_ids)):
        if index == 0 or episode_ids[index] != episode_ids[index - 1]:
            start = index
        starts[index] = start
    offsets = np.arange(1 - int(context_steps), 1, dtype=np.int64)
    return np.maximum(indices[:, None] + offsets[None], starts[indices, None])


def episode_split(
    data: OneStepData, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    validation_episodes: set[int] = set()
    for track_id in np.unique(data.track_ids):
        track_samples = np.flatnonzero(data.track_ids == track_id)
        episodes = np.unique(data.episode_ids[track_samples])
        rng.shuffle(episodes)
        validation_count = max(1, int(round(len(episodes) * validation_fraction)))
        validation_episodes.update(int(item) for item in episodes[:validation_count])
    validation = np.asarray(
        [index for index, item in enumerate(data.episode_ids) if int(item) in validation_episodes],
        np.int64,
    )
    train = np.asarray(
        [index for index, item in enumerate(data.episode_ids) if int(item) not in validation_episodes],
        np.int64,
    )
    if not len(train) or not len(validation):
        raise ValueError("episode split requires at least two episodes")
    return train, validation


def resolve_ranked_checkpoint(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    manifest = candidate / "top-k.json"
    if manifest.is_file():
        import json

        entries = json.loads(manifest.read_text(encoding="utf-8"))["checkpoints"]
        if entries:
            return candidate / entries[0]["path"]
    latest = candidate / "latest.pt"
    if latest.is_file():
        return latest
    raise FileNotFoundError(f"no checkpoint found under {candidate}")


def load_policy_checkpoint(
    path: str | Path, device: str = "cpu", *, context_steps: int | None = None,
) -> tuple[PrivilegedMLPPolicy, FeatureNormalizer, dict[str, Any], Path]:
    resolved = resolve_ranked_checkpoint(path)
    state = torch.load(resolved, map_location=device, weights_only=False)
    model_config = dict(state["model_config"])
    source_context = int(model_config.get("context_steps", 18))
    target_context = source_context if context_steps is None else int(context_steps)
    if target_context < source_context:
        raise ValueError(
            "checkpoint context may only be preserved or expanded; "
            f"source={source_context} requested={target_context}"
        )
    model_config["context_steps"] = target_context
    policy = PrivilegedMLPPolicy(**model_config)
    model_state = dict(state["model"])
    if target_context != source_context:
        if policy.encoder_type != "transformer":
            raise ValueError("context expansion is currently defined only for Transformers")
        source_position = torch.as_tensor(model_state["temporal_position"])
        if source_position.shape[1] != source_context:
            raise ValueError("checkpoint temporal position length disagrees with model config")
        expanded = policy.temporal_position.detach().clone()
        # The encoder right-aligns every partial history. Preserve all learned
        # embeddings for the recent source window exactly. The newly exposed
        # older positions retain the model's small random initialization.
        expanded[:, -source_context:] = source_position
        model_state["temporal_position"] = expanded
    policy.load_state_dict(model_state)
    policy.to(device)
    normalizer = FeatureNormalizer.from_state_dict(state["normalizer"])
    policy.bind_feature_normalizer(normalizer, validate=True)
    return policy, normalizer, state, resolved


def checkpoint_payload(
    policy: PrivilegedMLPPolicy,
    normalizer: FeatureNormalizer,
    *,
    stage: str,
    track: str,
    optimizer: torch.optim.Optimizer | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "contract": "privileged-causal-one-action-ctbr-v1",
        "stage": stage,
        "track": track,
        "model": policy.state_dict(),
        "model_config": policy.model_config(),
        "normalizer": normalizer.state_dict(),
        "feature_dim": policy.input_dim,
        "action_horizon": 1,
        "control_hz": 90,
    }
    if optimizer is not None:
        result["optimizer"] = optimizer.state_dict()
    if policy.plant_settings_dim:
        from starscream.plant_privileged import PLANT_SETTINGS_NAMES, PLANT_STATIC_SCALE
        result['plant_settings_schema'] = dict(version=PLANT_OBSERVATION_CONTRACT,
            names=list(PLANT_SETTINGS_NAMES), static_scale=PLANT_STATIC_SCALE.tolist(),
            history_storage='normalized_physical_suffix', token_source='latest_history_row')
    if extra:
        result.update(dict(extra))
        if 'control_hz' not in extra:
            training = extra.get('training_config', {})
            settings = training.get(stage, training) if isinstance(training, Mapping) else {}
            frequency = float(settings.get('control_hz', 90))
            if not np.isfinite(frequency) or frequency <= 0:
                raise ValueError('checkpoint control frequency must be finite and positive')
            result['control_hz'] = frequency
    return result


def flatten_metrics(results: Iterable[Mapping[str, Any]], gate_count: int) -> dict[str, float]:
    values = list(results)
    if not values:
        return {}
    gates = np.asarray([item["gates"] for item in values], np.float32)
    metrics = {
        "episodes": float(len(values)),
        "mean_gates": float(gates.mean()),
        "crash_rate": float(np.mean([item["crashed"] for item in values])),
        "mean_return": float(np.mean([item["return"] for item in values])),
        "mean_steps": float(np.mean([item["steps"] for item in values])),
    }
    targets = np.asarray([item.get("target_gates", gate_count) for item in values], np.float32)
    # Preserve the complete survival curve. p1/p2/p3 remain the compact
    # selection metrics, while later pN values expose where long courses fail.
    for count in range(1, gate_count + 1):
        metrics[f"p{count}"] = float(np.mean(gates >= count))
    # A final-gate crossing and collision can occur in the same simulator tick.
    # Reaching the count is necessary, but a crashed episode is not a completion.
    passed = (gates >= targets) & ~np.asarray([bool(item["crashed"]) for item in values])
    metrics["full_course_success"] = float(np.mean(passed))
    metrics["successful_episodes"] = float(passed.sum())
    successful_speeds = [
        float(item["mean_speed_mps"])
        for item, success in zip(values, passed)
        if success and "mean_speed_mps" in item
        and np.isfinite(float(item["mean_speed_mps"]))
    ]
    metrics["successful_mean_speed_mps"] = (
        float(np.mean(successful_speeds)) if successful_speeds else float("nan")
    )
    successful_steps = np.asarray([
        item["steps"] for item, success in zip(values, passed) if success
    ], np.float32)
    metrics["successful_mean_steps"] = (
        float(successful_steps.mean()) if len(successful_steps) else float("nan")
    )
    metrics["successful_minimum_steps"] = (
        float(successful_steps.min()) if len(successful_steps) else float("nan")
    )
    metrics["successful_median_steps"] = (
        float(np.median(successful_steps)) if len(successful_steps) else float("nan")
    )
    metrics["successful_p10_steps"] = (
        float(np.quantile(successful_steps, 0.10))
        if len(successful_steps) else float("nan")
    )
    metrics["successful_p90_steps"] = (
        float(np.quantile(successful_steps, 0.90))
        if len(successful_steps) else float("nan")
    )
    metrics["performance_weighted_success_per_step"] = (
        metrics["full_course_success"] / metrics["successful_minimum_steps"]
        if len(successful_steps) else 0.0
    )
    # Legacy 90-Hz reference score. Keep the unit-bearing value beside
    # the frequency-neutral score so a non-90-Hz experiment cannot be confused
    # with the paper's success / fastest-lap-time metric.
    metrics["performance_weighted_success_hz90"] = (
        90.0 * metrics["performance_weighted_success_per_step"]
    )
    metrics["selection_score"] = (
        4.0 * metrics["full_course_success"]
        + 2.0 * metrics.get("p3", 0.0)
        + metrics.get("p2", 0.0)
        + 0.25 * metrics.get("p1", 0.0)
        - 0.5 * metrics["crash_rate"]
    )
    standard = {"gates", "crashed", "return", "steps", "track", "target_gates"}
    extra_keys = sorted(set().union(*(item.keys() for item in values)) - standard)
    for key in extra_keys:
        samples = []
        for item in values:
            try:
                sample = float(item[key])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(sample):
                samples.append(sample)
        if samples:
            metrics[str(key)] = float(np.mean(samples))
            if key in {
                "mean_speed_mps", "mean_gate_speed_mps", "maximum_speed_mps",
            }:
                metrics[f"episode_{key}_p10"] = float(np.quantile(samples, 0.10))
                metrics[f"episode_{key}_median"] = float(np.median(samples))
                metrics[f"episode_{key}_p90"] = float(np.quantile(samples, 0.90))
    lap_counts = sorted({
        int(item["rollout_laps"])
        for item in values if item.get("rollout_laps") is not None
    })
    for laps in lap_counts:
        subset = [
            item for item in values
            if int(item.get("rollout_laps", -1)) == laps
        ]
        subset_targets = np.asarray([
            item.get("target_gates", gate_count) for item in subset
        ], np.float32)
        subset_gates = np.asarray([item["gates"] for item in subset], np.float32)
        metrics[f"lap_count/{laps}/episodes"] = float(len(subset))
        metrics[f"lap_count/{laps}/full_course_success"] = float(
            np.mean(subset_gates >= subset_targets)
        )
        metrics[f"lap_count/{laps}/mean_gates"] = float(subset_gates.mean())
    return metrics
