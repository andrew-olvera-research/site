"""Multi-rate observation and action tokenizers with JEPA pretraining wrappers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import scaled_dot_product_attention


def _normalization_values(
    normalization: dict[str, Any] | None,
    name: str,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    settings = normalization or {}
    center = torch.as_tensor(settings.get(f"{name}_center", [0.0] * width), dtype=torch.float32)
    scale = torch.as_tensor(settings.get(f"{name}_scale", [1.0] * width), dtype=torch.float32)
    if center.shape != (width,) or scale.shape != (width,):
        raise ValueError(f"{name} normalization must contain exactly {width} values")
    if torch.any(scale <= 0):
        raise ValueError(f"{name} normalization scales must be positive")
    return center, scale


def _require_patchable(value: torch.Tensor, patch_size: int, name: str) -> tuple[int, int, int]:
    if value.ndim < 3:
        raise ValueError(f"{name} must have batch and time dimensions")
    batch, polls = value.shape[:2]
    if polls % patch_size:
        raise ValueError(f"{name} time dimension {polls} must be divisible by patch_size={patch_size}")
    return batch, polls, polls // patch_size


def macro_last(value: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Select the last causally available sample in every temporal patch."""

    batch, _, macros = _require_patchable(value, patch_size, "value")
    return value.reshape(batch, macros, patch_size, *value.shape[2:])[:, :, -1]


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        batch, q_length, width = query.shape
        k_length = source.shape[1]
        q = self.q(query).view(batch, q_length, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(source).chunk(2, dim=-1)
        k = k.view(batch, k_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, k_length, self.n_heads, self.head_dim).transpose(1, 2)
        result = scaled_dot_product_attention(q, k, v)
        return self.out(result.transpose(1, 2).contiguous().view(batch, q_length, width))


class ResamplerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.query_norm = nn.RMSNorm(d_model)
        self.source_norm = nn.RMSNorm(d_model)
        self.attention = CrossAttention(d_model, n_heads)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, queries: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        queries = queries + self.attention(self.query_norm(queries), self.source_norm(source))
        return queries + self.mlp(self.mlp_norm(queries))


class VisualTokenLearner(nn.Module):
    """Cheap spatial encoder followed by adaptive fixed-count token pooling."""

    def __init__(
        self, d_model: int, n_tokens: int, *, coordinate_features: bool = False,
        spatial_anchors: bool = False, anchor_strength: float = 4.0,
    ) -> None:
        super().__init__()
        channels = (32, 64, 128, d_model)
        layers: list[nn.Module] = []
        in_channels = 1
        for width in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(min(8, width), width),
                    nn.SiLU(),
                ]
            )
            in_channels = width
        self.backbone = nn.Sequential(*layers)
        self.coordinate_features = bool(coordinate_features)
        self.coordinate_projection = (
            nn.Conv2d(6, d_model, 1, bias=False) if self.coordinate_features else None
        )
        self.scores = nn.Conv2d(d_model, n_tokens, 1)
        self.n_tokens = int(n_tokens)
        self.spatial_anchors = bool(spatial_anchors)
        self.anchor_strength = float(anchor_strength)
        if self.spatial_anchors:
            side = int(self.n_tokens ** 0.5)
            if side * side != self.n_tokens:
                raise ValueError("spatially anchored visual tokens require a square token count")
            centers = torch.linspace(-0.65, 0.65, side)
            anchors = torch.stack(
                [torch.stack((x, y)) for y in centers for x in centers]
            )
            self.register_buffer("anchors", anchors, persistent=False)
        else:
            self.register_buffer("anchors", torch.empty(0, 2), persistent=False)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        features = self.backbone(mask)
        batch, width, height, columns = features.shape
        if self.coordinate_projection is not None:
            y = torch.linspace(-1.0, 1.0, height, device=features.device, dtype=features.dtype)
            x = torch.linspace(-1.0, 1.0, columns, device=features.device, dtype=features.dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            coordinates = torch.stack(
                [xx, yy, torch.sin(torch.pi * xx), torch.cos(torch.pi * xx),
                 torch.sin(torch.pi * yy), torch.cos(torch.pi * yy)], dim=0
            ).unsqueeze(0).expand(batch, -1, -1, -1)
            features = features + self.coordinate_projection(coordinates)
        values = features.flatten(2).transpose(1, 2)
        scores = self.scores(features)
        if self.spatial_anchors:
            y = torch.linspace(-1.0, 1.0, height, device=features.device, dtype=features.dtype)
            x = torch.linspace(-1.0, 1.0, columns, device=features.device, dtype=features.dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            distance = (
                (xx.unsqueeze(0) - self.anchors[:, 0, None, None]).square()
                + (yy.unsqueeze(0) - self.anchors[:, 1, None, None]).square()
            )
            scores = scores - self.anchor_strength * distance.unsqueeze(0)
        weights = scores.flatten(2).softmax(dim=-1)
        return torch.einsum("bnp,bpd->bnd", weights, values)


class CausalTemporalPatcher(nn.Module):
    """Ordered-poll temporal patch embedding with low/high-frequency paths."""

    def __init__(self, input_dim: int, d_model: int, patch_size: int, n_tokens: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.n_tokens = int(n_tokens)
        self.poll = nn.Sequential(nn.Linear(input_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.position = nn.Parameter(torch.randn(self.patch_size, d_model) * 0.02)
        # mean + last + ordered residuals preserves the block's DC and transient content.
        self.mix = nn.Sequential(
            nn.Linear((self.patch_size + 2) * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, n_tokens * d_model),
        )
        self.d_model = int(d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, _, macros = _require_patchable(values, self.patch_size, "temporal input")
        x = self.poll(values).reshape(batch, macros, self.patch_size, self.d_model)
        x = x + self.position.view(1, 1, self.patch_size, self.d_model)
        low = x.mean(dim=2)
        residual = x - low.unsqueeze(2)
        summary = torch.cat([low, x[:, :, -1], residual.flatten(2)], dim=-1)
        return self.mix(summary).view(batch, macros, self.n_tokens, self.d_model)


class CausalIntraPatchEncoder(nn.Module):
    """Causal attention over every high-rate poll inside a macro step.

    This is a residual companion to :class:`CausalTemporalPatcher`: existing
    checkpoints retain their ordered mean/last/residual path while attention can
    learn interactions between previous actions, rates, motor state, and timing.
    """

    def __init__(
        self, input_dim: int, d_model: int, patch_size: int, n_tokens: int,
        n_heads: int, depth: int,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.n_tokens = int(n_tokens)
        self.d_model = int(d_model)
        self.input = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.randn(self.patch_size, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            activation="gelu",
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.queries = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.resampler = ResamplerBlock(d_model, n_heads)
        self.output = nn.Linear(d_model, d_model)
        # Make checkpoint transfer initially function preserving. The output
        # projection learns first, then opens gradients into the new temporal path.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, _, macros = _require_patchable(values, self.patch_size, "intra-patch input")
        x = self.input(values).view(batch * macros, self.patch_size, self.d_model)
        x = x + self.position.unsqueeze(0)
        causal_mask = torch.ones(
            self.patch_size, self.patch_size, dtype=torch.bool, device=x.device
        ).triu(1)
        x = self.layers(x, mask=causal_mask)
        queries = self.queries.unsqueeze(0).expand(batch * macros, -1, -1)
        tokens = self.resampler(queries, x)
        return self.output(tokens).view(batch, macros, self.n_tokens, self.d_model)


class CausalMacroBeliefEncoder(nn.Module):
    """Causal transformer that carries belief across low-rate macro steps."""

    def __init__(
        self, n_latents: int, d_latent: int, d_model: int, n_heads: int,
        depth: int, max_steps: int, output_latents: int,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.output_latents = int(output_latents)
        self.d_latent = int(d_latent)
        self.input = nn.Linear(n_latents * d_latent, d_model)
        self.position = nn.Parameter(torch.randn(self.max_steps, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            activation="gelu",
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, output_latents * d_latent)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        batch, steps = latents.shape[:2]
        if steps > self.max_steps:
            raise ValueError(
                f"macro sequence has {steps} steps, exceeding max_macro_steps={self.max_steps}"
            )
        x = self.input(latents.flatten(2)) + self.position[:steps].unsqueeze(0)
        causal_mask = torch.ones(steps, steps, dtype=torch.bool, device=x.device).triu(1)
        x = self.layers(x, mask=causal_mask)
        return self.output(self.norm(x)).view(
            batch, steps, self.output_latents, self.d_latent
        )


class RouteTokenizer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_tokens: int, gates: int = 3) -> None:
        super().__init__()
        self.gates = int(gates)
        self.gate_projection = nn.Sequential(nn.Linear(13, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.gate_position = nn.Parameter(torch.randn(gates, d_model) * 0.02)
        self.queries = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.resampler = ResamplerBlock(d_model, n_heads)

    def forward(self, route: torch.Tensor) -> torch.Tensor:
        if route.shape[-2:] != (self.gates, 13):
            raise ValueError(f"route must end in ({self.gates}, 13)")
        batch, time = route.shape[:2]
        source = self.gate_projection(route) + self.gate_position.view(1, 1, self.gates, -1)
        source = source.flatten(0, 1)
        queries = self.queries.unsqueeze(0).expand(batch * time, -1, -1)
        return self.resampler(queries, source).view(batch, time, queries.shape[1], -1)


@dataclass(frozen=True)
class TokenizerOutput:
    latents: torch.Tensor
    mask_logits: torch.Tensor
    proprio: torch.Tensor
    route: torch.Tensor
    vector: torch.Tensor
    state_mean: torch.Tensor
    state_log_scale: torch.Tensor
    modality_tokens: torch.Tensor
    mae_mask: torch.Tensor
    keep_probability: torch.Tensor
    patch_size: int
    mask_geometry: torch.Tensor | None = None
    latent_group_sizes: tuple[int, ...] = ()
    latent_group_names: tuple[str, ...] = ()
    visual_state_mean: torch.Tensor | None = None
    fusion_state_mean: torch.Tensor | None = None
    state_delta_mean: torch.Tensor | None = None
    future_state_mean: torch.Tensor | None = None
    future_state_horizons: tuple[int, ...] = ()
    inverse_action_mean: torch.Tensor | None = None
    progress_mean: torch.Tensor | None = None
    action_state_delta_mean: torch.Tensor | None = None


class MultiModalTokenizer(nn.Module):
    """Fixed-token multi-rate state encoder.

    The compatibility arguments from the prototype tokenizer are accepted so old
    checkpoints/configs fail with a state-dict error rather than a constructor error.
    """

    def __init__(
        self,
        *,
        image_size: tuple[int, int] = (128, 160),
        vector_dim: int = 50,
        proprio_dim: int = 11,
        route_gates: int = 3,
        timing_dim: int = 10,
        temporal_patch_size: int = 3,
        d_model: int = 256,
        n_heads: int = 8,
        n_visual_tokens: int = 4,
        n_proprio_tokens: int = 2,
        n_route_tokens: int = 2,
        estimate_dim: int = 0,
        n_estimate_tokens: int = 0,
        n_latents: int = 4,
        d_bottleneck: int = 128,
        resampler_depth: int = 2,
        normalization: dict[str, Any] | None = None,
        visual_coordinates: bool = False,
        visual_spatial_anchors: bool = False,
        visual_anchor_strength: float = 4.0,
        mask_geometry_head: bool = False,
        predict_state_uncertainty: bool = True,
        reserved_modality_latents: dict[str, int] | None = None,
        modality_specific_decoders: bool = False,
        direct_visual_latents: bool = False,
        spatial_mask_decoder: bool = False,
        state_decoder_tokens: str = "fusion",
        visual_state_auxiliary: bool = False,
        fusion_state_auxiliary: bool = False,
        causal_intra_patch: bool = False,
        intra_patch_depth: int = 2,
        intra_patch_heads: int | None = None,
        macro_belief_depth: int = 0,
        macro_belief_heads: int | None = None,
        macro_belief_dim: int | None = None,
        max_macro_steps: int = 64,
        macro_belief_residual_scale: float = 1.0,
        state_delta_head: bool = False,
        future_state_horizons: tuple[int, ...] | list[int] = (),
        inverse_action_head: bool = False,
        action_dim: int = 4,
        progress_head: bool = False,
        progress_dim: int = 6,
        action_state_delta_head: bool = False,
        dynamics_action_timing_dim: int = 3,
        dynamics_hidden_dim: int | None = None,
        patch_size: int | None = None,
        d_token: int | None = None,
        encoder_depth: int | None = None,
        decoder_depth: int | None = None,
        mae_p_max: float | None = None,
    ) -> None:
        super().__init__()
        del patch_size, d_token, encoder_depth, decoder_depth, mae_p_max
        self.image_size = tuple(image_size)
        self.vector_dim = int(vector_dim)
        self.proprio_dim = int(proprio_dim)
        self.route_gates = int(route_gates)
        self.timing_dim = int(timing_dim)
        self.estimate_dim = int(estimate_dim)
        self.n_estimate_tokens = int(n_estimate_tokens)
        if (self.estimate_dim == 0) != (self.n_estimate_tokens == 0):
            raise ValueError(
                "estimate_dim and n_estimate_tokens must either both be zero or both be positive"
            )
        self.temporal_patch_size = int(temporal_patch_size)
        self.n_latents = int(n_latents)
        self.d_bottleneck = int(d_bottleneck)
        self.predict_state_uncertainty = bool(predict_state_uncertainty)
        reserved = dict(reserved_modality_latents or {})
        if reserved:
            valid_layouts = (("visual", "proprio", "route"), ("visual", "fusion"))
            layout = next((names for names in valid_layouts if set(reserved) == set(names)), None)
            if layout is None or any(int(reserved[name]) < 1 for name in reserved):
                raise ValueError(
                    "reserved_modality_latents must define either visual/proprio/route "
                    "or visual/fusion positive counts"
                )
            if sum(int(reserved[name]) for name in layout) != self.n_latents:
                raise ValueError("reserved modality latent counts must sum to n_latents")
            self.latent_group_names = layout
            self.latent_group_sizes = tuple(int(reserved[name]) for name in layout)
        else:
            self.latent_group_names = ()
            self.latent_group_sizes = ()
        self.modality_specific_decoders = bool(modality_specific_decoders)
        if self.modality_specific_decoders and self.latent_group_names != ("visual", "fusion"):
            raise ValueError(
                "modality_specific_decoders requires visual/fusion reserved latents"
            )
        self.direct_visual_latents = bool(direct_visual_latents)
        self.spatial_mask_decoder = bool(spatial_mask_decoder)
        self.state_decoder_tokens = str(state_decoder_tokens)
        self.visual_state_auxiliary = bool(visual_state_auxiliary)
        self.fusion_state_auxiliary = bool(fusion_state_auxiliary)
        self.causal_intra_patch = bool(causal_intra_patch)
        self.macro_belief_depth = int(macro_belief_depth)
        self.macro_belief_residual_scale = float(macro_belief_residual_scale)
        self.state_delta_enabled = bool(state_delta_head)
        self.future_state_horizons = tuple(int(value) for value in future_state_horizons)
        self.inverse_action_enabled = bool(inverse_action_head)
        self.action_dim = int(action_dim)
        self.progress_enabled = bool(progress_head)
        self.progress_dim = int(progress_dim)
        self.action_state_delta_enabled = bool(action_state_delta_head)
        self.dynamics_action_timing_dim = int(dynamics_action_timing_dim)
        if any(value < 1 for value in self.future_state_horizons):
            raise ValueError("future_state_horizons must contain positive macro-step offsets")
        if len(set(self.future_state_horizons)) != len(self.future_state_horizons):
            raise ValueError("future_state_horizons must not contain duplicates")
        if self.state_decoder_tokens not in {"fusion", "all"}:
            raise ValueError("state_decoder_tokens must be 'fusion' or 'all'")
        if self.direct_visual_latents:
            if self.latent_group_names != ("visual", "fusion"):
                raise ValueError("direct_visual_latents requires visual/fusion reserved latents")
            if self.latent_group_sizes[0] != int(n_visual_tokens):
                raise ValueError("direct visual latent count must equal n_visual_tokens")
        if self.spatial_mask_decoder:
            if not self.direct_visual_latents or self.latent_group_sizes[0] != 4:
                raise ValueError("spatial_mask_decoder requires four direct visual latents")
            if mask_geometry_head:
                raise ValueError("spatial_mask_decoder does not support mask_geometry_head")
        proprio_center, proprio_scale = _normalization_values(
            normalization, "proprio", proprio_dim
        )
        route_center, route_scale = _normalization_values(normalization, "route", 13)
        timing_center, timing_scale = _normalization_values(
            normalization, "timing", timing_dim
        )
        estimate_center, estimate_scale = _normalization_values(
            normalization, "estimate", self.estimate_dim
        )
        # Non-persistent buffers keep old checkpoint state dicts loadable. The
        # values live in model_config and are reconstructed before loading.
        self.register_buffer("proprio_center", proprio_center, persistent=False)
        self.register_buffer("proprio_scale", proprio_scale, persistent=False)
        self.register_buffer("route_center", route_center, persistent=False)
        self.register_buffer("route_scale", route_scale, persistent=False)
        self.register_buffer("timing_center", timing_center, persistent=False)
        self.register_buffer("timing_scale", timing_scale, persistent=False)
        self.register_buffer("estimate_center", estimate_center, persistent=False)
        self.register_buffer("estimate_scale", estimate_scale, persistent=False)
        self.visual = VisualTokenLearner(
            d_model, n_visual_tokens, coordinate_features=visual_coordinates,
            spatial_anchors=visual_spatial_anchors,
            anchor_strength=visual_anchor_strength,
        )
        self.proprio_patcher = CausalTemporalPatcher(
            proprio_dim + timing_dim, d_model, temporal_patch_size, n_proprio_tokens
        )
        self.causal_proprio_patcher = (
            CausalIntraPatchEncoder(
                proprio_dim + timing_dim,
                d_model,
                temporal_patch_size,
                n_proprio_tokens,
                int(intra_patch_heads or n_heads),
                int(intra_patch_depth),
            )
            if self.causal_intra_patch else None
        )
        self.route_tokenizer = RouteTokenizer(d_model, n_heads, n_route_tokens, route_gates)
        self.estimate_patcher = (
            CausalTemporalPatcher(
                self.estimate_dim, d_model, temporal_patch_size, self.n_estimate_tokens
            )
            if self.estimate_dim else None
        )
        total_modalities = (
            n_visual_tokens + n_proprio_tokens + n_route_tokens + self.n_estimate_tokens
        )
        self.modality_embedding = nn.Parameter(torch.randn(total_modalities, d_model) * 0.02)
        if self.latent_group_sizes:
            names = self.latent_group_names
            learned_names = tuple(
                name for name in names
                if not (name == "visual" and self.direct_visual_latents)
            )
            self.modality_queries = nn.ParameterDict({
                name: nn.Parameter(torch.randn(count, d_model) * 0.02)
                for name, count in zip(names, self.latent_group_sizes)
                if name in learned_names
            })
            self.modality_resamplers = nn.ModuleDict({
                name: nn.ModuleList(
                    ResamplerBlock(d_model, n_heads) for _ in range(resampler_depth)
                ) for name in learned_names
            })
            self.modality_bottlenecks = nn.ModuleDict({
                name: nn.Sequential(
                    nn.RMSNorm(d_model), nn.Linear(d_model, d_bottleneck), nn.Tanh()
                ) for name in names
            })
            self.queries = None
            self.resampler = None
            self.bottleneck = None
        else:
            self.queries = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
            self.resampler = nn.ModuleList(
                ResamplerBlock(d_model, n_heads) for _ in range(resampler_depth)
            )
            self.bottleneck = nn.Sequential(
                nn.RMSNorm(d_model), nn.Linear(d_model, d_bottleneck), nn.Tanh()
            )

        self.mask_shape = (32, 40)
        if self.modality_specific_decoders:
            visual_width = self.latent_group_sizes[0] * d_bottleneck
            fusion_width = self.latent_group_sizes[1] * d_bottleneck
            state_width = (
                visual_width + fusion_width
                if self.state_decoder_tokens == "all" else fusion_width
            )
            visual_hidden = max(256, d_bottleneck if self.spatial_mask_decoder else visual_width)
            self.decoder_trunk = None
            self.mask_decoder_trunk = nn.Sequential(
                nn.Linear(
                    d_bottleneck if self.spatial_mask_decoder else visual_width,
                    visual_hidden,
                ),
                nn.SiLU(),
            )
            self.mask_head = nn.Linear(
                visual_hidden,
                (self.mask_shape[0] // 2) * (self.mask_shape[1] // 2)
                if self.spatial_mask_decoder else self.mask_shape[0] * self.mask_shape[1],
            )
            # Direct linear heads make compact state/vector information accessible
            # without relying on a nonlinear decoder to fuse isolated banks.
            self.proprio_head = nn.Linear(fusion_width, proprio_dim)
            self.route_head = nn.Linear(fusion_width, route_gates * 13)
            self.state_head = nn.Linear(
                state_width, 38 if self.predict_state_uncertainty else 19
            )
            self.visual_state_head = (
                nn.Linear(visual_width, 19) if self.visual_state_auxiliary else None
            )
            self.fusion_state_head = (
                nn.Linear(fusion_width, 19) if self.fusion_state_auxiliary else None
            )
            self.mask_geometry_head = (
                nn.Linear(visual_hidden, 6) if mask_geometry_head else None
            )
        else:
            flattened = n_latents * d_bottleneck
            decoder_hidden = max(256, flattened)
            self.decoder_trunk = nn.Sequential(
                nn.Linear(flattened, decoder_hidden), nn.SiLU()
            )
            self.mask_decoder_trunk = None
            self.mask_head = nn.Linear(
                decoder_hidden, self.mask_shape[0] * self.mask_shape[1]
            )
            self.proprio_head = nn.Linear(decoder_hidden, proprio_dim)
            self.route_head = nn.Linear(decoder_hidden, route_gates * 13)
            self.state_head = nn.Linear(
                decoder_hidden, 38 if self.predict_state_uncertainty else 19
            )
            if (
                self.state_decoder_tokens != "fusion"
                or self.visual_state_auxiliary
                or self.fusion_state_auxiliary
            ):
                raise ValueError(
                    "state decoder routing options require visual/fusion modality decoders"
                )
            self.visual_state_head = None
            self.fusion_state_head = None
            self.mask_geometry_head = (
                nn.Linear(decoder_hidden, 6) if mask_geometry_head else None
            )

        if self.macro_belief_depth:
            belief_latents = (
                self.latent_group_sizes[self.latent_group_names.index("fusion")]
                if "fusion" in self.latent_group_names else self.n_latents
            )
            self.macro_belief = CausalMacroBeliefEncoder(
                self.n_latents,
                self.d_bottleneck,
                int(macro_belief_dim or d_model),
                int(macro_belief_heads or n_heads),
                self.macro_belief_depth,
                int(max_macro_steps),
                belief_latents,
            )
        else:
            self.macro_belief = None

        auxiliary_width = (
            self.latent_group_sizes[self.latent_group_names.index("fusion")] * d_bottleneck
            if "fusion" in self.latent_group_names else self.n_latents * d_bottleneck
        )
        self.state_delta_head = (
            nn.Linear(auxiliary_width, 19) if self.state_delta_enabled else None
        )
        self.future_state_head = (
            nn.Linear(auxiliary_width, len(self.future_state_horizons) * 19)
            if self.future_state_horizons else None
        )
        self.inverse_action_head = (
            nn.Sequential(
                nn.Linear(2 * auxiliary_width, auxiliary_width),
                nn.SiLU(),
                nn.Linear(auxiliary_width, self.temporal_patch_size * self.action_dim),
            )
            if self.inverse_action_enabled else None
        )
        self.progress_head = (
            nn.Linear(auxiliary_width, self.progress_dim) if self.progress_enabled else None
        )
        dynamics_width = int(dynamics_hidden_dim or max(256, auxiliary_width))
        self.action_state_delta_head = (
            nn.Sequential(
                nn.Linear(
                    auxiliary_width + self.action_dim + self.dynamics_action_timing_dim,
                    dynamics_width,
                ),
                nn.SiLU(),
                nn.Linear(dynamics_width, 19),
            )
            if self.action_state_delta_enabled else None
        )

    def _coerce_inputs(
        self,
        mask: torch.Tensor | dict[str, torch.Tensor],
        vector: torch.Tensor | None = None,
        *,
        proprio: torch.Tensor | None = None,
        route: torch.Tensor | None = None,
        timing: torch.Tensor | None = None,
        estimate: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        if isinstance(mask, dict):
            batch = mask
            mask = batch["mask"]
            proprio = batch.get("proprio")
            route = batch.get("route")
            timing = batch.get("timing")
            estimate = batch.get("estimate")
            vector = batch.get("vector")
        if proprio is None or route is None:
            if vector is None or vector.shape[-1] != self.vector_dim:
                raise ValueError("provide proprio/route or a compatible 50D vector")
            proprio = vector[..., : self.proprio_dim]
            route = vector[..., self.proprio_dim :].reshape(*vector.shape[:-1], self.route_gates, 13)
        if timing is None:
            timing = proprio.new_zeros(*proprio.shape[:-1], self.timing_dim)
        if timing.shape[-1] < self.timing_dim:
            timing = F.pad(timing, (0, self.timing_dim - timing.shape[-1]))
        elif timing.shape[-1] > self.timing_dim:
            timing = timing[..., : self.timing_dim]
        if self.estimate_dim:
            if estimate is None or estimate.shape[-1] != self.estimate_dim:
                raise ValueError(
                    f"estimate modality must end in {self.estimate_dim} values"
                )
        return mask, proprio, route, timing, estimate

    def encode(
        self,
        mask: torch.Tensor | dict[str, torch.Tensor],
        vector: torch.Tensor | None = None,
        **modalities: torch.Tensor,
    ) -> torch.Tensor:
        mask, proprio, route, timing, estimate = self._coerce_inputs(
            mask, vector, **modalities
        )
        proprio = (proprio - self.proprio_center) / self.proprio_scale
        route = (route - self.route_center) / self.route_scale
        timing = (timing - self.timing_center) / self.timing_scale
        if estimate is not None:
            estimate = (estimate - self.estimate_center) / self.estimate_scale
        batch, _, macros = _require_patchable(proprio, self.temporal_patch_size, "proprio")
        macro_mask = macro_last(mask, self.temporal_patch_size)
        visual = self.visual(macro_mask.flatten(0, 1)).view(batch, macros, -1, self.modality_embedding.shape[-1])
        temporal_input = torch.cat([proprio, timing], dim=-1)
        prop = self.proprio_patcher(temporal_input)
        if self.causal_proprio_patcher is not None:
            prop = prop + self.causal_proprio_patcher(temporal_input)
        macro_route = macro_last(route, self.temporal_patch_size)
        route_tokens = self.route_tokenizer(macro_route)
        estimate_tokens = (
            self.estimate_patcher(estimate)
            if self.estimate_patcher is not None and estimate is not None else None
        )
        modality_names = ["visual", "proprio", "route"]
        modality_values = [visual, prop, route_tokens]
        if estimate_tokens is not None:
            modality_names.append("estimate")
            modality_values.append(estimate_tokens)
        counts = tuple(value.shape[2] for value in modality_values)
        embeddings = self.modality_embedding.split(counts, dim=0)
        sources = {
            name: tokens + embedding.view(1, 1, -1, embedding.shape[-1])
            for name, tokens, embedding in zip(
                modality_names,
                modality_values,
                embeddings,
            )
        }
        if self.latent_group_sizes:
            groups = []
            for name in self.latent_group_names:
                group_source = (
                    torch.cat(tuple(sources.values()), dim=2)
                    if name == "fusion" else sources[name]
                )
                source_flat = group_source.flatten(0, 1)
                if name == "visual" and self.direct_visual_latents:
                    queries = source_flat
                else:
                    queries = self.modality_queries[name].unsqueeze(0).expand(
                        batch * macros, -1, -1
                    )
                    for layer in self.modality_resamplers[name]:
                        queries = layer(queries, source_flat)
                groups.append(self.modality_bottlenecks[name](queries))
            latents = torch.cat(groups, dim=1)
        else:
            source_flat = torch.cat(tuple(sources.values()), dim=2).flatten(0, 1)
            queries = self.queries.unsqueeze(0).expand(batch * macros, -1, -1)
            for layer in self.resampler:
                queries = layer(queries, source_flat)
            latents = self.bottleneck(queries)
        latents = latents.view(batch, macros, self.n_latents, self.d_bottleneck)
        if self.macro_belief is not None:
            belief = self.macro_belief(latents)
            belief = belief * self.macro_belief_residual_scale
            if "fusion" in self.latent_group_names:
                index = self.latent_group_names.index("fusion")
                start = sum(self.latent_group_sizes[:index])
                stop = start + self.latent_group_sizes[index]
                latents = torch.cat(
                    (latents[..., :start, :], latents[..., start:stop, :] + belief,
                     latents[..., stop:, :]),
                    dim=-2,
                )
            else:
                latents = latents + belief
        return latents

    def select_latent_group(self, latents: torch.Tensor, name: str) -> torch.Tensor:
        if name not in self.latent_group_names:
            raise ValueError(f"latent group {name!r} is not available")
        index = self.latent_group_names.index(name)
        start = sum(self.latent_group_sizes[:index])
        return latents[..., start : start + self.latent_group_sizes[index], :]

    def estimate_state(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode the tokenizer-trained state estimate without rendering masks."""

        if latents.ndim != 4 or latents.shape[-2:] != (self.n_latents, self.d_bottleneck):
            raise ValueError(
                f"latents must have shape (B,T,{self.n_latents},{self.d_bottleneck})"
            )
        if self.modality_specific_decoders:
            visual = self.select_latent_group(latents, "visual").flatten(2)
            fusion = self.select_latent_group(latents, "fusion").flatten(2)
            features = (
                torch.cat((visual, fusion), dim=-1)
                if self.state_decoder_tokens == "all" else fusion
            )
        else:
            features = self.decoder_trunk(latents.flatten(2))
        output = self.state_head(features)
        if self.predict_state_uncertainty:
            mean, log_scale = output.chunk(2, dim=-1)
        else:
            mean, log_scale = output, torch.zeros_like(output)
        return mean, log_scale.clamp(-7.0, 3.0)

    def estimate_proprio(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode policy proprioception from latents without rendering masks."""

        if latents.ndim != 4 or latents.shape[-2:] != (self.n_latents, self.d_bottleneck):
            raise ValueError(
                f"latents must have shape (B,T,{self.n_latents},{self.d_bottleneck})"
            )
        if self.modality_specific_decoders:
            features = self.select_latent_group(latents, "fusion").flatten(2)
        else:
            features = self.decoder_trunk(latents.flatten(2))
        return self.proprio_head(features)

    def forward(
        self,
        mask: torch.Tensor | dict[str, torch.Tensor],
        vector: torch.Tensor | None = None,
        **modalities: torch.Tensor,
    ) -> TokenizerOutput:
        batch = mask if isinstance(mask, dict) else None
        mask_value, proprio, route, timing, estimate = self._coerce_inputs(
            mask, vector, **modalities
        )
        latents = self.encode(
            mask_value, proprio=proprio, route=route, timing=timing,
            estimate=estimate,
        )
        normalized_action = batch.get("normalized_action") if batch is not None else None
        action_timing = batch.get("action_timing") if batch is not None else None
        return self.decode(
            latents,
            normalized_action=normalized_action,
            action_timing=action_timing,
        )

    def decode(
        self,
        latents: torch.Tensor,
        *,
        normalized_action: torch.Tensor | None = None,
        action_timing: torch.Tensor | None = None,
    ) -> TokenizerOutput:
        """Decode externally predicted bottleneck tokens for rendering/probes."""
        if latents.ndim != 4 or latents.shape[-2:] != (self.n_latents, self.d_bottleneck):
            raise ValueError(
                f"latents must have shape (B,T,{self.n_latents},{self.d_bottleneck})"
            )
        batch, macros = latents.shape[:2]
        if self.modality_specific_decoders:
            visual_group = self.select_latent_group(latents, "visual")
            visual_flat = visual_group.flatten(2)
            visual_latents = (
                visual_group if self.spatial_mask_decoder else visual_group.flatten(2)
            )
            fusion_latents = self.select_latent_group(latents, "fusion").flatten(2)
            auxiliary_features = fusion_latents
            mask_features = self.mask_decoder_trunk(visual_latents)
            vector_features = fusion_latents
            state_features = (
                torch.cat((visual_flat, fusion_latents), dim=-1)
                if self.state_decoder_tokens == "all" else fusion_latents
            )
            visual_state_mean = (
                self.visual_state_head(visual_flat)
                if self.visual_state_head is not None else None
            )
            fusion_state_mean = (
                self.fusion_state_head(fusion_latents)
                if self.fusion_state_head is not None else None
            )
        else:
            auxiliary_features = latents.flatten(2)
            decoded = self.decoder_trunk(auxiliary_features)
            mask_features = vector_features = decoded
            state_features = decoded
            visual_state_mean = None
            fusion_state_mean = None
        if self.spatial_mask_decoder:
            patch_height, patch_width = self.mask_shape[0] // 2, self.mask_shape[1] // 2
            patches = self.mask_head(mask_features).view(
                batch, macros, 2, 2, patch_height, patch_width
            )
            mask_logits = torch.cat(
                [torch.cat([patches[:, :, row, column] for column in range(2)], dim=-1)
                 for row in range(2)],
                dim=-2,
            ).reshape(batch * macros, 1, *self.mask_shape)
        else:
            mask_logits = self.mask_head(mask_features).view(
                batch * macros, 1, *self.mask_shape
            )
        mask_logits = F.interpolate(mask_logits, size=self.image_size, mode="bilinear", align_corners=False)
        mask_logits = mask_logits.view(batch, macros, 1, *self.image_size)
        proprio_prediction = self.proprio_head(vector_features)
        route_prediction = self.route_head(vector_features).view(batch, macros, self.route_gates, 13)
        state_output = self.state_head(state_features)
        if self.predict_state_uncertainty:
            state_mean, state_log_scale = state_output.chunk(2, dim=-1)
        else:
            state_mean = state_output
            state_log_scale = torch.zeros_like(state_mean)
        mask_geometry = (
            self.mask_geometry_head(mask_features)
            if self.mask_geometry_head is not None else None
        )
        vector_prediction = torch.cat([proprio_prediction, route_prediction.flatten(2)], dim=-1)
        state_delta_mean = (
            self.state_delta_head(auxiliary_features)
            if self.state_delta_head is not None else None
        )
        future_state_mean = (
            self.future_state_head(auxiliary_features).view(
                batch, macros, len(self.future_state_horizons), 19
            )
            if self.future_state_head is not None else None
        )
        inverse_action_mean = None
        if self.inverse_action_head is not None:
            inverse_features = torch.cat(
                (auxiliary_features[:, :-1], auxiliary_features[:, 1:]), dim=-1
            )
            inverse_action_mean = self.inverse_action_head(inverse_features).view(
                batch, max(0, macros - 1), self.temporal_patch_size, self.action_dim
            )
        progress_mean = (
            self.progress_head(auxiliary_features)
            if self.progress_head is not None else None
        )
        action_state_delta_mean = None
        if self.action_state_delta_head is not None:
            if self.temporal_patch_size != 1:
                raise ValueError("action-conditioned state delta requires temporal_patch_size=1")
            if normalized_action is not None and action_timing is not None:
                if normalized_action.shape[:2] != auxiliary_features.shape[:2]:
                    raise ValueError("normalized actions must align with observation latents")
                if action_timing.shape[:2] != auxiliary_features.shape[:2]:
                    raise ValueError("action timing must align with observation latents")
                if action_timing.shape[-1] < self.dynamics_action_timing_dim:
                    action_timing = F.pad(
                        action_timing,
                        (0, self.dynamics_action_timing_dim - action_timing.shape[-1]),
                    )
                dynamics_input = torch.cat(
                    (
                        auxiliary_features,
                        normalized_action,
                        action_timing[..., : self.dynamics_action_timing_dim],
                    ),
                    dim=-1,
                )
                action_state_delta_mean = self.action_state_delta_head(dynamics_input)
        modality_tokens = latents.new_empty(batch, macros, 0, latents.shape[-1])
        return TokenizerOutput(
            latents=latents,
            mask_logits=mask_logits,
            proprio=proprio_prediction,
            route=route_prediction,
            vector=vector_prediction,
            state_mean=state_mean,
            state_log_scale=state_log_scale.clamp(-7.0, 3.0),
            modality_tokens=modality_tokens,
            mae_mask=torch.zeros(batch, macros, 1, 1, dtype=torch.bool, device=latents.device),
            keep_probability=torch.ones(batch, macros, 1, device=latents.device, dtype=latents.dtype),
            patch_size=self.temporal_patch_size,
            mask_geometry=mask_geometry,
            latent_group_sizes=self.latent_group_sizes,
            latent_group_names=self.latent_group_names,
            visual_state_mean=visual_state_mean,
            fusion_state_mean=fusion_state_mean,
            state_delta_mean=state_delta_mean,
            future_state_mean=future_state_mean,
            future_state_horizons=self.future_state_horizons,
            inverse_action_mean=inverse_action_mean,
            progress_mean=progress_mean,
            action_state_delta_mean=action_state_delta_mean,
        )


class CausalLatentPredictorBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.space_norm = nn.RMSNorm(d_model)
        self.space_qkv = nn.Linear(d_model, 3 * d_model)
        self.space_out = nn.Linear(d_model, d_model)
        self.time_norm = nn.RMSNorm(d_model)
        self.time_qkv = nn.Linear(d_model, 3 * d_model)
        self.time_out = nn.Linear(d_model, d_model)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(approximate="tanh"), nn.Linear(4 * d_model, d_model)
        )

    def _attention(
        self,
        x: torch.Tensor,
        qkv: nn.Linear,
        output: nn.Linear,
        *,
        causal: bool,
    ) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = qkv(x).chunk(3, dim=-1)
        reshape = lambda value: value.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        result = scaled_dot_product_attention(
            reshape(q), reshape(k), reshape(v), is_causal=causal
        )
        return output(result.transpose(1, 2).contiguous().view(batch, length, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, tokens, width = x.shape
        spatial = self.space_norm(x).reshape(batch * time, tokens, width)
        spatial = self._attention(spatial, self.space_qkv, self.space_out, causal=False)
        x = x + spatial.view(batch, time, tokens, width)
        temporal = self.time_norm(x).permute(0, 2, 1, 3).reshape(batch * tokens, time, width)
        temporal = self._attention(temporal, self.time_qkv, self.time_out, causal=True)
        temporal = temporal.view(batch, tokens, time, width).permute(0, 2, 1, 3)
        x = x + temporal
        return x + self.mlp(self.mlp_norm(x))


class CausalLatentPredictor(nn.Module):
    def __init__(
        self, n_tokens: int, d_latent: int, d_model: int, n_heads: int, depth: int,
        conditioning_dim: int = 0,
    ) -> None:
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.input = nn.Linear(d_latent, d_model)
        self.conditioning = nn.Linear(conditioning_dim, d_model) if conditioning_dim else None
        self.token_position = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.blocks = nn.ModuleList(CausalLatentPredictorBlock(d_model, n_heads) for _ in range(depth))
        self.norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, d_latent)

    def forward(
        self, latents: torch.Tensor, conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, time, tokens, _ = latents.shape
        if tokens != self.n_tokens:
            raise ValueError("latent token count mismatch")
        x = self.input(latents) + self.token_position.view(1, 1, tokens, -1)
        if self.conditioning is not None:
            if conditioning is None or conditioning.shape[:2] != (batch, time):
                raise ValueError("predictor conditioning must match latent batch/time dimensions")
            x = x + self.conditioning(conditioning).unsqueeze(2)
        elif conditioning is not None:
            raise ValueError("conditioning was provided to an unconditional predictor")
        for block in self.blocks:
            x = block(x)
        return torch.tanh(self.output(self.norm(x)))


@dataclass(frozen=True)
class ObservationJEPAOutput:
    reconstruction: TokenizerOutput
    prediction: torch.Tensor
    target: torch.Tensor
    unconditioned_prediction: torch.Tensor | None = None


class ObservationJEPA(nn.Module):
    """Online/EMA-target observation encoder with causal next-patch prediction."""

    def __init__(
        self, encoder: MultiModalTokenizer, *, predictor_dim: int = 256,
        predictor_heads: int = 8, predictor_depth: int = 4,
        action_conditioned: bool = False, action_dim: int = 4, action_timing_dim: int = 3,
        action_conditioning_dropout: float = 0.0, dual_prediction: bool = False,
        prediction_tokens: str = "all", stop_jepa_gradient_to_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder).requires_grad_(False)
        self.action_conditioned = bool(action_conditioned)
        if not 0.0 <= float(action_conditioning_dropout) < 1.0:
            raise ValueError("action_conditioning_dropout must be in [0, 1)")
        self.action_conditioning_dropout = float(action_conditioning_dropout)
        self.dual_prediction = bool(dual_prediction)
        if prediction_tokens not in {"all", "summary", "fusion"}:
            raise ValueError("prediction_tokens must be 'all', 'summary', or 'fusion'")
        if prediction_tokens == "fusion" and "fusion" not in encoder.latent_group_names:
            raise ValueError("prediction_tokens='fusion' requires a fusion latent group")
        self.prediction_tokens = prediction_tokens
        self.stop_jepa_gradient_to_encoder = bool(stop_jepa_gradient_to_encoder)
        conditioning_dim = (
            encoder.temporal_patch_size * (int(action_dim) + int(action_timing_dim))
            if self.action_conditioned else 0
        )
        if self.prediction_tokens == "summary":
            predicted_tokens = 1
        elif self.prediction_tokens == "fusion":
            predicted_tokens = encoder.latent_group_sizes[
                encoder.latent_group_names.index("fusion")
            ]
        else:
            predicted_tokens = encoder.n_latents
        self.predictor = CausalLatentPredictor(
            predicted_tokens, encoder.d_bottleneck, predictor_dim, predictor_heads,
            predictor_depth, conditioning_dim=conditioning_dim,
        )
        self.unconditioned_predictor = (
            CausalLatentPredictor(
                predicted_tokens, encoder.d_bottleneck, predictor_dim,
                predictor_heads, predictor_depth,
            ) if self.dual_prediction else None
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> ObservationJEPAOutput:
        reconstruction = self.encoder(batch)
        with torch.no_grad():
            target = self.target_encoder.encode(batch)
        if target.shape[1] < 2:
            raise ValueError("JEPA training requires at least two macro steps")
        conditioning = None
        if self.action_conditioned:
            if "normalized_action" not in batch or "action_timing" not in batch:
                raise ValueError("action-conditioned observation JEPA requires normalized_action and action_timing")
            actions = torch.cat([batch["normalized_action"], batch["action_timing"]], dim=-1)
            batch_size, _, macros = _require_patchable(
                actions, self.encoder.temporal_patch_size, "observation JEPA actions"
            )
            # action[t] causes observation[t] -> observation[t+1]. Condition
            # the predictor for latent[t+1] on the action aligned with latent[t],
            # never on the future command at t+1.
            conditioning = actions.reshape(batch_size, macros, -1)[:, :-1]
            if self.training and self.action_conditioning_dropout:
                keep = (
                    torch.rand(*conditioning.shape[:2], 1, device=conditioning.device)
                    >= self.action_conditioning_dropout
                ).to(conditioning.dtype)
                conditioning = conditioning * keep / (1.0 - self.action_conditioning_dropout)
        online_latents = reconstruction.latents
        target_latents = target
        if self.prediction_tokens == "summary":
            online_latents = online_latents.mean(dim=2, keepdim=True)
            target_latents = target_latents.mean(dim=2, keepdim=True)
        elif self.prediction_tokens == "fusion":
            online_latents = self.encoder.select_latent_group(online_latents, "fusion")
            target_latents = self.target_encoder.select_latent_group(
                target_latents, "fusion"
            )
        predictor_input = online_latents[:, :-1]
        if self.stop_jepa_gradient_to_encoder:
            predictor_input = predictor_input.detach()
        prediction = self.predictor(predictor_input, conditioning)
        unconditioned_prediction = (
            self.unconditioned_predictor(predictor_input)
            if self.unconditioned_predictor is not None else None
        )
        return ObservationJEPAOutput(
            reconstruction, prediction, target_latents[:, 1:].detach(), unconditioned_prediction,
        )

    @torch.no_grad()
    def update_target(self, rate: float = 0.005) -> None:
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target.lerp_(online, rate)


@dataclass(frozen=True)
class ActionTokenizerOutput:
    tokens: torch.Tensor
    context: torch.Tensor
    mean: torch.Tensor
    log_std: torch.Tensor
    patch_size: int
    state_delta_mean: torch.Tensor | None = None


class ActionTokenizer(nn.Module):
    """Ordered applied-action temporal patch tokenizer with Gaussian reconstruction."""

    def __init__(
        self,
        *,
        action_dim: int = 4,
        timing_dim: int = 3,
        temporal_patch_size: int = 3,
        d_model: int = 256,
        d_latent: int = 128,
        n_tokens: int = 2,
        normalization: dict[str, Any] | None = None,
        predict_uncertainty: bool = True,
        state_dim: int = 19,
        predict_state_effect: bool = False,
        state_effect_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.timing_dim = int(timing_dim)
        self.temporal_patch_size = int(temporal_patch_size)
        self.n_tokens = int(n_tokens)
        self.d_latent = int(d_latent)
        self.predict_uncertainty = bool(predict_uncertainty)
        self.state_dim = int(state_dim)
        self.predict_state_effect = bool(predict_state_effect)
        action_center, action_scale = _normalization_values(
            normalization, "action", action_dim
        )
        timing_center, timing_scale = _normalization_values(
            normalization, "timing", timing_dim
        )
        self.register_buffer("action_center", action_center, persistent=False)
        self.register_buffer("action_scale", action_scale, persistent=False)
        self.register_buffer("timing_center", timing_center, persistent=False)
        self.register_buffer("timing_scale", timing_scale, persistent=False)
        self.patcher = CausalTemporalPatcher(
            action_dim + timing_dim, d_model, temporal_patch_size, n_tokens
        )
        self.to_latent = nn.Sequential(nn.RMSNorm(d_model), nn.Linear(d_model, d_latent), nn.Tanh())
        self.context_head = nn.Sequential(
            nn.Linear(n_tokens * d_latent, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_tokens * d_latent, 2 * d_model), nn.SiLU(),
            nn.Linear(
                2 * d_model,
                (2 if self.predict_uncertainty else 1) * temporal_patch_size * action_dim,
            ),
        )
        self.context_dim = int(d_model)
        effect_hidden = int(state_effect_hidden_dim or d_model)
        self.state_effect_head = (
            nn.Sequential(
                nn.Linear(self.context_dim + self.state_dim, effect_hidden),
                nn.SiLU(),
                nn.Linear(effect_hidden, self.state_dim),
            )
            if self.predict_state_effect else None
        )

    def encode(self, actions: torch.Tensor, timing: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if timing is None:
            timing = actions.new_zeros(*actions.shape[:-1], self.timing_dim)
        if timing.shape[-1] < self.timing_dim:
            timing = F.pad(timing, (0, self.timing_dim - timing.shape[-1]))
        timing = timing[..., : self.timing_dim]
        actions = (actions - self.action_center) / self.action_scale
        timing = (timing - self.timing_center) / self.timing_scale
        tokens = self.to_latent(self.patcher(torch.cat([actions, timing], dim=-1)))
        return tokens, self.context_head(tokens.flatten(2))

    def decode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one or more frozen action-token macro steps to normalized CTBR."""

        if (
            tokens.ndim != 4
            or tokens.shape[2] != self.n_tokens
            or tokens.shape[3] != self.d_latent
        ):
            raise ValueError(
                "action tokens must have shape (B,T,n_tokens,d_latent)"
            )
        batch, macros = tokens.shape[:2]
        decoded = self.decoder(tokens.flatten(2))
        if self.predict_uncertainty:
            mean, log_std = decoded.view(
                batch, macros, self.temporal_patch_size, self.action_dim, 2
            ).unbind(dim=-1)
        else:
            mean = decoded.view(
                batch, macros, self.temporal_patch_size, self.action_dim
            )
            log_std = torch.zeros_like(mean)
        return mean.tanh(), log_std.clamp(-5.0, 2.0)

    def forward(self, actions: torch.Tensor | dict[str, torch.Tensor], timing: torch.Tensor | None = None) -> ActionTokenizerOutput:
        task_state = None
        if isinstance(actions, dict):
            task_state = actions.get("task_state")
            timing = actions.get("action_timing")
            actions = actions["applied_action"]
        tokens, context = self.encode(actions, timing)
        batch, macros = tokens.shape[:2]
        mean, log_std = self.decode(tokens)
        state_delta_mean = None
        if self.state_effect_head is not None and task_state is not None:
            if self.temporal_patch_size != 1:
                raise ValueError("action state-effect prediction requires temporal_patch_size=1")
            if task_state.shape[1] != macros + 1:
                raise ValueError(
                    "action state-effect prediction requires T+1 task_state samples"
                )
            state_delta_mean = self.state_effect_head(
                torch.cat((context, task_state[:, :-1]), dim=-1)
            )
        return ActionTokenizerOutput(
            tokens=tokens,
            context=context,
            mean=mean,
            log_std=log_std,
            patch_size=self.temporal_patch_size,
            state_delta_mean=state_delta_mean,
        )


@dataclass(frozen=True)
class ActionJEPAOutput:
    reconstruction: ActionTokenizerOutput
    prediction: torch.Tensor | None
    target: torch.Tensor | None
    reversed_tokens: torch.Tensor | None = None


class ActionJEPA(nn.Module):
    def __init__(
        self, tokenizer: ActionTokenizer, *, predictor_dim: int = 256,
        predictor_heads: int = 8, predictor_depth: int = 3,
        prediction_mode: str = "next",
        order_contrastive: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        if prediction_mode not in {"next", "none"}:
            raise ValueError("action prediction_mode must be 'next' or 'none'")
        self.prediction_mode = prediction_mode
        self.order_contrastive = bool(order_contrastive)
        self.target_tokenizer = (
            copy.deepcopy(tokenizer).requires_grad_(False) if prediction_mode == "next" else None
        )
        self.predictor = (
            CausalLatentPredictor(
                tokenizer.n_tokens, tokenizer.d_latent, predictor_dim,
                predictor_heads, predictor_depth,
            ) if prediction_mode == "next" else None
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> ActionJEPAOutput:
        reconstruction = self.tokenizer(batch)
        reversed_tokens = None
        if self.order_contrastive:
            actions = batch["applied_action"]
            batch_size, polls = actions.shape[:2]
            patch = self.tokenizer.temporal_patch_size
            if polls % patch:
                raise ValueError("action sequence length must be divisible by the action patch size")
            reversed_actions = actions.reshape(
                batch_size, polls // patch, patch, actions.shape[-1]
            ).flip(2).reshape_as(actions)
            reversed_tokens, _ = self.tokenizer.encode(
                reversed_actions, batch.get("action_timing")
            )
        if self.prediction_mode == "none":
            return ActionJEPAOutput(reconstruction, None, None, reversed_tokens)
        assert self.target_tokenizer is not None and self.predictor is not None
        with torch.no_grad():
            target, _ = self.target_tokenizer.encode(batch["applied_action"], batch.get("action_timing"))
        if target.shape[1] < 2:
            raise ValueError("action JEPA training requires at least two temporal patches")
        prediction = self.predictor(reconstruction.tokens[:, :-1])
        return ActionJEPAOutput(
            reconstruction, prediction, target[:, 1:].detach(), reversed_tokens
        )

    @torch.no_grad()
    def update_target(self, rate: float = 0.005) -> None:
        if self.target_tokenizer is None:
            return
        for target, online in zip(self.target_tokenizer.parameters(), self.tokenizer.parameters()):
            target.lerp_(online, rate)
