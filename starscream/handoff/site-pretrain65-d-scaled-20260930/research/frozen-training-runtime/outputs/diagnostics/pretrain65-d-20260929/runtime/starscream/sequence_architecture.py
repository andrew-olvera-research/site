"""Configurable continuous-token sequence blocks for privileged racing.

The legacy racing policy projects a complete transition to one token.  This
module keeps the same causal observation and CTBR contracts while exposing the
semantic structure that attention can use: exact plant state, previous command
and timing, and one token per future gate.  All optional mechanisms are local
to representation learning and preserve the four-dimensional PPO action.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class FixedRandomFourierFeatures(nn.Module):
    """Deterministic non-trainable Fourier lift for normalized coordinates."""

    def __init__(
        self, input_dim: int, features: int, *, scale: float = 2.0, seed: int = 0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or features < 1 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid random Fourier feature dimensions")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        frequency = torch.randn(input_dim, features, generator=generator) * float(scale)
        phase = torch.rand(features, generator=generator) * (2.0 * math.pi)
        self.register_buffer("frequency", frequency)
        self.register_buffer("phase", phase)
        self.output_dim = int(input_dim + 2 * features)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        angle = 2.0 * math.pi * value @ self.frequency + self.phase
        return torch.cat([value, angle.sin(), angle.cos()], dim=-1)


class SwiGLUProjection(nn.Module):
    """RMS-normalized nonlinear projection for continuous observations."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        fourier_features: int = 0,
        fourier_scale: float = 2.0,
        seed: int = 0,
        input_rms_norm: bool = True,
    ) -> None:
        super().__init__()
        self.fourier: nn.Module | None = None
        lifted_dim = int(input_dim)
        if fourier_features:
            fourier = FixedRandomFourierFeatures(
                input_dim, fourier_features, scale=fourier_scale, seed=seed,
            )
            self.fourier = fourier
            lifted_dim = fourier.output_dim
        self.norm = nn.RMSNorm(lifted_dim) if input_rms_norm else nn.Identity()
        self.input = nn.Linear(lifted_dim, 2 * output_dim)
        self.output = nn.Linear(output_dim, output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.fourier is not None:
            value = self.fourier(value)
        gate, content = self.input(self.norm(value)).chunk(2, dim=-1)
        return self.output(F.silu(gate) * content)


class BiasedSelfAttention(nn.Module):
    """Self-attention accepting per-head causal/ALiBi additive bias."""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        if heads < 1 or d_model % heads:
            raise ValueError("attention width must be divisible by head count")
        self.heads = int(heads)
        self.head_dim = int(d_model) // self.heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)

    def forward(
        self,
        value: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        *,
        is_causal: bool = False,
        linear_bias_slopes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, width = value.shape
        q, k, v = self.qkv(value).chunk(3, dim=-1)
        q = q.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        scale: float | None = None
        trim_width = self.head_dim
        if linear_bias_slopes is not None:
            if attention_bias is not None or not is_causal:
                raise ValueError(
                    "linear attention bias requires mask-free causal attention"
                )
            if linear_bias_slopes.shape != (self.heads,):
                raise ValueError("linear attention bias slopes must match heads")
            # For causal q >= k, ALiBi is -m(q-k) = m*k - m*q.  The
            # query-only term vanishes under softmax, so append q'=1 and
            # k'=m*k_position/scale. This is exactly equivalent but keeps the
            # SDPA mask null, allowing the fused FlashAttention kernel.
            scale = 1.0 / math.sqrt(self.head_dim)
            position = torch.arange(
                length, device=value.device, dtype=value.dtype
            )
            query_bias = value.new_ones(batch, self.heads, length, 1)
            key_bias = (
                linear_bias_slopes.to(device=value.device, dtype=value.dtype)[
                    None, :, None, None
                ]
                * position[None, None, :, None]
                / scale
            ).expand(batch, -1, -1, -1)
            q = torch.cat([q, query_bias], dim=-1)
            k = torch.cat([k, key_bias], dim=-1)
            v = torch.cat([v, torch.zeros_like(query_bias)], dim=-1)
        attended = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=scale,
        )[..., :trim_width]
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.output(attended)

    def probabilities(
        self,
        value: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        *,
        is_causal: bool = False,
        linear_bias_slopes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return exact pre-dropout attention probabilities for diagnostics."""

        batch, length, _ = value.shape
        q, k, _ = self.qkv(value).chunk(3, dim=-1)
        q = q.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if attention_bias is not None:
            logits = logits + attention_bias
        if linear_bias_slopes is not None:
            query = torch.arange(length, device=value.device)
            key = torch.arange(length, device=value.device)
            age = (query[:, None] - key[None, :]).clamp_min(0).to(value.dtype)
            logits = logits - linear_bias_slopes.to(
                device=value.device, dtype=value.dtype
            )[None, :, None, None] * age[None, None]
        if is_causal:
            causal = torch.ones(
                length, length, dtype=torch.bool, device=value.device
            ).tril()
            logits = logits.masked_fill(
                ~causal[None, None], torch.finfo(logits.dtype).min
            )
        return logits.float().softmax(dim=-1).to(value.dtype)


class CrossAttention(nn.Module):
    """Fused multi-head cross-attention for small learned control queries."""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        if heads < 1 or d_model % heads:
            raise ValueError("cross-attention width must be divisible by head count")
        self.heads = int(heads)
        self.head_dim = int(d_model) // self.heads
        self.query = nn.Linear(d_model, d_model)
        self.key_value = nn.Linear(d_model, 2 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)

    def _qkv(
        self, query: torch.Tensor, memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, query_length, width = query.shape
        memory_length = memory.shape[1]
        q = self.query(query).reshape(
            batch, query_length, self.heads, self.head_dim
        ).transpose(1, 2)
        k, v = self.key_value(memory).chunk(2, dim=-1)
        k = k.reshape(
            batch, memory_length, self.heads, self.head_dim
        ).transpose(1, 2)
        v = v.reshape(
            batch, memory_length, self.heads, self.head_dim
        ).transpose(1, 2)
        return q, k, v

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch, query_length, width = query.shape
        q, k, v = self._qkv(query, memory)
        attended = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, query_length, width)
        return self.output(attended)

    def probabilities(
        self, query: torch.Tensor, memory: torch.Tensor,
    ) -> torch.Tensor:
        q, k, _ = self._qkv(query, memory)
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        return logits.float().softmax(dim=-1).to(query.dtype)


class ControlDecoderBlock(nn.Module):
    """Pre-RMSNorm query self-attention, cross-attention, and SwiGLU FFN."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        feedforward_dim: int,
        *,
        dropout: float = 0.0,
        query_self_attention: bool = True,
    ) -> None:
        super().__init__()
        if feedforward_dim < d_model:
            raise ValueError("control decoder feedforward width cannot be below d_model")
        self.query_self_attention = bool(query_self_attention)
        self.self_norm: nn.Module | None = None
        self.self_attention: nn.Module | None = None
        if self.query_self_attention:
            self.self_norm = nn.RMSNorm(d_model, elementwise_affine=False)
            self.self_attention = BiasedSelfAttention(d_model, heads, dropout)
        self.query_norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.memory_norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.cross_attention = CrossAttention(d_model, heads, dropout)
        self.feedforward_norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.feedforward_input = nn.Linear(d_model, 2 * feedforward_dim)
        self.feedforward_output = nn.Linear(feedforward_dim, d_model)
        self.residual_dropout = nn.Dropout(float(dropout))

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor,
    ) -> torch.Tensor:
        if self.self_attention is not None and self.self_norm is not None:
            query = query + self.residual_dropout(
                self.self_attention(self.self_norm(query))
            )
        query = query + self.residual_dropout(self.cross_attention(
            self.query_norm(query), self.memory_norm(memory),
        ))
        gate, content = self.feedforward_input(
            self.feedforward_norm(query)
        ).chunk(2, dim=-1)
        return query + self.residual_dropout(
            self.feedforward_output(F.silu(gate) * content)
        )

    @torch.no_grad()
    def cross_attention_probabilities(
        self, query: torch.Tensor, memory: torch.Tensor,
    ) -> torch.Tensor:
        return self.cross_attention.probabilities(
            self.query_norm(query), self.memory_norm(memory),
        )


class ControlQueryDecoder(nn.Module):
    """Learned-query decoder that reduces all encoded tokens to one policy latent."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        feedforward_dim: int,
        *,
        depth: int,
        queries: int,
        dropout: float = 0.0,
        initialization_std: float = 0.02,
    ) -> None:
        super().__init__()
        if depth < 1 or queries < 1 or initialization_std <= 0.0:
            raise ValueError("invalid control-query decoder dimensions")
        self.query_count = int(queries)
        self.query = nn.Parameter(torch.empty(1, self.query_count, d_model))
        nn.init.trunc_normal_(self.query, std=float(initialization_std))
        self.blocks = nn.ModuleList(
            ControlDecoderBlock(
                d_model, heads, feedforward_dim,
                dropout=dropout, query_self_attention=True,
            )
            for _ in range(int(depth))
        )
        self.output_norm = nn.RMSNorm(d_model)
        self.fusion = SwiGLUProjection(self.query_count * d_model, d_model)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(memory.shape[0], -1, -1)
        for block in self.blocks:
            query = block(query, memory)
        query = self.output_norm(query)
        return self.fusion(query.flatten(start_dim=1))[:, None]

    @torch.no_grad()
    def attention_maps(self, memory: torch.Tensor) -> list[torch.Tensor]:
        query = self.query.expand(memory.shape[0], -1, -1)
        maps: list[torch.Tensor] = []
        for block in self.blocks:
            maps.append(block.cross_attention_probabilities(query, memory))
            query = block(query, memory)
        return maps


class AdaRMSNormSwiGLUBlock(nn.Module):
    """Pre-RMSNorm attention/SwiGLU block with optional AdaLN-Zero gates."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        feedforward_dim: int,
        *,
        dropout: float = 0.0,
        adaln_zero: bool = True,
    ) -> None:
        super().__init__()
        if feedforward_dim < d_model:
            raise ValueError("SwiGLU feedforward width cannot be below d_model")
        self.adaln_zero = bool(adaln_zero)
        self.attention_norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.attention = BiasedSelfAttention(d_model, heads, dropout)
        self.feedforward_norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.feedforward_input = nn.Linear(d_model, 2 * feedforward_dim)
        self.feedforward_output = nn.Linear(feedforward_dim, d_model)
        self.residual_dropout = nn.Dropout(float(dropout))
        self.modulation: nn.Module | None = None
        if self.adaln_zero:
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
            nn.init.zeros_(self.modulation[-1].weight)
            nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(
        value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor,
    ) -> torch.Tensor:
        return value * (1.0 + scale[:, None]) + shift[:, None]

    def forward(
        self,
        value: torch.Tensor,
        *,
        condition: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        is_causal: bool = False,
        linear_bias_slopes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation is None:
            shift_attention = scale_attention = shift_ff = scale_ff = None
            gate_attention = gate_ff = value.new_ones((len(value), value.shape[-1]))
        else:
            (
                shift_attention, scale_attention, gate_attention,
                shift_ff, scale_ff, gate_ff,
            ) = self.modulation(condition).chunk(6, dim=-1)
        attention_input = self.attention_norm(value)
        if shift_attention is not None and scale_attention is not None:
            attention_input = self._modulate(
                attention_input, shift_attention, scale_attention
            )
        value = value + gate_attention[:, None] * self.residual_dropout(
            self.attention(
                attention_input, attention_bias,
                is_causal=is_causal,
                linear_bias_slopes=linear_bias_slopes,
            )
        )
        ff_input = self.feedforward_norm(value)
        if shift_ff is not None and scale_ff is not None:
            ff_input = self._modulate(ff_input, shift_ff, scale_ff)
        gate, content = self.feedforward_input(ff_input).chunk(2, dim=-1)
        ff = self.feedforward_output(F.silu(gate) * content)
        return value + gate_ff[:, None] * self.residual_dropout(ff)

    def attention_probabilities(
        self,
        value: torch.Tensor,
        *,
        condition: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        is_causal: bool = False,
        linear_bias_slopes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expose the attention kernel without changing the forward contract."""

        attention_input = self.attention_norm(value)
        if self.modulation is not None:
            shift, scale, *_ = self.modulation(condition).chunk(6, dim=-1)
            attention_input = self._modulate(attention_input, shift, scale)
        return self.attention.probabilities(
            attention_input, attention_bias,
            is_causal=is_causal,
            linear_bias_slopes=linear_bias_slopes,
        )


def alibi_slopes(heads: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return monotone ALiBi slopes for arbitrary head counts."""

    if heads < 1:
        raise ValueError("ALiBi requires at least one head")
    closest_power = 2 ** math.floor(math.log2(heads))
    base = 2.0 ** (-8.0 / closest_power)
    primary = base ** torch.arange(1, closest_power + 1, device=device, dtype=dtype)
    if closest_power == heads:
        return primary
    extra_base = 2.0 ** (-4.0 / closest_power)
    extra = extra_base ** torch.arange(
        1, 2 * (heads - closest_power) + 1, 2, device=device, dtype=dtype
    )
    return torch.cat([primary, extra], dim=0)


def causal_attention_bias(
    length: int,
    heads: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    use_alibi: bool,
) -> torch.Tensor:
    query = torch.arange(length, device=device)
    key = torch.arange(length, device=device)
    age = (query[:, None] - key[None, :]).clamp_min(0).to(dtype)
    allowed = key[None, :] <= query[:, None]
    if use_alibi:
        slopes = alibi_slopes(heads, device=device, dtype=dtype)
        bias = -slopes[:, None, None] * age[None]
    else:
        bias = torch.zeros(heads, length, length, device=device, dtype=dtype)
    bias = bias.masked_fill(~allowed[None], torch.finfo(dtype).min)
    return bias[None]


class RouteSemanticStepTokenizer(nn.Module):
    """Tokenize one legacy route observation into state/control/gate tokens."""

    TASK_DIM = 19
    ROUTE_DIM = 13
    TRAILER_DIM = 6

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        *,
        layers: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
        adaln_zero: bool,
        fourier_features: int = 0,
        fourier_scale: float = 2.0,
    ) -> None:
        super().__init__()
        payload = int(input_dim) - self.TASK_DIM - self.TRAILER_DIM
        if payload < self.ROUTE_DIM or payload % self.ROUTE_DIM:
            raise ValueError("semantic tokenizer requires a legacy route observation")
        self.route_gates = payload // self.ROUTE_DIM
        self.d_model = int(d_model)
        self.task_projection = SwiGLUProjection(
            self.TASK_DIM, d_model,
            fourier_features=fourier_features, fourier_scale=fourier_scale, seed=11,
        )
        self.route_projection = SwiGLUProjection(
            self.ROUTE_DIM, d_model,
            fourier_features=fourier_features, fourier_scale=fourier_scale, seed=23,
        )
        self.control_projection = SwiGLUProjection(
            self.TRAILER_DIM, d_model,
            fourier_features=fourier_features, fourier_scale=fourier_scale, seed=37,
        )
        self.slot = nn.Parameter(torch.empty(1, 1, self.route_gates + 3, d_model))
        self.readout = nn.Parameter(torch.empty(1, 1, 1, d_model))
        nn.init.trunc_normal_(self.slot, std=0.02)
        nn.init.trunc_normal_(self.readout, std=0.02)
        self.blocks = nn.ModuleList(
            AdaRMSNormSwiGLUBlock(
                d_model, heads, feedforward_dim,
                dropout=dropout, adaln_zero=adaln_zero,
            )
            for _ in range(int(layers))
        )
        self.output_norm = nn.RMSNorm(d_model)

    def split(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, _ = features.shape
        task = features[..., : self.TASK_DIM]
        route_end = self.TASK_DIM + self.route_gates * self.ROUTE_DIM
        route = features[..., self.TASK_DIM:route_end].reshape(
            batch, steps, self.route_gates, self.ROUTE_DIM
        )
        control = features[..., route_end:]
        return task, route, control

    def _initial_tokens(
        self, features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if features.ndim != 3:
            raise ValueError("semantic route features must have shape [B,T,D]")
        batch, steps, _ = features.shape
        task, route, control = self.split(features)
        task_token = self.task_projection(task)
        control_token = self.control_projection(control)
        route_tokens = self.route_projection(route)
        semantic = torch.cat(
            [task_token[:, :, None], control_token[:, :, None], route_tokens], dim=2
        )
        # The readout has an observation-dependent direct path before the
        # zero-gated residual blocks, avoiding a constant-policy warmup.
        readout = semantic.mean(dim=2, keepdim=True) + self.readout
        tokens = torch.cat([readout, semantic], dim=2) + self.slot
        tokens = tokens.reshape(batch * steps, tokens.shape[2], self.d_model)
        condition = (task_token + control_token).reshape(batch * steps, self.d_model)
        return tokens, condition, batch, steps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens, condition, batch, steps = self._initial_tokens(features)
        for block in self.blocks:
            tokens = block(tokens, condition=condition)
        return self.output_norm(tokens[:, 0]).reshape(batch, steps, self.d_model)

    @torch.no_grad()
    def attention_maps(self, features: torch.Tensor) -> list[torch.Tensor]:
        """Return per-block [B,T,H,Q,K] semantic attention maps."""

        tokens, condition, batch, steps = self._initial_tokens(features)
        maps: list[torch.Tensor] = []
        for block in self.blocks:
            probabilities = block.attention_probabilities(
                tokens, condition=condition
            )
            maps.append(probabilities.reshape(
                batch, steps, probabilities.shape[1],
                probabilities.shape[2], probabilities.shape[3],
            ))
            tokens = block(tokens, condition=condition)
        return maps

    def active_gate_values(self, features: torch.Tensor) -> torch.Tensor:
        """Return the normalized active-gate record from the latest step."""

        _, route, _ = self.split(features)
        return route[:, -1, 0]


class ModernTemporalEncoder(nn.Module):
    """Causal pre-RMSNorm temporal encoder with optional fading memory."""

    def __init__(
        self,
        d_model: int,
        *,
        depth: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
        adaln_zero: bool,
        alibi: bool,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.alibi = bool(alibi)
        self.blocks = nn.ModuleList(
            AdaRMSNormSwiGLUBlock(
                d_model, heads, feedforward_dim,
                dropout=dropout, adaln_zero=adaln_zero,
            )
            for _ in range(int(depth))
        )
        self.output_norm = nn.RMSNorm(d_model)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        slopes = (
            alibi_slopes(
                self.heads, device=value.device, dtype=value.dtype
            ) if self.alibi else None
        )
        condition = value[:, -1]
        for block in self.blocks:
            value = block(
                value, condition=condition, is_causal=True,
                linear_bias_slopes=slopes,
            )
        return self.output_norm(value)

    @torch.no_grad()
    def attention_maps(self, value: torch.Tensor) -> list[torch.Tensor]:
        """Return per-block [B,H,T,T] causal temporal attention maps."""

        slopes = (
            alibi_slopes(
                self.heads, device=value.device, dtype=value.dtype
            ) if self.alibi else None
        )
        condition = value[:, -1]
        maps: list[torch.Tensor] = []
        for block in self.blocks:
            maps.append(block.attention_probabilities(
                value, condition=condition, is_causal=True,
                linear_bias_slopes=slopes,
            ))
            value = block(
                value, condition=condition, is_causal=True,
                linear_bias_slopes=slopes,
            )
        return maps


class UnifiedRouteTransformer(nn.Module):
    """One causal Transformer over every route token at every history step.

    Tokens are ordered as historical state/control pairs followed by only the
    six route records from the current observation.  The final route token is
    the policy latent, so ordinary causal SDPA lets it consume the complete
    state/action history and every nearer future reference. There is no
    invented readout token, per-step attention/pooling stage, or separate
    temporal Transformer.
    """

    TASK_DIM = 19
    ROUTE_DIM = 13
    TRAILER_DIM = 6

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        *,
        context_steps: int,
        depth: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
        adaln_zero: bool,
        alibi: bool,
        readout_mode: str = "final_route",
        readout_decoder_depth: int = 2,
        readout_decoder_queries: int = 2,
        readout_initialization_std: float = 0.02,
        input_rms_norm: bool = True,
        bidirectional: bool = False,
        route_feature_mode: str = "none",
        route_head_modulation: bool = False,
        separate_route_encoder: bool = False,
        plant_settings_dim: int = 0,
        fourier_features: int = 0,
        fourier_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.plant_settings_dim = int(plant_settings_dim)
        if self.plant_settings_dim and (separate_route_encoder or route_head_modulation):
            raise ValueError('Plant prefix is qualified for the canonical route transformer')
        payload = int(input_dim) - self.plant_settings_dim - self.TASK_DIM - self.TRAILER_DIM
        if payload < self.ROUTE_DIM or payload % self.ROUTE_DIM:
            raise ValueError("unified route Transformer requires a legacy route observation")
        if context_steps < 1 or depth < 1:
            raise ValueError("unified route Transformer requires positive context and depth")
        self.route_gates = payload // self.ROUTE_DIM
        self.context_steps = int(context_steps)
        self.d_model = int(d_model)
        self.heads = int(heads)
        self.alibi = bool(alibi)
        self.readout_mode = str(readout_mode)
        self.bidirectional = bool(bidirectional)
        self.route_feature_mode = str(route_feature_mode)
        # Runtime optimization, deliberately absent from checkpoint architecture.
        # The discarded historical route projections never enter attention.
        self.current_route_projection_only = False
        self.current_route_projection_training = False
        self.route_head_modulation = bool(route_head_modulation)
        self.separate_route_encoder = bool(separate_route_encoder)
        if self.route_feature_mode not in {'none','approach','body_targets'}:
            raise ValueError('invalid geometric route feature mode')
        if (self.route_head_modulation or self.separate_route_encoder) and readout_mode != 'action_token':
            raise ValueError('route conditioning arms require action-token readout')
        if self.separate_route_encoder and (bidirectional or alibi or adaln_zero):
            raise ValueError('separate route encoder uses causal temporal blocks and unmodulated route attention')
        if self.separate_route_encoder and self.route_head_modulation:
            raise ValueError('route encoder and head modulation are separate ablation arms')
        if self.route_feature_mode != 'none':
            self.register_buffer('geometry_mean',torch.zeros(input_dim))
            self.register_buffer('geometry_std',torch.ones(input_dim))
        if self.bidirectional and alibi:
            raise ValueError("bidirectional route encoder cannot use causal ALiBi")
        valid_readouts = {
            "final_route", "action_token", "action_token_cross_attention",
            "cross_attention_decoder",
        }
        if self.readout_mode not in valid_readouts:
            raise ValueError(
                f"unified readout mode must be one of {sorted(valid_readouts)}"
            )
        if readout_initialization_std <= 0.0:
            raise ValueError("readout initialization std must be positive")
        self.task_projection = SwiGLUProjection(
            self.TASK_DIM, d_model,
            fourier_features=fourier_features, fourier_scale=fourier_scale, seed=11,
            input_rms_norm=input_rms_norm,
        )
        self.route_projection = SwiGLUProjection(
            self.ROUTE_DIM + (9 if self.route_feature_mode == 'approach' else 0), d_model,
            fourier_features=fourier_features, fourier_scale=fourier_scale, seed=23,
            input_rms_norm=input_rms_norm,
        )
        self.control_projection = SwiGLUProjection(
            self.TRAILER_DIM, d_model,
            fourier_features=fourier_features, fourier_scale=fourier_scale, seed=37,
            input_rms_norm=input_rms_norm,
        )
        self.state_action_slot = nn.Parameter(torch.empty(1, 1, 2, d_model))
        self.route_slot = nn.Parameter(torch.empty(1, self.route_gates, d_model))
        self.temporal_position = nn.Parameter(
            torch.empty(1, self.context_steps, 1, d_model)
        )
        self.null_history = nn.Parameter(torch.empty(1, 1, 2, d_model))
        nn.init.trunc_normal_(self.state_action_slot, std=0.02)
        nn.init.trunc_normal_(self.route_slot, std=0.02)
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        nn.init.trunc_normal_(self.null_history, std=0.02)
        self.action_token: nn.Parameter | None = None
        if self.readout_mode in {"action_token", "action_token_cross_attention"}:
            self.action_token = nn.Parameter(torch.empty(1, 1, d_model))
            nn.init.trunc_normal_(
                self.action_token, std=float(readout_initialization_std)
            )
        self.blocks = nn.ModuleList(
            AdaRMSNormSwiGLUBlock(
                d_model, heads, feedforward_dim,
                dropout=dropout, adaln_zero=adaln_zero,
            )
            for _ in range(int(depth))
        )
        self.output_norm = nn.RMSNorm(d_model)
        self.route_encoder = None
        self.route_cross_readout = None
        self.route_summary_attention = None
        self.route_modulation = None
        if self.separate_route_encoder:
            self.route_encoder = AdaRMSNormSwiGLUBlock(d_model,heads,feedforward_dim,dropout=dropout,adaln_zero=False)
            self.route_cross_readout = ControlDecoderBlock(d_model,heads,feedforward_dim,dropout=dropout,query_self_attention=False)
        if self.route_head_modulation:
            self.route_summary_attention = CrossAttention(d_model,heads,dropout)
            self.route_modulation = nn.Linear(d_model,2*d_model)
            nn.init.zeros_(self.route_modulation.weight)
            nn.init.zeros_(self.route_modulation.bias)
        self.action_token_refinement: ControlDecoderBlock | None = None
        if self.readout_mode == "action_token_cross_attention":
            self.action_token_refinement = ControlDecoderBlock(
                d_model, heads, feedforward_dim,
                dropout=dropout, query_self_attention=False,
            )
        self.control_decoder: ControlQueryDecoder | None = None
        if self.readout_mode == "cross_attention_decoder":
            self.control_decoder = ControlQueryDecoder(
                d_model, heads, feedforward_dim,
                depth=readout_decoder_depth,
                queries=readout_decoder_queries,
                dropout=dropout,
                initialization_std=readout_initialization_std,
            )
        # Initialize new modules after the original backbone to preserve its
        # scratch seed/weights in the paired plant-observation experiment.
        self.settings_projection = None
        self.settings_film = None
        self.register_parameter('settings_slot', None)
        if self.plant_settings_dim:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed((torch.initial_seed() + 21764) % (2**63))
                self.settings_projection = SwiGLUProjection(self.plant_settings_dim, d_model,
                    input_rms_norm=False, fourier_features=0)
                self.settings_slot = nn.Parameter(torch.zeros(1, 1, d_model))
                self.settings_film = nn.Linear(d_model, 2*d_model)
                nn.init.zeros_(self.settings_film.weight)
                nn.init.zeros_(self.settings_film.bias)

    def split(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, _ = features.shape
        task = features[..., : self.TASK_DIM]
        route_end = self.TASK_DIM + self.route_gates * self.ROUTE_DIM
        route = features[..., self.TASK_DIM:route_end].reshape(
            batch, steps, self.route_gates, self.ROUTE_DIM
        )
        control = features[..., route_end:route_end + self.TRAILER_DIM]
        return task, route, control

    def _tokens(
        self,
        features: torch.Tensor,
        *,
        drop_old_history: torch.Tensor | bool | None = None,
        embedding_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if features.ndim != 3:
            raise ValueError("unified route features must have shape [B,T,D]")
        batch, steps, _ = features.shape
        if steps > self.context_steps:
            raise ValueError("unified route history exceeds configured context")
        task, route, control = self.split(features)
        task_token = self.task_projection(task)
        control_token = self.control_projection(control)
        # Keep DAgger/BF16 training GEMM shapes unchanged by default. Strict-FP32
        # PPO can explicitly avoid projecting historical route rows which are
        # discarded below; the runtime update-kernel audit checks likelihoods.
        if self.current_route_projection_only and (
            not torch.is_grad_enabled() or self.current_route_projection_training
        ):
            route_tokens = self.route_projection(self.geometric_routes(
                features[:, -1:], route[:, -1:]))
        else:
            route_tokens = self.route_projection(self.geometric_routes(features, route))
        state_action = torch.stack([task_token, control_token], dim=2)
        state_action = state_action + self.state_action_slot
        if not self.alibi:
            state_action = state_action + self.temporal_position[:, -steps:]
        if drop_old_history is not None and steps > 1:
            if isinstance(drop_old_history, bool):
                mask = torch.full(
                    (batch, 1, 1, 1), drop_old_history,
                    dtype=torch.bool, device=features.device,
                )
            else:
                mask = drop_old_history.to(device=features.device, dtype=torch.bool)
                if mask.shape == (batch,):
                    mask = mask[:, None, None, None]
                if mask.shape != (batch, 1, 1, 1):
                    raise ValueError("history mask must be scalar or [B]")
            null = self.null_history.expand(batch, steps - 1, -1, -1)
            if not self.alibi:
                null = null + self.temporal_position[:, -(steps - 1):]
            older = torch.where(mask, null, state_action[:, :-1])
            state_action = torch.cat([older, state_action[:, -1:]], dim=1)
        current_route = route_tokens[:, -1] + self.route_slot
        tokens = torch.cat([
            state_action.reshape(batch, 2 * steps, self.d_model),
            current_route,
        ], dim=1)
        if embedding_dropout > 0.0:
            tokens = F.dropout(tokens, p=float(embedding_dropout), training=True)
        if self.action_token is not None:
            tokens = torch.cat([
                tokens, self.action_token.expand(batch, -1, -1),
            ], dim=1)
        condition = task_token[:, -1] + control_token[:, -1]
        if self.settings_projection is not None:
            settings = self.settings_projection(features[:, -1, -self.plant_settings_dim:])
            scale, shift = self.settings_film(settings).chunk(2, dim=-1)
            tokens = tokens * (1 + scale[:, None]) + shift[:, None]
            tokens = torch.cat([settings[:, None] + self.settings_slot, tokens], dim=1)
            condition = condition + settings
        return tokens, condition, batch, steps

    def geometric_routes(self, features, normalized_route):
        if self.route_feature_mode == 'none':
            return normalized_route
        raw=features*self.geometry_std+self.geometry_mean
        task, route, _=self.split(raw)
        delta=route[...,:3]-task[...,:3].unsqueeze(-2)
        if self.route_feature_mode == 'body_targets':
            first=task[...,6:9];second=task[...,9:12]
            rotation=torch.stack([first,second,torch.cross(first,second,dim=-1)],dim=-1)
            # Row-vector transform G -> B: v_B = v_G @ R_GB.
            body_delta=torch.einsum('btni,btij->btnj',delta,rotation)/5.
            body_normal=torch.einsum('btni,btij->btnj',route[...,3:6],rotation)
            body_up=torch.einsum('btni,btij->btnj',route[...,6:9],rotation)
            return torch.cat([body_delta,body_normal,body_up,normalized_route[...,9:]],dim=-1)
        normal=route[...,3:6];up=route[...,6:9]
        rotation=torch.stack([normal,torch.cross(up,normal,dim=-1),up],dim=-1)
        vehicle_relative=-delta
        local_position=torch.einsum('btni,btnij->btnj',vehicle_relative,rotation)/5.
        local_velocity=torch.einsum('bti,btnij->btnj',task[...,3:6],rotation)/10.
        following=torch.cat([route[...,1:,:3],route[...,-1:,:3]],dim=-2)-route[...,:3]
        local_exit=torch.einsum('btni,btnij->btnj',following,rotation)/5.
        return torch.cat([normalized_route,local_position,local_velocity,local_exit],dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        *,
        drop_old_history: torch.Tensor | bool | None = None,
        embedding_dropout: float = 0.0,
    ) -> torch.Tensor:
        tokens, condition, batch, steps = self._tokens(
            features,
            drop_old_history=drop_old_history,
            embedding_dropout=embedding_dropout,
        )
        slopes = (
            alibi_slopes(self.heads, device=tokens.device, dtype=tokens.dtype)
            if self.alibi else None
        )
        route_memory = None
        if self.separate_route_encoder:
            route_memory=tokens[:,2*steps:2*steps+self.route_gates]
            route_memory=self.route_encoder(route_memory,condition=condition,is_causal=False)
            tokens=torch.cat([tokens[:,:2*steps],tokens[:,-1:]],dim=1)
        for block in self.blocks:
            tokens = block(
                tokens, condition=condition, is_causal=not self.bidirectional,
                linear_bias_slopes=slopes,
            )
        tokens = self.output_norm(tokens)
        prefix = int(bool(self.plant_settings_dim))
        state_action_outputs = tokens[:, prefix:prefix + 2 * steps].reshape(
            batch, steps, 2, self.d_model
        )[:, :, 1]
        if self.readout_mode == "final_route":
            readout = tokens[:, -1:]
        elif self.readout_mode == "action_token":
            readout = tokens[:, -1:]
        elif self.readout_mode == "action_token_cross_attention":
            if self.action_token_refinement is None:
                raise RuntimeError("action-token cross-attention module is missing")
            readout = self.action_token_refinement(tokens[:, -1:], tokens[:, :-1])
        else:
            if self.control_decoder is None:
                raise RuntimeError("control-query decoder is missing")
            readout = self.control_decoder(tokens)
        if self.separate_route_encoder:
            readout=self.route_cross_readout(readout,route_memory)
        if self.route_head_modulation:
            memory=tokens[:,2*steps:2*steps+self.route_gates]
            summary=self.route_summary_attention(readout,memory)
            gamma,beta=self.route_modulation(summary).chunk(2,dim=-1)
            readout=readout*(1+gamma)+beta
        # Preserve the historical [B,T,D] interface while replacing only the
        # latest/deployed latent with the selected global readout.
        return torch.cat([state_action_outputs[:, :-1], readout], dim=1)

    @torch.no_grad()
    def attention_maps(self, features: torch.Tensor) -> list[torch.Tensor]:
        tokens, condition, _, _ = self._tokens(features)
        if self.separate_route_encoder:
            steps = features.shape[1]
            tokens = torch.cat([tokens[:, :2*steps], tokens[:, -1:]], dim=1)
        slopes = (
            alibi_slopes(self.heads, device=tokens.device, dtype=tokens.dtype)
            if self.alibi else None
        )
        maps: list[torch.Tensor] = []
        for block in self.blocks:
            maps.append(block.attention_probabilities(
                tokens, condition=condition, is_causal=not self.bidirectional,
                linear_bias_slopes=slopes,
            ))
            tokens = block(
                tokens, condition=condition, is_causal=not self.bidirectional,
                linear_bias_slopes=slopes,
            )
        return maps

    @torch.no_grad()
    def readout_attention_maps(self, features: torch.Tensor) -> list[torch.Tensor]:
        """Return post-encoder cross-attention maps for readout diagnostics."""

        tokens, condition, _, _ = self._tokens(features)
        route_memory = None
        if self.separate_route_encoder:
            steps = features.shape[1]
            route_memory = self.route_encoder(tokens[:, 2*steps:2*steps+self.route_gates], condition=condition, is_causal=False)
            tokens = torch.cat([tokens[:, :2*steps], tokens[:, -1:]], dim=1)
        slopes = (
            alibi_slopes(self.heads, device=tokens.device, dtype=tokens.dtype)
            if self.alibi else None
        )
        for block in self.blocks:
            tokens = block(
                tokens, condition=condition, is_causal=not self.bidirectional,
                linear_bias_slopes=slopes,
            )
        tokens = self.output_norm(tokens)
        if self.separate_route_encoder:
            return [self.route_cross_readout.cross_attention_probabilities(tokens[:, -1:], route_memory)]
        if self.route_head_modulation:
            return [self.route_summary_attention.probabilities(tokens[:, -1:], tokens[:, 2*features.shape[1]:2*features.shape[1]+self.route_gates])]
        if self.readout_mode == "action_token_cross_attention":
            if self.action_token_refinement is None:
                raise RuntimeError("action-token cross-attention module is missing")
            return [self.action_token_refinement.cross_attention_probabilities(
                tokens[:, -1:], tokens[:, :-1],
            )]
        if self.readout_mode == "cross_attention_decoder":
            if self.control_decoder is None:
                raise RuntimeError("control-query decoder is missing")
            return self.control_decoder.attention_maps(tokens)
        return []

    def active_gate_values(self, features: torch.Tensor) -> torch.Tensor:
        _, route, _ = self.split(features)
        return route[:, -1, 0]


class SwiGLUActionMLP(nn.Module):
    """Gated replacement for the legacy linear/LeakyReLU action projector."""

    def __init__(
        self, input_dim: int, hidden_dim: int, depth: int, output_dim: int,
        *, output_scale: float = 1.0e-3,
    ) -> None:
        super().__init__()
        widths = [int(input_dim)] + [int(hidden_dim)] * int(depth)
        self.norms = nn.ModuleList(nn.RMSNorm(width) for width in widths[:-1])
        self.inputs = nn.ModuleList(
            nn.Linear(source, 2 * target)
            for source, target in zip(widths[:-1], widths[1:], strict=True)
        )
        self.output = nn.Linear(widths[-1], int(output_dim))
        nn.init.uniform_(self.output.weight, -float(output_scale), float(output_scale))
        nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for norm, projection in zip(self.norms, self.inputs, strict=True):
            gate, content = projection(norm(value)).chunk(2, dim=-1)
            value = F.silu(gate) * content
        return self.output(value)
