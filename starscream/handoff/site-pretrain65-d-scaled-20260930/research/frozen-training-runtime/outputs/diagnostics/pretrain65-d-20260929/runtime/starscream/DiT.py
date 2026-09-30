"""Flow-matching Diffusion Transformer action head for CTBR chunks."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from .attention import scaled_dot_product_attention


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, *, cross: bool = False) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.cross = cross

    def forward(self, query: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        context = query if context is None else context
        batch, q_length, width = query.shape
        k_length = context.shape[1]
        q = self.q(query).view(batch, q_length, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(context).chunk(2, -1)
        k = k.view(batch, k_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, k_length, self.n_heads, self.head_dim).transpose(1, 2)
        output = scaled_dot_product_attention(q, k, v)
        return self.out(output.transpose(1, 2).contiguous().view(batch, q_length, width))


class DiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attention = Attention(d_model, n_heads)
        self.cross_attention = Attention(d_model, n_heads, cross=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(approximate="tanh"), nn.Linear(4 * d_model, d_model)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 9 * d_model))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        values = self.modulation(condition).chunk(9, -1)
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = values
        mod = lambda value, scale, bias: value * (1 + scale[:, None]) + bias[:, None]
        x = x + g1[:, None] * self.self_attention(mod(self.norm1(x), s1, b1))
        x = x + g2[:, None] * self.cross_attention(mod(self.norm2(x), s2, b2), context)
        x = x + g3[:, None] * self.mlp(mod(self.norm3(x), s3, b3))
        return x


class FlowDiTActor(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int = 384,
        d_model: int = 256,
        n_heads: int = 8,
        depth: int = 6,
        action_horizon: int = 8,
        action_dim: int = 4,
        source_noise: float = 0.35,
    ) -> None:
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.source_noise = float(source_noise)
        self.action_projection = nn.Linear(action_dim, d_model)
        self.context_projection = nn.Linear(context_dim, d_model)
        self.position = nn.Parameter(torch.randn(action_horizon, d_model) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(64, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList(DiTBlock(d_model, n_heads) for _ in range(depth))
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.velocity = nn.Linear(d_model, action_dim)
        nn.init.zeros_(self.velocity.weight)
        nn.init.zeros_(self.velocity.bias)

    @staticmethod
    def time_features(time: torch.Tensor, width: int = 64) -> torch.Tensor:
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(width // 2, device=time.device) / (width // 2 - 1)
        )
        angles = time[:, None] * frequencies[None] * 2 * math.pi
        return torch.cat([angles.sin(), angles.cos()], -1)

    def forward(self, noisy_actions: torch.Tensor, flow_time: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if noisy_actions.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError("noisy action chunk has the wrong shape")
        condition = self.time_mlp(self.time_features(flow_time))
        x = self.action_projection(noisy_actions) + self.position[None]
        context = self.context_projection(context)
        for block in self.blocks:
            x = block(x, context, condition)
        return self.velocity(self.final_norm(x))

    def flow_training_pair(self, target: torch.Tensor, previous_action: torch.Tensor):
        source = previous_action[:, None].expand_as(target) + self.source_noise * torch.randn_like(target)
        time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        noisy = source + time[:, None, None] * (target - source)
        return noisy, time, target - source

    @torch.no_grad()
    def sample(self, context: torch.Tensor, previous_action: torch.Tensor, *, steps: int = 4, method: str = "heun") -> torch.Tensor:
        if steps < 1 or method not in {"euler", "heun"}:
            raise ValueError("steps must be positive and method must be euler or heun")
        x = previous_action[:, None].expand(-1, self.action_horizon, -1).clone()
        x.add_(self.source_noise * torch.randn_like(x))
        dt = 1.0 / steps
        for index in range(steps):
            time = x.new_full((x.shape[0],), index / steps)
            velocity = self(x, time, context)
            if method == "heun" and index + 1 < steps:
                proposal = x + dt * velocity
                next_time = x.new_full((x.shape[0],), (index + 1) / steps)
                velocity = 0.5 * (velocity + self(proposal, next_time, context))
            x = x + dt * velocity
        return x.clamp(-1, 1)

    @staticmethod
    def denormalize_ctbr(action: torch.Tensor) -> torch.Tensor:
        return torch.cat([(action[..., :1] + 1.0) * 15.0, action[..., 1:] * 6.0], -1)


def repeat_first_history(tokens: torch.Tensor, valid_steps: torch.Tensor | int) -> torch.Tensor:
    """Right-align history and repeat its first available step into left padding."""

    if tokens.ndim < 3:
        raise ValueError("history tokens must have shape (B,T,...)")
    batch, steps = tokens.shape[:2]
    valid = torch.as_tensor(valid_steps, device=tokens.device, dtype=torch.long)
    if valid.ndim == 0:
        valid = valid.expand(batch)
    if valid.shape != (batch,) or torch.any(valid < 1) or torch.any(valid > steps):
        raise ValueError("valid_steps must provide one value in [1,T] per batch item")
    result = torch.empty_like(tokens)
    for index, count in enumerate(valid.tolist()):
        available = tokens[index, steps - count :]
        padding = available[:1].expand(steps - count, *available.shape[1:])
        result[index] = torch.cat([padding, available], dim=0)
    return result


def _rotary_embedding(
    tensor: torch.Tensor, positions: torch.Tensor, *, base: float = 10000.0
) -> torch.Tensor:
    """Apply RoPE to ``(B,H,T,D)`` attention tensors at float coordinates."""

    width = tensor.shape[-1]
    rotary_width = width - width % 2
    if rotary_width < 2:
        return tensor
    positions = positions.to(device=tensor.device, dtype=torch.float32)
    inverse = torch.exp(
        -math.log(float(base))
        * torch.arange(0, rotary_width, 2, device=tensor.device, dtype=torch.float32)
        / rotary_width
    )
    angles = positions[:, None] * inverse[None]
    cosine = angles.cos().to(tensor.dtype)[None, None]
    sine = angles.sin().to(tensor.dtype)[None, None]
    rotary, remainder = tensor[..., :rotary_width], tensor[..., rotary_width:]
    even, odd = rotary[..., 0::2], rotary[..., 1::2]
    rotated = torch.stack(
        [even * cosine - odd * sine, even * sine + odd * cosine], dim=-1
    ).flatten(-2)
    return torch.cat([rotated, remainder], dim=-1)


class RotaryAttention(nn.Module):
    """Self/cross attention with explicit RoPE coordinates for both token banks."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model) // self.n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_positions: torch.Tensor,
        context_positions: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, query_length, width = query.shape
        context_length = context.shape[1]
        q = self.q(query).view(
            batch, query_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        k, v = self.kv(context).chunk(2, dim=-1)
        k = k.view(batch, context_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, context_length, self.n_heads, self.head_dim).transpose(1, 2)
        q = _rotary_embedding(q, query_positions)
        k = _rotary_embedding(k, context_positions)
        attended = scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        return self.out(attended.transpose(1, 2).contiguous().view(batch, query_length, width))


class TokenHistoryBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(d_model)
        self.attention = RotaryAttention(d_model, n_heads)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio) * d_model),
            nn.SiLU(),
            nn.Linear(int(mlp_ratio) * d_model, d_model),
        )

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.attention_norm(x)
        x = x + self.attention(
            normalized, normalized, positions, positions, attention_mask=attention_mask
        )
        return x + self.mlp(self.mlp_norm(x))


class ShortcutFlowBlock(nn.Module):
    """DiT block with AdaLN-Zero gates for flow time and shortcut size."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.self_norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.cross_norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.mlp_norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.self_attention = RotaryAttention(d_model, n_heads)
        self.cross_attention = RotaryAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio) * d_model),
            nn.SiLU(),
            nn.Linear(int(mlp_ratio) * d_model, d_model),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 9 * d_model))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(
        value: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return value * (1.0 + scale[:, None]) + bias[:, None]

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        condition: torch.Tensor,
        action_positions: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.modulation(condition).chunk(9, -1)
        self_input = self._modulate(self.self_norm(x), s1, b1)
        x = x + g1[:, None] * self.self_attention(
            self_input, self_input, action_positions, action_positions
        )
        cross_input = self._modulate(self.cross_norm(x), s2, b2)
        x = x + g2[:, None] * self.cross_attention(
            cross_input, context, action_positions, context_positions
        )
        x = x + g3[:, None] * self.mlp(
            self._modulate(self.mlp_norm(x), s3, b3)
        )
        return x


class TokenizerFlowPolicy(nn.Module):
    """Direct multimodal CTBR policy over frozen observation/action tokens.

    This is deliberately not a world model. Three macro-steps of observation
    latents (visual plus fused state/route tokens) and causal applied-action
    tokens are encoded as a short sequence. A shortcut-conditioned flow DiT
    then generates one three-command CTBR chunk.
    """

    def __init__(
        self,
        *,
        observation_token_dim: int = 128,
        action_token_dim: int = 128,
        observation_tokens: int = 7,
        visual_tokens: int = 4,
        action_tokens: int = 2,
        history_steps: int = 3,
        action_horizon: int = 3,
        action_dim: int = 4,
        d_model: int = 384,
        n_heads: int = 8,
        context_depth: int = 4,
        flow_depth: int = 8,
        mlp_ratio: int = 4,
        k_max: int = 8,
        source_mode: str = "gaussian",
        source_noise: float = 1.0,
        shortcut_step_sizes: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        super().__init__()
        if observation_tokens < visual_tokens:
            raise ValueError("visual_tokens cannot exceed observation_tokens")
        if history_steps < 1 or action_horizon < 1:
            raise ValueError("history_steps and action_horizon must be positive")
        if k_max < 1 or k_max & (k_max - 1):
            raise ValueError("k_max must be a positive power of two")
        if source_mode not in {"gaussian", "previous_action"}:
            raise ValueError("source_mode must be gaussian or previous_action")
        self.observation_tokens = int(observation_tokens)
        self.visual_tokens = int(visual_tokens)
        self.action_tokens = int(action_tokens)
        self.history_steps = int(history_steps)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.k_max = int(k_max)
        self.source_mode = str(source_mode)
        self.source_noise = float(source_noise)
        self.shortcut_step_sizes = tuple(
            float(value) for value in (shortcut_step_sizes or ())
        )
        if self.shortcut_step_sizes and (
            any(
                not math.isfinite(value) or value <= 0.0 or value > 1.0
                for value in self.shortcut_step_sizes
            )
            or tuple(sorted(set(self.shortcut_step_sizes)))
            != self.shortcut_step_sizes
        ):
            raise ValueError(
                "shortcut_step_sizes must be unique increasing values in (0,1]"
            )
        self.tokens_per_step = self.observation_tokens + self.action_tokens

        self.observation_projection = nn.Linear(observation_token_dim, d_model)
        self.action_token_projection = nn.Linear(action_token_dim, d_model)
        self.token_slot = nn.Parameter(torch.randn(self.tokens_per_step, d_model) * 0.02)
        self.modality_embedding = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.context_blocks = nn.ModuleList(
            TokenHistoryBlock(d_model, n_heads, mlp_ratio)
            for _ in range(int(context_depth))
        )
        self.context_norm = nn.RMSNorm(d_model)

        self.noisy_action_projection = nn.Linear(action_dim, d_model)
        self.action_slot = nn.Parameter(torch.randn(action_horizon, d_model) * 0.02)
        self.condition_mlp = nn.Sequential(
            nn.Linear(128, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.flow_blocks = nn.ModuleList(
            ShortcutFlowBlock(d_model, n_heads, mlp_ratio)
            for _ in range(int(flow_depth))
        )
        self.final_norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.velocity = nn.Linear(d_model, action_dim)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.velocity.weight)
        nn.init.zeros_(self.velocity.bias)

        macro = torch.arange(self.history_steps, dtype=torch.float32)
        context_positions = macro.repeat_interleave(self.tokens_per_step)
        action_positions = self.history_steps + (
            torch.arange(self.action_horizon, dtype=torch.float32) + 0.5
        ) / self.action_horizon
        context_macro = macro.repeat_interleave(self.tokens_per_step)
        causal = context_macro[:, None] >= context_macro[None, :]
        self.register_buffer("context_positions", context_positions, persistent=False)
        self.register_buffer("action_positions", action_positions, persistent=False)
        self.register_buffer("context_causal_mask", causal, persistent=False)

    @staticmethod
    def time_features(value: torch.Tensor, width: int = 64) -> torch.Tensor:
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(width // 2, device=value.device, dtype=torch.float32)
            / max(1, width // 2 - 1)
        )
        angles = value.float()[:, None] * frequencies[None] * 2.0 * math.pi
        return torch.cat([angles.sin(), angles.cos()], dim=-1).to(value.dtype)

    def encode_context(
        self, observation_tokens: torch.Tensor, action_tokens: torch.Tensor
    ) -> torch.Tensor:
        expected_observation = (
            self.history_steps, self.observation_tokens, self.observation_projection.in_features
        )
        expected_action = (
            self.history_steps, self.action_tokens, self.action_token_projection.in_features
        )
        if observation_tokens.shape[1:] != expected_observation:
            raise ValueError(
                f"observation tokens must have shape (B,{expected_observation})"
            )
        if action_tokens.shape[1:] != expected_action:
            raise ValueError(f"action tokens must have shape (B,{expected_action})")
        observation = self.observation_projection(observation_tokens)
        actions = self.action_token_projection(action_tokens)
        tokens = torch.cat([observation, actions], dim=2)
        modality_ids = torch.cat([
            torch.zeros(self.visual_tokens, dtype=torch.long, device=tokens.device),
            torch.ones(
                self.observation_tokens - self.visual_tokens,
                dtype=torch.long, device=tokens.device,
            ),
            torch.full(
                (self.action_tokens,), 2, dtype=torch.long, device=tokens.device
            ),
        ])
        tokens = tokens + self.token_slot[None, None]
        tokens = tokens + self.modality_embedding[modality_ids][None, None]
        tokens = tokens.flatten(1, 2)
        for block in self.context_blocks:
            tokens = block(tokens, self.context_positions, self.context_causal_mask)
        return self.context_norm(tokens)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor,
        step_size: torch.Tensor,
        observation_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_context(observation_tokens, action_tokens)
        return self.velocity_from_context(noisy_actions, flow_time, step_size, context)

    def velocity_from_context(
        self,
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor,
        step_size: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_actions.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError("noisy action chunk has the wrong shape")
        if context.shape[1:] != (
            self.history_steps * self.tokens_per_step, self.d_model
        ):
            raise ValueError("encoded policy context has the wrong shape")
        condition = self.condition_mlp(torch.cat([
            self.time_features(flow_time), self.time_features(step_size)
        ], dim=-1))
        x = self.noisy_action_projection(noisy_actions) + self.action_slot[None]
        for block in self.flow_blocks:
            x = block(
                x, context, condition, self.action_positions, self.context_positions
            )
        scale, bias = self.final_modulation(condition).chunk(2, dim=-1)
        x = self.final_norm(x) * (1.0 + scale[:, None]) + bias[:, None]
        return self.velocity(x)

    def source(
        self, target: torch.Tensor, previous_action: torch.Tensor | None = None
    ) -> torch.Tensor:
        noise = self.source_noise * torch.randn_like(target)
        if self.source_mode == "gaussian":
            return noise
        if previous_action is None or previous_action.shape != target.shape[:1] + (self.action_dim,):
            raise ValueError("previous_action source requires shape (B,action_dim)")
        return previous_action[:, None].expand_as(target) + noise

    def shortcut_training_loss(
        self,
        target: torch.Tensor,
        observation_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        *,
        previous_action: torch.Tensor | None = None,
        direct_weight: float = 0.5,
        bootstrap_weight: float = 1.0,
        context: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Flow matching plus half-step self-distillation for large shortcuts."""

        batch = target.shape[0]
        if reduction not in {"mean", "none"}:
            raise ValueError("shortcut flow reduction must be mean or none")
        if self.shortcut_step_sizes:
            candidates = target.new_tensor(self.shortcut_step_sizes)
            step_index = torch.randint(
                0, len(candidates), (batch,), device=target.device
            )
            step_size = candidates[step_index]
            cells = torch.floor(
                (1.0 + 1.0e-6) / step_size
            ).to(torch.long).clamp_min(1)
        else:
            bins = int(math.log2(self.k_max)) + 1
            step_index = torch.randint(0, bins, (batch,), device=target.device)
            step_size = (2.0 ** step_index.to(target.dtype)) / self.k_max
            cells = (1.0 / step_size).to(torch.long)
        cell = (torch.rand(batch, device=target.device) * cells).to(torch.long)
        flow_time = cell.to(target.dtype) * step_size
        source = self.source(target, previous_action)
        noisy = source + flow_time[:, None, None] * (target - source)
        context = (
            self.encode_context(observation_tokens, action_tokens)
            if context is None else context
        )
        predicted = self.velocity_from_context(noisy, flow_time, step_size, context)
        direct_target = target - source
        direct_error = (predicted.float() - direct_target.float()).square().mean((1, 2))
        bootstrap_mask = step_index > 0
        bootstrap_error = torch.zeros_like(direct_error)
        if bootstrap_mask.any():
            half_step = step_size / 2.0
            with torch.no_grad():
                first = self.velocity_from_context(
                    noisy, flow_time, half_step, context
                )
                midpoint = noisy + half_step[:, None, None] * first
                second = self.velocity_from_context(
                    midpoint,
                    flow_time + half_step,
                    half_step,
                    context,
                )
                bootstrap_target = 0.5 * (first + second)
            bootstrap_error = (
                predicted.float() - bootstrap_target.float()
            ).square().mean((1, 2))
        loss_per_sample = float(direct_weight) * direct_error + torch.where(
            bootstrap_mask,
            float(bootstrap_weight) * bootstrap_error,
            (1.0 - float(direct_weight)) * direct_error,
        )
        endpoint = noisy.float() + (1.0 - flow_time)[:, None, None] * predicted.float()
        metrics = {
            "loss": loss_per_sample.mean().detach(),
            "flow_matching_loss": direct_error.mean().detach(),
            "shortcut_bootstrap_loss": (
                bootstrap_error[bootstrap_mask].mean().detach()
                if bootstrap_mask.any() else direct_error.new_zeros(())
            ),
            "shortcut_fraction": bootstrap_mask.float().mean().detach(),
            "endpoint_mse": (endpoint - target.float()).square().mean().detach(),
            "flow_time": flow_time.mean().detach(),
            "step_size": step_size.mean().detach(),
            "target_saturation": (target.abs() >= 0.98).float().mean().detach(),
        }
        return (
            loss_per_sample.mean() if reduction == "mean" else loss_per_sample,
            metrics,
        )

    def integrate(
        self,
        observation_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        *,
        steps: int = 4,
        method: str = "heun",
        previous_action: torch.Tensor | None = None,
        deterministic: bool = False,
        context: torch.Tensor | None = None,
        source_noise: float | None = None,
    ) -> torch.Tensor:
        """Integrate the learned flow while preserving model gradients.

        Training losses that supervise the command actually deployed must use
        this path.  ``sample`` is the inference-only wrapper below.
        """
        if steps < 1 or method not in {"euler", "heun"}:
            raise ValueError("steps must be positive and method must be euler or heun")
        batch = observation_tokens.shape[0]
        shape = (batch, self.action_horizon, self.action_dim)
        if deterministic:
            x = observation_tokens.new_zeros(shape)
            if self.source_mode == "previous_action":
                if previous_action is None:
                    raise ValueError("previous_action source requires previous_action")
                x = previous_action[:, None].expand(shape).clone()
        else:
            noise_scale = self.source_noise if source_noise is None else float(source_noise)
            if noise_scale < 0.0:
                raise ValueError("source_noise must be non-negative")
            x = noise_scale * torch.randn(
                shape, device=observation_tokens.device,
                dtype=observation_tokens.dtype,
            )
            if self.source_mode == "previous_action":
                if previous_action is None:
                    raise ValueError("previous_action source requires previous_action")
                x = x + previous_action[:, None]
        context = (
            self.encode_context(observation_tokens, action_tokens)
            if context is None else context
        )
        step_size = 1.0 / steps
        for index in range(steps):
            time = x.new_full((batch,), index * step_size)
            step = x.new_full((batch,), step_size)
            velocity = self.velocity_from_context(x, time, step, context)
            if method == "heun" and index + 1 < steps:
                proposal = x + step_size * velocity
                next_time = x.new_full((batch,), (index + 1) * step_size)
                correction = self.velocity_from_context(
                    proposal, next_time, step, context
                )
                velocity = 0.5 * (velocity + correction)
            x = x + step_size * velocity
        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample(
        self,
        observation_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        *,
        steps: int = 4,
        method: str = "heun",
        previous_action: torch.Tensor | None = None,
        deterministic: bool = False,
        context: torch.Tensor | None = None,
        source_noise: float | None = None,
    ) -> torch.Tensor:
        """Inference-only wrapper around the deployment integrator."""

        return self.integrate(
            observation_tokens,
            action_tokens,
            steps=steps,
            method=method,
            previous_action=previous_action,
            deterministic=deterministic,
            context=context,
            source_noise=source_noise,
        )
