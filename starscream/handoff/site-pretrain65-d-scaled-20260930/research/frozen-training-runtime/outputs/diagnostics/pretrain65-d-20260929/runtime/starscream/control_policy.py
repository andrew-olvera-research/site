"""Multi-rate causal observation representation and one-step flow controller."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .DiT import TokenizerFlowPolicy


@dataclass
class ControlRepresentationOutput:
    tokens: torch.Tensor
    pooled: torch.Tensor
    fast_sequence: torch.Tensor
    phase_logits: torch.Tensor
    gate_logits: torch.Tensor
    survival_logits: torch.Tensor


class SpatialMaskEncoder(nn.Module):
    """Retain four spatial gate-mask tokens for each slow visual frame."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        widths = (32, 64, 96, d_model)
        layers: list[nn.Module] = []
        source = 1
        for width in widths:
            layers.extend(
                [
                    # Sparse or fully blank gate masks make per-sample spatial
                    # normalization ill-conditioned.  The final token RMSNorm
                    # supplies scale control without amplifying blank frames.
                    nn.Conv2d(source, width, 3, stride=2, padding=1, bias=True),
                    nn.SiLU(),
                ]
            )
            source = width
        self.network = nn.Sequential(*layers)
        self.position = nn.Parameter(torch.randn(4, d_model) * 0.02)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask frames must have shape (B,1,H,W)")
        features = F.adaptive_avg_pool2d(self.network(mask), (2, 2))
        tokens = features.flatten(2).transpose(1, 2)
        return tokens + self.position.unsqueeze(0)


class MultiRateControlRepresentation(nn.Module):
    """Late-fuse fast dynamics, slow visual geometry, and route context.

    Raw 90 Hz state/action/timing samples are never pooled before a recurrent
    dynamics encoder sees them.  Visual masks are sampled at their slower rate
    and retain a 2x2 spatial layout.  A direct current-state token bypasses the
    recurrent bottleneck.
    """

    def __init__(
        self,
        *,
        task_state_dim: int = 19,
        timing_dim: int = 10,
        action_dim: int = 4,
        route_gates: int = 3,
        route_dim: int = 13,
        d_model: int = 192,
        visual_frames: int = 3,
        phase_classes: int = 5,
        temporal_depth: int = 0,
        temporal_heads: int = 6,
        temporal_context: int = 16,
        fusion_depth: int = 0,
        fusion_heads: int = 6,
    ) -> None:
        super().__init__()
        self.task_state_dim = int(task_state_dim)
        self.timing_dim = int(timing_dim)
        self.action_dim = int(action_dim)
        self.route_gates = int(route_gates)
        self.route_dim = int(route_dim)
        self.d_model = int(d_model)
        self.visual_frames = int(visual_frames)
        self.phase_classes = int(phase_classes)
        self.temporal_depth = int(temporal_depth)
        self.temporal_context = int(temporal_context)
        self.fusion_depth = int(fusion_depth)
        fast_dim = 2 * task_state_dim + 2 * action_dim + timing_dim
        self.fast_input = nn.Sequential(
            nn.LayerNorm(fast_dim), nn.Linear(fast_dim, d_model), nn.SiLU()
        )
        self.fast_gru = (
            nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
            if self.temporal_depth == 0 else None
        )
        if self.temporal_depth > 0:
            if d_model % int(temporal_heads) or self.temporal_context < 1:
                raise ValueError("temporal transformer dimensions are inconsistent")
            temporal_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=int(temporal_heads),
                dim_feedforward=4 * d_model, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.fast_transformer = nn.TransformerEncoder(
                temporal_layer, num_layers=self.temporal_depth,
                norm=nn.RMSNorm(d_model),
            )
            self.temporal_position = nn.Parameter(
                torch.randn(self.temporal_context, d_model) * 0.02
            )
        else:
            self.fast_transformer = None
            self.temporal_position = None
        self.current_state = nn.Sequential(
            nn.LayerNorm(task_state_dim + action_dim + timing_dim),
            nn.Linear(task_state_dim + action_dim + timing_dim, d_model),
            nn.SiLU(),
        )
        self.visual = SpatialMaskEncoder(d_model)
        self.visual_gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.route = nn.Sequential(
            nn.LayerNorm(route_dim), nn.Linear(route_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.route_position = nn.Parameter(torch.randn(route_gates, d_model) * 0.02)
        # gate-frame position/velocity, time since gate change, and cyclic gate id
        self.phase = nn.Sequential(
            nn.LayerNorm(9), nn.Linear(9, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        token_count = 4 + 2 + route_gates + 1
        if self.fusion_depth > 0:
            if d_model % int(fusion_heads):
                raise ValueError("fusion transformer dimensions are inconsistent")
            fusion_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=int(fusion_heads),
                dim_feedforward=4 * d_model, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.fusion_transformer = nn.TransformerEncoder(
                fusion_layer, num_layers=self.fusion_depth,
                norm=nn.RMSNorm(d_model),
            )
            self.fusion_position = nn.Parameter(
                torch.randn(token_count, d_model) * 0.02
            )
        else:
            self.fusion_transformer = None
            self.fusion_position = None
        self.token_norm = nn.RMSNorm(d_model)
        self.pool = nn.Sequential(
            nn.LayerNorm(token_count * d_model),
            nn.Linear(token_count * d_model, d_model),
            nn.SiLU(),
        )
        self.phase_head = nn.Linear(d_model, phase_classes)
        self.gate_head = nn.Linear(d_model, 1)
        self.survival_head = nn.Linear(d_model, 1)
        self.token_count = token_count

    @staticmethod
    def _visual_indices(history: int, count: int, device: torch.device) -> torch.Tensor:
        if history < 1 or count < 1:
            raise ValueError("history and visual frame count must be positive")
        count = min(history, count)
        return torch.linspace(0, history - 1, count, device=device).round().long()

    @staticmethod
    def _causal_phase_features(
        task_state: torch.Tensor, gate_index: torch.Tensor
    ) -> torch.Tensor:
        batch, history = gate_index.shape
        changes = torch.zeros_like(gate_index, dtype=torch.bool)
        changes[:, 1:] = gate_index[:, 1:] != gate_index[:, :-1]
        step = torch.arange(history, device=gate_index.device).view(1, -1).expand(batch, -1)
        last_change = torch.where(changes, step, torch.zeros_like(step)).cummax(dim=1).values
        since = (step[:, -1] - last_change[:, -1]).to(task_state.dtype)
        since = since / max(1, history - 1)
        gate = gate_index[:, -1].to(task_state.dtype)
        cyclic = torch.stack((torch.sin(gate), torch.cos(gate)), dim=-1)
        return torch.cat((task_state[:, -1, :6], since[:, None], cyclic), dim=-1)

    def forward(
        self,
        *,
        mask: torch.Tensor,
        task_state: torch.Tensor,
        previous_action: torch.Tensor,
        timing: torch.Tensor,
        route: torch.Tensor,
        gate_index: torch.Tensor,
    ) -> ControlRepresentationOutput:
        if task_state.ndim != 3 or task_state.shape[-1] != self.task_state_dim:
            raise ValueError("task_state has the wrong shape")
        batch, history = task_state.shape[:2]
        expected = (batch, history)
        if previous_action.shape != expected + (self.action_dim,):
            raise ValueError("previous_action has the wrong shape")
        if timing.shape != expected + (self.timing_dim,):
            raise ValueError("timing has the wrong shape")
        if mask.shape[:2] != expected or mask.ndim != 5:
            raise ValueError("mask must have shape (B,T,1,H,W)")
        if route.shape[:2] != expected or route.shape[-2:] != (
            self.route_gates, self.route_dim
        ):
            raise ValueError("route has the wrong shape")
        if gate_index.shape != expected:
            raise ValueError("gate_index must have shape (B,T)")

        state_delta = torch.zeros_like(task_state)
        action_delta = torch.zeros_like(previous_action)
        state_delta[:, 1:] = task_state[:, 1:] - task_state[:, :-1]
        action_delta[:, 1:] = previous_action[:, 1:] - previous_action[:, :-1]
        fast = torch.cat(
            (task_state, state_delta, previous_action, action_delta, timing), dim=-1
        )
        fast_embedded = self.fast_input(fast)
        if self.fast_transformer is None:
            fast_sequence, _ = self.fast_gru(fast_embedded)
        else:
            if history > self.temporal_context:
                raise ValueError("history exceeds temporal transformer context")
            fast_embedded = fast_embedded + self.temporal_position[:history].unsqueeze(0)
            causal_mask = torch.triu(
                torch.ones(history, history, device=fast.device, dtype=torch.bool),
                diagonal=1,
            )
            fast_sequence = self.fast_transformer(
                fast_embedded, mask=causal_mask, is_causal=True
            )
        recurrent_token = fast_sequence[:, -1]
        direct_token = self.current_state(
            torch.cat((task_state[:, -1], previous_action[:, -1], timing[:, -1]), dim=-1)
        )

        visual_indices = self._visual_indices(history, self.visual_frames, mask.device)
        visual_mask = mask.index_select(1, visual_indices)
        frames = visual_mask.shape[1]
        visual = self.visual(visual_mask.flatten(0, 1)).view(batch, frames, 4, self.d_model)
        visual = visual.permute(0, 2, 1, 3).reshape(batch * 4, frames, self.d_model)
        visual, _ = self.visual_gru(visual)
        visual_tokens = visual[:, -1].view(batch, 4, self.d_model)

        route_tokens = self.route(route[:, -1]) + self.route_position.unsqueeze(0)
        phase_token = self.phase(
            self._causal_phase_features(task_state, gate_index)
        ).unsqueeze(1)
        tokens = torch.cat(
            (
                visual_tokens,
                recurrent_token.unsqueeze(1),
                direct_token.unsqueeze(1),
                route_tokens,
                phase_token,
            ),
            dim=1,
        )
        if self.fusion_transformer is not None:
            tokens = self.fusion_transformer(
                tokens + self.fusion_position.unsqueeze(0)
            )
        tokens = self.token_norm(tokens)
        pooled = self.pool(tokens.flatten(1))
        return ControlRepresentationOutput(
            tokens=tokens,
            pooled=pooled,
            fast_sequence=fast_sequence,
            phase_logits=self.phase_head(pooled),
            gate_logits=self.gate_head(pooled).squeeze(-1),
            survival_logits=self.survival_head(pooled).squeeze(-1),
        )


class MultiRateFlowPolicy(nn.Module):
    """One-step shortcut-flow BC policy over the multi-rate representation."""

    def __init__(
        self,
        *,
        future_horizons: tuple[int, ...] | list[int] = (1, 2, 4, 8),
        representation: dict | None = None,
        flow: dict | None = None,
    ) -> None:
        super().__init__()
        representation = dict(representation or {})
        flow = dict(flow or {})
        self.representation = MultiRateControlRepresentation(**representation)
        self.future_horizons = tuple(int(value) for value in future_horizons)
        if not self.future_horizons or min(self.future_horizons) < 1:
            raise ValueError("future_horizons must contain positive offsets")
        d_model = self.representation.d_model
        self.future_head = nn.Sequential(
            nn.LayerNorm(d_model + 4),
            nn.Linear(d_model + 4, d_model),
            nn.SiLU(),
            nn.Linear(d_model, len(self.future_horizons) * self.representation.task_state_dim),
        )
        flow.setdefault("observation_token_dim", d_model)
        flow.setdefault("action_token_dim", d_model)
        flow.setdefault("observation_tokens", self.representation.token_count)
        flow.setdefault("visual_tokens", 4)
        flow.setdefault("action_tokens", 0)
        flow.setdefault("history_steps", 1)
        flow.setdefault("action_horizon", 1)
        flow.setdefault("action_dim", 4)
        flow.setdefault("source_mode", "previous_action")
        self.flow = TokenizerFlowPolicy(**flow)

    def encode(self, batch: dict[str, torch.Tensor], history: int) -> ControlRepresentationOutput:
        source = slice(0, int(history))
        return self.representation(
            mask=batch["mask"][:, source],
            task_state=batch["deployable_task_state"][:, source],
            previous_action=batch["previous_action"][:, source],
            timing=batch["timing"][:, source],
            route=batch["route"][:, source],
            gate_index=batch["gate_index"][:, source, 0],
        )

    def _empty_action_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.new_empty(tokens.shape[0], 1, 0, self.representation.d_model)

    def predict_future(
        self, pooled: torch.Tensor, applied_action: torch.Tensor
    ) -> torch.Tensor:
        prediction = self.future_head(torch.cat((pooled, applied_action), dim=-1))
        return prediction.view(
            len(pooled), len(self.future_horizons), self.representation.task_state_dim
        )

    def training_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        history: int,
        future_task_delta: torch.Tensor,
        phase_target: torch.Tensor,
        gate_target: torch.Tensor,
        survival_target: torch.Tensor,
        flow_direct_weight: float = 0.5,
        flow_bootstrap_weight: float = 1.0,
        future_weight: float = 0.25,
        phase_weight: float = 0.10,
        gate_weight: float = 0.05,
        survival_weight: float = 0.05,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_index = int(history) - 1
        encoded = self.encode(batch, history)
        target = batch["expert_action"][:, target_index : target_index + 1]
        previous = batch["previous_action"][:, target_index]
        observation = encoded.tokens.unsqueeze(1)
        empty_actions = self._empty_action_tokens(encoded.tokens)
        flow_loss, flow_metrics = self.flow.shortcut_training_loss(
            target,
            observation,
            empty_actions,
            previous_action=previous,
            direct_weight=flow_direct_weight,
            bootstrap_weight=flow_bootstrap_weight,
        )
        applied = batch["normalized_action"][:, target_index]
        future_prediction = self.predict_future(encoded.pooled, applied)
        future_loss = F.smooth_l1_loss(future_prediction.float(), future_task_delta.float(), beta=0.05)
        phase_loss = F.cross_entropy(encoded.phase_logits.float(), phase_target)
        gate_loss = F.binary_cross_entropy_with_logits(
            encoded.gate_logits.float(), gate_target.float()
        )
        survival_loss = F.binary_cross_entropy_with_logits(
            encoded.survival_logits.float(), survival_target.float()
        )
        loss = (
            flow_loss
            + float(future_weight) * future_loss
            + float(phase_weight) * phase_loss
            + float(gate_weight) * gate_loss
            + float(survival_weight) * survival_loss
        )
        metrics = {
            **flow_metrics,
            "total_loss": loss.detach(),
            "future_task_loss": future_loss.detach(),
            "phase_loss": phase_loss.detach(),
            "phase_accuracy": (encoded.phase_logits.argmax(-1) == phase_target).float().mean().detach(),
            "gate_loss": gate_loss.detach(),
            "survival_loss": survival_loss.detach(),
        }
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, torch.Tensor],
        *,
        history: int,
        steps: int = 1,
        method: str = "euler",
        deterministic: bool = True,
    ) -> torch.Tensor:
        encoded = self.encode(batch, history)
        target_index = int(history) - 1
        action = self.flow.sample(
            encoded.tokens.unsqueeze(1),
            self._empty_action_tokens(encoded.tokens),
            previous_action=batch["previous_action"][:, target_index],
            steps=steps,
            method=method,
            deterministic=deterministic,
        )
        return action[:, 0]


class LocalKnotOutcomeModel(nn.Module):
    """Short-horizon task-state predictor used as a frozen policy critic.

    The model is deliberately conditioned on observable state/route context and
    the complete three-command knot.  It is fitted only on expert transitions,
    then frozen before actor optimization so the actor cannot co-adapt a weak
    outcome model to make its own commands look successful.
    """

    def __init__(
        self,
        *,
        task_state_dim: int = 19,
        route_gates: int = 3,
        route_dim: int = 13,
        action_dim: int = 4,
        knot_steps: int = 3,
        horizons: tuple[int, ...] | list[int] = (1, 3, 6, 9),
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.task_state_dim = int(task_state_dim)
        self.route_gates = int(route_gates)
        self.route_dim = int(route_dim)
        self.action_dim = int(action_dim)
        self.knot_steps = int(knot_steps)
        self.horizons = tuple(int(value) for value in horizons)
        if self.knot_steps < 1 or not self.horizons or min(self.horizons) < 1:
            raise ValueError("knot_steps and outcome horizons must be positive")
        input_dim = (
            task_state_dim + route_gates * route_dim + action_dim
            + knot_steps * action_dim
        )
        output_dim = len(self.horizons) * task_state_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        task_state: torch.Tensor,
        route: torch.Tensor,
        previous_action: torch.Tensor,
        knot: torch.Tensor,
    ) -> torch.Tensor:
        if knot.shape[-2:] != (self.knot_steps, self.action_dim):
            raise ValueError("control knot has the wrong shape")
        source = torch.cat(
            (task_state, route.flatten(1), previous_action, knot.flatten(1)), dim=-1
        )
        prediction = self.network(source)
        return prediction.view(
            len(source), len(self.horizons), self.task_state_dim
        )


class BoundedResidualKnotPolicy(nn.Module):
    """30 Hz actor that emits a hard-bounded three-tick 90 Hz CTBR knot.

    The actor predicts a bounded residual from the command currently applied by
    the vehicle and two bounded per-tick derivatives.  The command integrator is
    part of the model contract, so neither training nor deployment can violate
    the configured command slew envelope.
    """

    def __init__(
        self,
        *,
        representation: dict | None = None,
        knot_steps: int = 3,
        delta_bounds: tuple[float, ...] | list[float] = (0.27, 0.34, 0.34, 0.34),
        derivative_bounds: tuple[float, ...] | list[float] = (0.27, 0.34, 0.34, 0.34),
        outcome_horizons: tuple[int, ...] | list[int] = (1, 3, 6, 9),
        outcome_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        representation = dict(representation or {})
        self.representation = MultiRateControlRepresentation(**representation)
        self.knot_steps = int(knot_steps)
        if self.knot_steps != 3:
            raise ValueError("the residual-knot contract currently requires three commands")
        action_dim = self.representation.action_dim
        delta = torch.as_tensor(delta_bounds, dtype=torch.float32)
        derivative = torch.as_tensor(derivative_bounds, dtype=torch.float32)
        if delta.shape != (action_dim,) or derivative.shape != (action_dim,):
            raise ValueError("delta and derivative bounds must match action_dim")
        if not bool((delta > 0).all() and (derivative > 0).all()):
            raise ValueError("control-knot bounds must be positive")
        self.register_buffer("delta_bounds", delta)
        self.register_buffer("derivative_bounds", derivative)
        d_model = self.representation.d_model
        self.knot_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, self.knot_steps * action_dim),
        )
        # Starting from a hold-last knot makes the initial deployment behavior
        # benign and leaves all expressiveness in learned bounded residuals.
        nn.init.zeros_(self.knot_head[-1].weight)
        nn.init.zeros_(self.knot_head[-1].bias)
        self.outcome_model = LocalKnotOutcomeModel(
            task_state_dim=self.representation.task_state_dim,
            route_gates=self.representation.route_gates,
            route_dim=self.representation.route_dim,
            action_dim=action_dim,
            knot_steps=self.knot_steps,
            horizons=outcome_horizons,
            hidden_dim=int(outcome_hidden_dim),
        )

    @property
    def outcome_horizons(self) -> tuple[int, ...]:
        return self.outcome_model.horizons

    def encode(
        self, batch: dict[str, torch.Tensor], history: int, *, start: int = 0
    ) -> ControlRepresentationOutput:
        source = slice(int(start), int(start) + int(history))
        return self.representation(
            mask=batch["mask"][:, source],
            task_state=batch["deployable_task_state"][:, source],
            previous_action=batch["previous_action"][:, source],
            timing=batch["timing"][:, source],
            route=batch["route"][:, source],
            gate_index=batch["gate_index"][:, source, 0],
        )

    def integrate_knot(
        self, raw: torch.Tensor, previous_action: torch.Tensor
    ) -> torch.Tensor:
        batch = raw.shape[0]
        raw = raw.view(batch, self.knot_steps, self.representation.action_dim)
        commands = []
        command = previous_action + torch.tanh(raw[:, 0]) * self.delta_bounds
        command = command.clamp(-1.0, 1.0)
        commands.append(command)
        for edge in range(1, self.knot_steps):
            command = command + torch.tanh(raw[:, edge]) * self.derivative_bounds
            command = command.clamp(-1.0, 1.0)
            commands.append(command)
        return torch.stack(commands, dim=1)

    def forward_knot(
        self,
        batch: dict[str, torch.Tensor],
        *,
        history: int,
        start: int = 0,
        anchor_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ControlRepresentationOutput]:
        encoded = self.encode(batch, history, start=start)
        target_index = int(start) + int(history) - 1
        if anchor_action is None:
            anchor_action = batch["previous_action"][:, target_index]
        return self.integrate_knot(self.knot_head(encoded.pooled), anchor_action), encoded

    def outcome_prediction(
        self,
        batch: dict[str, torch.Tensor],
        knot: torch.Tensor,
        *,
        history: int,
        start: int = 0,
    ) -> torch.Tensor:
        target = int(start) + int(history) - 1
        return self.outcome_model(
            batch["deployable_task_state"][:, target],
            batch["route"][:, target],
            batch["previous_action"][:, target],
            knot,
        )

    @torch.no_grad()
    def sample_knot(
        self,
        batch: dict[str, torch.Tensor],
        *,
        history: int,
        anchor_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_knot(
            batch, history=history, anchor_action=anchor_action
        )[0]


class BoundedResidualFlowPolicy(nn.Module):
    """Local stochastic flow correction around a frozen residual-knot policy.

    The zero latent is exactly the base policy.  Exploration and FPO++ operate
    in a bounded latent correction space before the command integrator, so RL
    cannot violate the base policy's absolute or per-tick slew bounds.
    """

    def __init__(
        self,
        base: BoundedResidualKnotPolicy,
        *,
        history_raw_steps: int = 9,
        correction_logit_scale: float = 0.35,
        flow: dict | None = None,
    ) -> None:
        super().__init__()
        if history_raw_steps < 1 or correction_logit_scale <= 0:
            raise ValueError("bounded residual flow settings must be positive")
        self.base = base.eval().requires_grad_(False)
        self.history_raw_steps = int(history_raw_steps)
        self.correction_logit_scale = float(correction_logit_scale)
        flow = dict(flow or {})
        representation = base.representation
        flow.setdefault("observation_token_dim", representation.d_model)
        flow.setdefault("action_token_dim", representation.d_model)
        flow.setdefault("observation_tokens", representation.token_count)
        flow.setdefault("visual_tokens", 4)
        flow.setdefault("action_tokens", 0)
        flow.setdefault("history_steps", 1)
        flow.setdefault("action_horizon", base.knot_steps)
        flow.setdefault("action_dim", representation.action_dim)
        flow.setdefault("d_model", representation.d_model)
        flow.setdefault("n_heads", 6)
        flow.setdefault("context_depth", 2)
        flow.setdefault("flow_depth", 4)
        flow.setdefault("mlp_ratio", 3)
        flow.setdefault("k_max", 4)
        flow.setdefault("source_mode", "gaussian")
        flow.setdefault("source_noise", 0.35)
        self.flow_config = flow
        self.flow = TokenizerFlowPolicy(**flow)

    @property
    def history_steps(self) -> int:
        return self.flow.history_steps

    @property
    def action_horizon(self) -> int:
        return self.flow.action_horizon

    @property
    def action_dim(self) -> int:
        return self.flow.action_dim

    def policy_tokens(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            encoded = self.base.encode(batch, self.history_raw_steps)
        observation = encoded.tokens.unsqueeze(1)
        actions = observation.new_empty(
            len(observation), self.history_steps, 0,
            self.base.representation.d_model,
        )
        return observation, actions

    def encode_context(
        self, observation_tokens: torch.Tensor, action_tokens: torch.Tensor
    ) -> torch.Tensor:
        return self.flow.encode_context(observation_tokens, action_tokens)

    def velocity_from_context(
        self,
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor,
        step_size: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.flow.velocity_from_context(
            noisy_actions, flow_time, step_size, context
        )

    def latent_to_knot(
        self,
        latent: torch.Tensor,
        observation_tokens: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        if latent.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError("residual-flow latent has the wrong shape")
        tokens = observation_tokens[:, 0]
        with torch.no_grad():
            pooled = self.base.representation.pool(tokens.flatten(1))
            base_logits = self.base.knot_head(pooled).view(
                len(tokens), self.action_horizon, self.action_dim
            )
        corrected = base_logits + self.correction_logit_scale * latent
        return self.base.integrate_knot(corrected, previous_action)

    @torch.no_grad()
    def sample_policy(
        self,
        observation_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        *,
        previous_action: torch.Tensor,
        steps: int = 4,
        method: str = "heun",
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.flow.sample(
            observation_tokens, action_tokens,
            steps=steps, method=method, deterministic=deterministic,
        )
        knot = self.latent_to_knot(latent, observation_tokens, previous_action)
        return knot, latent

    def zero_residual_loss(
        self,
        observation_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = observation_tokens.new_zeros(
            len(observation_tokens), self.action_horizon, self.action_dim
        )
        return self.flow.shortcut_training_loss(
            target, observation_tokens, action_tokens,
            direct_weight=0.5, bootstrap_weight=1.0,
        )

    def model_config(self) -> dict:
        return {
            "history_raw_steps": self.history_raw_steps,
            "correction_logit_scale": self.correction_logit_scale,
            "flow": dict(self.flow_config),
        }


class BoundedResidualGaussianPolicy(nn.Module):
    """Exact-likelihood local residual around a frozen bounded-knot policy.

    The Gaussian lives in unconstrained pre-integrator logit space.  Its zero
    mean reproduces the base policy exactly, while ``integrate_knot`` retains
    the base actor's hard absolute-command and per-tick slew bounds.  PPO stores
    and scores the raw latent, so no clipped/squashed-density approximation is
    required.
    """

    def __init__(
        self,
        base: BoundedResidualKnotPolicy,
        *,
        history_raw_steps: int = 9,
        correction_logit_scale: float = 0.35,
        hidden_dim: int = 192,
        initial_std: float = 0.18,
        minimum_std: float = 0.03,
        maximum_std: float = 0.50,
    ) -> None:
        super().__init__()
        if min(history_raw_steps, hidden_dim) < 1 or correction_logit_scale <= 0:
            raise ValueError("Gaussian residual dimensions and scale must be positive")
        if not 0 < minimum_std <= initial_std <= maximum_std:
            raise ValueError("Gaussian residual standard deviations are inconsistent")
        self.base = base.eval().requires_grad_(False)
        self.history_raw_steps = int(history_raw_steps)
        self.correction_logit_scale = float(correction_logit_scale)
        self.hidden_dim = int(hidden_dim)
        self.minimum_std = float(minimum_std)
        self.maximum_std = float(maximum_std)
        d_model = base.representation.d_model
        output = base.knot_steps * base.representation.action_dim
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, output),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.log_std = nn.Parameter(torch.full(
            (base.knot_steps, base.representation.action_dim),
            float(math.log(initial_std)),
        ))

    @property
    def action_horizon(self) -> int:
        return self.base.knot_steps

    @property
    def action_dim(self) -> int:
        return self.base.representation.action_dim

    def policy_tokens(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            encoded = self.base.encode(batch, self.history_raw_steps)
        observation = encoded.tokens.unsqueeze(1)
        actions = observation.new_empty(
            len(observation), 1, 0, self.base.representation.d_model
        )
        return observation, actions

    def distribution(self, observation_tokens: torch.Tensor):
        # Local import avoids coupling the BC control contract to actor-critic
        # training utilities at module import time.
        from starscream.actor_critic import GaussianPolicyOutput

        tokens = observation_tokens[:, 0]
        pooled = self.base.representation.pool(tokens.flatten(1))
        mean = self.residual_head(pooled).view(
            len(tokens), self.action_horizon, self.action_dim
        )
        log_std = self.log_std.clamp(
            math.log(self.minimum_std), math.log(self.maximum_std)
        ).unsqueeze(0).expand_as(mean)
        return GaussianPolicyOutput(mean=mean, log_std=log_std)

    def latent_to_knot(
        self,
        latent: torch.Tensor,
        observation_tokens: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        if latent.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError("Gaussian residual latent has the wrong shape")
        tokens = observation_tokens[:, 0]
        with torch.no_grad():
            pooled = self.base.representation.pool(tokens.flatten(1))
            base_logits = self.base.knot_head(pooled).view_as(latent)
        return self.base.integrate_knot(
            base_logits + self.correction_logit_scale * latent,
            previous_action,
        )

    @torch.no_grad()
    def sample_policy(
        self,
        observation_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        *,
        previous_action: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del action_tokens
        policy = self.distribution(observation_tokens)
        latent = policy.mean if deterministic else policy.mean + policy.std * torch.randn_like(policy.mean)
        return self.latent_to_knot(latent, observation_tokens, previous_action), latent

    def model_config(self) -> dict:
        return {
            "history_raw_steps": self.history_raw_steps,
            "correction_logit_scale": self.correction_logit_scale,
            "hidden_dim": self.hidden_dim,
            "initial_std": float(self.log_std.detach().exp().mean()),
            "minimum_std": self.minimum_std,
            "maximum_std": self.maximum_std,
        }
