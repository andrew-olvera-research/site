"""Direct deployment-observation adaptation for a frozen privileged policy.

The adapter estimates the privileged policy's explicit 19-value task state;
known route, command, and timing fields bypass it.  The original privileged
Transformer and CTBR head remain frozen and differentiable with respect to the
estimated input, preserving the solved controller while exposing a
control-sensitive learning signal to perception.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from .privileged_racing import FeatureNormalizer, PrivilegedMLPPolicy


RAW_FEATURE_DIM = 84
RAW_ESTIMATE_DIM = 32
TASK_DIM = 19
ROUTE_DIM = 39
COMMON_DIM = ROUTE_DIM + 4 + 2
PRIVILEGED_FEATURE_DIM = TASK_DIM + COMMON_DIM

TASK_SCALE = (
    20.0, 20.0, 20.0,
    30.0, 30.0, 30.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    6.0, 6.0, 6.0,
    4000.0, 4000.0, 4000.0, 4000.0,
)
DEFAULT_CORRECTION_BOUNDS = (
    0.075, 0.075, 0.075,
    0.10, 0.10, 0.10,
    0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
    1.0, 1.0, 1.0,
    1.5, 1.5, 1.5, 1.5,
)


def task_to_physical(task: torch.Tensor) -> torch.Tensor:
    """Undo the deployment task-state normalization."""

    scale = task.new_tensor(TASK_SCALE)
    return task * scale


def route_to_physical(route: torch.Tensor) -> torch.Tensor:
    """Undo the online/offline route normalization for three 13D records."""

    if route.shape[-1] != ROUTE_DIM:
        raise ValueError("route must contain three flattened 13-value records")
    shaped = route.reshape(*route.shape[:-1], 3, 13).clone()
    shaped[..., 0:3] *= 20.0
    shaped[..., 9:11] *= 5.0
    return shaped.flatten(-2)


@dataclass(frozen=True)
class AdapterOutput:
    task: torch.Tensor
    correction: torch.Tensor
    log_scale: torch.Tensor
    sequence_latent: torch.Tensor


class CausalPrivilegedStateAdapter(nn.Module):
    """Estimate the teacher's explicit task state as a bounded residual."""

    def __init__(
        self,
        *,
        input_dim: int = RAW_FEATURE_DIM,
        context_steps: int = 18,
        hidden_dim: int = 256,
        depth: int = 3,
        encoder_type: str = "transformer",
        attention_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.0,
        correction_bounds: Sequence[float] = DEFAULT_CORRECTION_BOUNDS,
        raw_normalizer: FeatureNormalizer,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.context_steps = int(context_steps)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.encoder_type = str(encoder_type).lower()
        self.attention_heads = int(attention_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.dropout = float(dropout)
        bounds = np.asarray(correction_bounds, np.float32)
        if (
            self.input_dim != RAW_FEATURE_DIM or self.context_steps < 2
            or self.hidden_dim < 32 or self.depth < 1
            or bounds.shape != (TASK_DIM,) or not np.all(np.isfinite(bounds))
            or np.any(bounds <= 0)
        ):
            raise ValueError("invalid privileged state adapter configuration")
        if self.encoder_type not in {"gru", "transformer"}:
            raise ValueError("adapter encoder_type must be gru or transformer")
        if self.encoder_type == "transformer" and self.hidden_dim % self.attention_heads:
            raise ValueError("adapter hidden width must be divisible by attention heads")
        if len(raw_normalizer.mean) != self.input_dim:
            raise ValueError("raw normalizer does not match adapter input width")
        self.register_buffer("raw_mean", torch.from_numpy(raw_normalizer.mean.copy()))
        self.register_buffer("raw_std", torch.from_numpy(raw_normalizer.std.copy()))
        self.register_buffer("correction_bounds", torch.from_numpy(bounds))
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
        )
        if self.encoder_type == "gru":
            self.temporal_position = None
            self.encoder: nn.Module = nn.GRU(
                self.hidden_dim, self.hidden_dim, num_layers=self.depth,
                batch_first=True,
            )
            self.output_norm: nn.Module = nn.LayerNorm(self.hidden_dim)
        else:
            self.temporal_position = nn.Parameter(
                torch.zeros(1, self.context_steps, self.hidden_dim)
            )
            nn.init.trunc_normal_(self.temporal_position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim, nhead=self.attention_heads,
                dim_feedforward=self.feedforward_dim, dropout=self.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=self.depth, enable_nested_tensor=False
            )
            self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.correction_head = nn.Linear(self.hidden_dim, TASK_DIM)
        self.uncertainty_head = nn.Linear(self.hidden_dim, TASK_DIM)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.constant_(self.uncertainty_head.bias, -3.0)

    def config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim, "context_steps": self.context_steps,
            "hidden_dim": self.hidden_dim, "depth": self.depth,
            "encoder_type": self.encoder_type, "attention_heads": self.attention_heads,
            "feedforward_dim": self.feedforward_dim, "dropout": self.dropout,
            "correction_bounds": self.correction_bounds.detach().cpu().tolist(),
        }

    def forward(self, raw_features: torch.Tensor) -> AdapterOutput:
        if raw_features.ndim != 3 or raw_features.shape[-1] != self.input_dim:
            raise ValueError(f"raw features must have shape (B,T,{self.input_dim})")
        if raw_features.shape[1] > self.context_steps:
            raw_features = raw_features[:, -self.context_steps:]
        normalized = (raw_features - self.raw_mean) / self.raw_std
        embedded = self.input_projection(normalized)
        if self.encoder_type == "gru":
            encoded, _ = self.encoder(embedded)
        else:
            length = embedded.shape[1]
            embedded = embedded + self.temporal_position[:, -length:]
            mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=embedded.device),
                diagonal=1,
            )
            encoded = self.encoder(embedded, mask=mask)
        encoded = self.output_norm(encoded)
        correction = torch.tanh(self.correction_head(encoded)) * self.correction_bounds
        baseline = raw_features.new_zeros((*raw_features.shape[:-1], TASK_DIM))
        baseline[..., :12] = raw_features[..., :12]
        # Body rates and motor speeds are direct normalized measurements, not
        # quantities the adapter should have to infer from action history.
        baseline[..., 12:19] = raw_features[..., 32:39]
        task = baseline + correction
        log_scale = self.uncertainty_head(encoded).clamp(-7.0, 2.0)
        return AdapterOutput(task, correction, log_scale, encoded)


class PrivilegedInterfacePolicy(nn.Module):
    """A deployable state adapter feeding an unchanged privileged teacher."""

    def __init__(
        self,
        adapter: CausalPrivilegedStateAdapter,
        teacher: PrivilegedMLPPolicy,
        teacher_normalizer: FeatureNormalizer,
        action_contract: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if teacher.input_dim != PRIVILEGED_FEATURE_DIM:
            raise ValueError("teacher must consume the 64-value privileged contract")
        if teacher.context_steps != adapter.context_steps:
            raise ValueError("adapter and teacher history lengths must match")
        if len(teacher_normalizer.mean) != PRIVILEGED_FEATURE_DIM:
            raise ValueError("teacher normalizer width mismatch")
        self.adapter = adapter
        self.teacher = teacher
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.register_buffer(
            "teacher_mean", torch.from_numpy(teacher_normalizer.mean.copy())
        )
        self.register_buffer(
            "teacher_std", torch.from_numpy(teacher_normalizer.std.copy())
        )
        raw_contract = dict(action_contract or {})
        self.action_contract = {
            "collective_mapping": str(raw_contract.get(
                "collective_mapping",
                raw_contract.get("collective_action_mapping", "legacy_cubic_tail"),
            )),
            "collective_maximum": float(raw_contract.get(
                "collective_maximum",
                raw_contract.get("collective_action_maximum", 40.0),
            )),
            "collective_reference_thrust": float(raw_contract.get(
                "collective_reference_thrust",
                raw_contract.get("collective_action_reference_thrust", 15.0),
            )),
            "collective_logit_scale": float(raw_contract.get(
                "collective_logit_scale",
                raw_contract.get("collective_action_logit_scale", 1.6),
            )),
        }
        if self.action_contract["collective_mapping"] not in {
            "legacy_cubic_tail", "logit_sigmoid_v1",
        }:
            raise ValueError("unsupported teacher collective-action contract")

    def train(self, mode: bool = True) -> "PrivilegedInterfacePolicy":
        super().train(mode)
        self.teacher.eval()
        return self

    def teacher_features(
        self, raw_features: torch.Tensor, normalized_task: torch.Tensor,
    ) -> torch.Tensor:
        if raw_features.shape[:-1] != normalized_task.shape[:-1]:
            raise ValueError("raw feature and task sequences must align")
        route = route_to_physical(raw_features[..., 39:78])
        previous = self.teacher_previous_action(raw_features[..., 78:82])
        # Deployment timing stores previous-command age in 90 Hz steps; the
        # original teacher stores seconds.
        age = raw_features[..., 82:83] / 90.0
        valid = raw_features[..., 83:84]
        return torch.cat(
            [task_to_physical(normalized_task), route, previous, age, valid], dim=-1
        )

    def teacher_previous_action(self, legacy: torch.Tensor) -> torch.Tensor:
        """Map the production history's legacy CTBR coordinates to the teacher."""

        if self.action_contract["collective_mapping"] == "legacy_cubic_tail":
            return legacy
        physical = 15.0 * (legacy[..., 0] + 1.0)
        maximum = self.action_contract["collective_maximum"]
        reference = self.action_contract["collective_reference_thrust"]
        scale = self.action_contract["collective_logit_scale"]
        fraction = (physical / maximum).clamp(1.0e-5, 1.0 - 1.0e-5)
        bias = math.log(reference / (maximum - reference))
        collective = torch.tanh((torch.logit(fraction) - bias) / scale)
        return torch.cat([collective.unsqueeze(-1), legacy[..., 1:]], dim=-1)

    def normalize_teacher(self, teacher_features: torch.Tensor) -> torch.Tensor:
        return (teacher_features - self.teacher_mean) / self.teacher_std

    def estimated_teacher_input(
        self, raw_features: torch.Tensor,
    ) -> tuple[torch.Tensor, AdapterOutput]:
        output = self.adapter(raw_features)
        features = self.teacher_features(raw_features, output.task)
        return self.normalize_teacher(features), output

    def exact_teacher_input(
        self, raw_features: torch.Tensor, exact_task: torch.Tensor,
    ) -> torch.Tensor:
        return self.normalize_teacher(self.teacher_features(raw_features, exact_task))

    def forward(
        self, raw_features: torch.Tensor,
        speed_command: torch.Tensor | None = None,
    ) -> torch.Tensor:
        estimated, _ = self.estimated_teacher_input(raw_features)
        return self.teacher(estimated, speed_command)

    def paired_forward(
        self, raw_features: torch.Tensor, exact_task: torch.Tensor,
        speed_command: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | AdapterOutput]:
        estimated_input, adapter = self.estimated_teacher_input(raw_features)
        exact_input = self.exact_teacher_input(raw_features, exact_task)
        estimated_latent = self.teacher.encode(estimated_input)
        estimated_control = self.teacher._conditioned_encoding(
            estimated_latent, speed_command
        )
        if self.teacher.topology_head is not None:
            assert self.teacher.topology_adapter is not None
            topology = self.teacher.topology_head(estimated_control)
            estimated_control = estimated_control + self.teacher.topology_adapter(topology)
        estimated_action = self.teacher._action_chunk_logits(
            estimated_control
        )[:, 0].tanh()
        estimated_dynamics = self.teacher.dynamics_head(estimated_latent)
        with torch.no_grad():
            exact_latent = self.teacher.encode(exact_input)
            exact_control = self.teacher._conditioned_encoding(
                exact_latent, speed_command
            )
            if self.teacher.topology_head is not None:
                assert self.teacher.topology_adapter is not None
                topology = self.teacher.topology_head(exact_control)
                exact_control = exact_control + self.teacher.topology_adapter(topology)
            exact_action = self.teacher._action_chunk_logits(
                exact_control
            )[:, 0].tanh()
            exact_dynamics = self.teacher.dynamics_head(exact_latent)
        return {
            "adapter": adapter,
            "estimated_input": estimated_input, "exact_input": exact_input,
            "estimated_latent": estimated_latent, "exact_latent": exact_latent,
            "estimated_action": estimated_action, "exact_action": exact_action,
            "estimated_dynamics": estimated_dynamics,
            "exact_dynamics": exact_dynamics,
        }

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "contract": "privileged-interface-distillation-v1",
            "adapter": self.adapter.state_dict(),
            "adapter_config": self.adapter.config(),
            "raw_normalizer": {
                "mean": self.adapter.raw_mean.detach().cpu().numpy(),
                "std": self.adapter.raw_std.detach().cpu().numpy(),
            },
            "teacher": self.teacher.state_dict(),
            "teacher_config": self.teacher.model_config(),
            "teacher_normalizer": {
                "mean": self.teacher_mean.detach().cpu().numpy(),
                "std": self.teacher_std.detach().cpu().numpy(),
            },
            "action_contract": dict(self.action_contract),
            "action_horizon": 1, "control_hz": 90,
            "observation_source": "raw_estimate",
        }


def load_interface_policy(
    state: Mapping[str, Any], device: str | torch.device = "cpu",
) -> PrivilegedInterfacePolicy:
    if state.get("contract") != "privileged-interface-distillation-v1":
        raise ValueError("not a privileged-interface distillation checkpoint")
    raw = FeatureNormalizer.from_state_dict(state["raw_normalizer"])
    adapter = CausalPrivilegedStateAdapter(
        **dict(state["adapter_config"]), raw_normalizer=raw
    )
    adapter.load_state_dict(state["adapter"])
    teacher = PrivilegedMLPPolicy(**dict(state["teacher_config"]))
    teacher.load_state_dict(state["teacher"])
    teacher_normalizer = FeatureNormalizer.from_state_dict(state["teacher_normalizer"])
    return PrivilegedInterfacePolicy(
        adapter, teacher, teacher_normalizer,
        action_contract=state.get("action_contract"),
    ).to(device)
