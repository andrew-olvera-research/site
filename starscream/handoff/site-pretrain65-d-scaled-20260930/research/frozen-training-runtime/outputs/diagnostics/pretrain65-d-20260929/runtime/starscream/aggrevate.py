"""Cost-to-go aggregation primitives for continuous-control imitation.

This module keeps the algorithmic contract independent from Flightmare:

* AggreVaTe fits the expert cost-to-go ``Q*(s, a)`` on states induced by a
  learner/expert roll-in mixture, then performs a cost-sensitive policy update.
* AggreVaTeD differentiates the fitted cost-to-go through a deterministic
  policy.  A behavior-cloning term and conservative critic ensemble keep that
  update inside the queried action support.

The environment collector lives in ``scripts/train_privileged_racing.py`` so
it can reuse the persistent MPCC workers.  Keeping the objective here makes it
possible to test the math without Flightmare or ACADOS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


AGGREVATE_VARIANTS = {
    "aggrevate",
    "aggrevated",
    "hybrid_aggrevate",
    "hybrid_aggrevated",
}


@dataclass(frozen=True)
class AggreVaTeConfig:
    """Validated algorithm settings shared by collection and optimization."""

    variant: str = "hybrid_aggrevated"
    discount: float = 1.0
    step_cost: float = 1.0 / 90.0
    failure_base_cost: float = 20.0
    failure_cost: float = 10.0
    crash_cost: float = 5.0
    incomplete_power: float = 1.0
    query_horizon: int = 450
    query_min_step: int = 0
    query_max_step: int = 360
    query_progress_strata: tuple[float, ...] = ()
    query_progress_jitter: float = 0.0
    expert_query_fraction: float = 0.20
    learner_query_fraction: float = 0.30
    teacher_perturbation_fraction: float = 0.50
    teacher_noise_std: tuple[float, float, float, float] = (
        0.10, 0.16, 0.16, 0.12,
    )
    critic_ensemble: int = 3
    critic_hidden_dim: int = 384
    critic_depth: int = 3
    critic_huber_beta: float = 0.25
    critic_target_clip: float = 8.0
    conservative_quantile: float = 1.0
    value_weight: float = 0.20
    behavior_cloning_weight: float = 1.0
    sampled_action_weight: float = 0.0
    candidate_noise_scales: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)
    actor_huber_beta: float = 0.05
    actor_advantage_clip: float = 4.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "AggreVaTeConfig":
        values = dict(raw or {})
        if "teacher_noise_std" in values:
            values["teacher_noise_std"] = tuple(
                float(item) for item in values["teacher_noise_std"]
            )
        if "candidate_noise_scales" in values:
            values["candidate_noise_scales"] = tuple(
                float(item) for item in values["candidate_noise_scales"]
            )
        if "query_progress_strata" in values:
            values["query_progress_strata"] = tuple(
                float(item) for item in values["query_progress_strata"]
            )
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if self.variant not in AGGREVATE_VARIANTS:
            raise ValueError(
                f"variant must be one of {sorted(AGGREVATE_VARIANTS)}"
            )
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("AggreVaTe discount must be in (0,1]")
        if (
            self.step_cost <= 0.0 or self.failure_base_cost < 0.0
            or self.failure_cost < 0.0 or self.crash_cost < 0.0
        ):
            raise ValueError("AggreVaTe costs must be finite and non-negative")
        if not all(np.isfinite((
            self.step_cost, self.failure_base_cost,
            self.failure_cost, self.crash_cost,
            self.incomplete_power,
        ))):
            raise ValueError("AggreVaTe costs must be finite")
        if self.incomplete_power <= 0.0:
            raise ValueError("incomplete_power must be positive")
        if self.query_horizon < 1:
            raise ValueError("query_horizon must be positive")
        if not 0 <= self.query_min_step <= self.query_max_step:
            raise ValueError("invalid AggreVaTe query-step interval")
        if self.query_progress_strata and (
            any(not 0.0 <= item <= 1.0 for item in self.query_progress_strata)
            or tuple(sorted(self.query_progress_strata))
            != self.query_progress_strata
        ):
            raise ValueError(
                "query_progress_strata must be sorted fractions in [0,1]"
            )
        if not 0.0 <= self.query_progress_jitter <= 0.5:
            raise ValueError("query_progress_jitter must be in [0,0.5]")
        fractions = np.asarray([
            self.expert_query_fraction,
            self.learner_query_fraction,
            self.teacher_perturbation_fraction,
        ], np.float64)
        if np.any(fractions < 0.0) or not np.isclose(fractions.sum(), 1.0):
            raise ValueError("AggreVaTe query-action fractions must sum to one")
        noise = np.asarray(self.teacher_noise_std, np.float64)
        if noise.shape != (4,) or np.any(noise < 0.0) or not np.all(np.isfinite(noise)):
            raise ValueError("teacher_noise_std must contain four finite non-negative values")
        if self.critic_ensemble < 1 or self.critic_hidden_dim < 16 or self.critic_depth < 1:
            raise ValueError("invalid AggreVaTe critic dimensions")
        if self.critic_huber_beta <= 0.0 or self.critic_target_clip <= 0.0:
            raise ValueError("invalid AggreVaTe critic loss settings")
        if not 0.5 <= self.conservative_quantile <= 1.0:
            raise ValueError("conservative_quantile must be in [0.5,1]")
        if self.value_weight < 0.0 or self.behavior_cloning_weight < 0.0:
            raise ValueError("AggreVaTe actor weights must be non-negative")
        if self.value_weight + self.behavior_cloning_weight <= 0.0:
            raise ValueError("AggreVaTe actor objective cannot be empty")
        if not self.candidate_noise_scales or any(
            (not np.isfinite(item) or item < 0.0)
            for item in self.candidate_noise_scales
        ):
            raise ValueError("candidate_noise_scales must be finite and non-negative")
        if self.actor_huber_beta <= 0.0 or self.actor_advantage_clip <= 0.0:
            raise ValueError("invalid AggreVaTe actor loss settings")

    @property
    def differentiable(self) -> bool:
        return self.variant in {"aggrevated", "hybrid_aggrevated"}

    @property
    def hybrid(self) -> bool:
        return self.variant.startswith("hybrid_")


@dataclass
class AggreVaTeQueryBatch:
    """One cost-to-go query per completed expert-rollout episode."""

    histories: np.ndarray
    query_actions: np.ndarray
    expert_actions: np.ndarray
    costs_to_go: np.ndarray
    remaining_fractions: np.ndarray
    speed_commands: np.ndarray
    track_ids: np.ndarray
    query_steps: np.ndarray
    query_modes: np.ndarray
    completed: np.ndarray
    crashed: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.histories)
        expected = {
            "query_actions": (count, 4),
            "expert_actions": (count, 4),
            "costs_to_go": (count,),
            "remaining_fractions": (count,),
            "speed_commands": (count,),
            "track_ids": (count,),
            "query_steps": (count,),
            "query_modes": (count,),
            "completed": (count,),
            "crashed": (count,),
        }
        if self.histories.ndim != 3:
            raise ValueError("AggreVaTe histories must have shape [N,T,D]")
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(
                    f"AggreVaTe query field {name} has shape {value.shape}, expected {shape}"
                )
        for name in (
            "histories", "query_actions", "expert_actions", "costs_to_go",
            "remaining_fractions", "speed_commands",
        ):
            if not np.all(np.isfinite(np.asarray(getattr(self, name)))):
                raise ValueError(f"AggreVaTe query field {name} is not finite")
        if np.any(self.remaining_fractions < 0.0) or np.any(
            self.remaining_fractions > 1.0
        ):
            raise ValueError("AggreVaTe remaining fractions must be in [0,1]")


@dataclass
class AggreVaTeImitationBatch:
    """All valid MPCC labels observed during cost-to-go query episodes."""

    histories: np.ndarray
    expert_actions: np.ndarray
    previous_actions: np.ndarray
    dynamics: np.ndarray
    dynamics_valid: np.ndarray
    speed_commands: np.ndarray
    track_ids: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.histories)
        if self.histories.ndim != 3:
            raise ValueError("hybrid imitation histories must have shape [N,T,D]")
        for name, shape in {
            "expert_actions": (count, 4),
            "previous_actions": (count, 4),
            "dynamics_valid": (count,),
            "speed_commands": (count,),
            "track_ids": (count,),
        }.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(
                    f"hybrid imitation field {name} has shape {value.shape}, expected {shape}"
                )
        if self.dynamics.ndim != 2 or self.dynamics.shape[0] != count:
            raise ValueError("hybrid imitation dynamics are misaligned")
        for name in (
            "histories", "expert_actions", "previous_actions", "dynamics",
            "speed_commands",
        ):
            if not np.all(np.isfinite(np.asarray(getattr(self, name)))):
                raise ValueError(f"hybrid imitation field {name} is not finite")


class CostNormalizer:
    """Numerically stable scalar normalization persisted with the critic."""

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = float(mean)
        self.std = float(std)
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std <= 0.0:
            raise ValueError("invalid cost normalization")

    @classmethod
    def fit(cls, values: np.ndarray | torch.Tensor) -> "CostNormalizer":
        array = np.asarray(
            values.detach().cpu().numpy() if torch.is_tensor(values) else values,
            np.float64,
        )
        if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
            raise ValueError("cost normalization requires a finite non-empty vector")
        return cls(float(array.mean()), max(float(array.std()), 1.0e-4))

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.std

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std + self.mean

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "CostNormalizer":
        return cls(float(state["mean"]), float(state["std"]))


def discounted_cost_to_go(
    step_costs: Sequence[float], discount: float,
    *, terminal_cost: float = 0.0,
) -> float:
    """Return ``sum gamma^k c_k + gamma^H terminal_cost``."""

    if not 0.0 < float(discount) <= 1.0:
        raise ValueError("discount must be in (0,1]")
    costs = np.asarray(step_costs, np.float64)
    if costs.ndim != 1 or not np.all(np.isfinite(costs)):
        raise ValueError("step costs must be a finite vector")
    if not np.isfinite(terminal_cost):
        raise ValueError("terminal cost must be finite")
    powers = np.power(float(discount), np.arange(len(costs), dtype=np.float64))
    return float(np.dot(costs, powers) + float(discount) ** len(costs) * terminal_cost)


def minimum_time_terminal_cost(
    *, completed: bool, crashed: bool, gate_fraction: float,
    config: AggreVaTeConfig,
) -> float:
    """Failure penalty used after the expert rollout from a queried action."""

    if completed:
        return 0.0
    remaining = 1.0 - float(np.clip(gate_fraction, 0.0, 1.0))
    return (
        config.failure_base_cost
        + config.failure_cost * remaining ** config.incomplete_power
        + config.crash_cost * float(crashed)
    )


class _QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = int(input_dim)
        for _ in range(int(depth)):
            layers.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


class AggreVaTeQCritic(nn.Module):
    """Ensemble approximation to expert cost-to-go ``Q*(s,a)``."""

    def __init__(
        self,
        latent_dim: int,
        *,
        hidden_dim: int = 384,
        depth: int = 3,
        ensemble: int = 3,
        speed_scale: float = 20.0,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or ensemble < 1 or speed_scale <= 0.0:
            raise ValueError("invalid AggreVaTe critic contract")
        self.latent_dim = int(latent_dim)
        self.ensemble_size = int(ensemble)
        self.speed_scale = float(speed_scale)
        # Encoded history + normalized CTBR + remaining horizon + pace.
        input_dim = self.latent_dim + 4 + 2
        self.members = nn.ModuleList([
            _QNetwork(input_dim, int(hidden_dim), int(depth))
            for _ in range(self.ensemble_size)
        ])

    def inputs(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        remaining_fraction: torch.Tensor,
        speed_command: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError("critic latent must have shape [B,latent_dim]")
        if action.shape != (len(latent), 4):
            raise ValueError("critic action must have shape [B,4]")
        remaining = remaining_fraction.reshape(-1, 1).to(latent)
        speed = speed_command.reshape(-1, 1).to(latent) / self.speed_scale
        if len(remaining) != len(latent) or len(speed) != len(latent):
            raise ValueError("critic context must align with the batch")
        return torch.cat((latent, action, remaining, speed), dim=-1)

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        remaining_fraction: torch.Tensor,
        speed_command: torch.Tensor,
    ) -> torch.Tensor:
        value = self.inputs(latent, action, remaining_fraction, speed_command)
        return torch.stack([member(value) for member in self.members], dim=-1)

    def conservative(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        remaining_fraction: torch.Tensor,
        speed_command: torch.Tensor,
        *,
        quantile: float = 1.0,
    ) -> torch.Tensor:
        estimates = self(latent, action, remaining_fraction, speed_command)
        if not 0.5 <= float(quantile) <= 1.0:
            raise ValueError("critic quantile must be in [0.5,1]")
        if self.ensemble_size == 1:
            return estimates[:, 0]
        # torch.quantile has no BF16 kernel. Keep the conservative aggregation
        # in FP32 while preserving autograd through the ensemble estimates.
        return torch.quantile(estimates.float(), float(quantile), dim=-1)


def aggrevate_critic_loss(
    estimates: torch.Tensor,
    target: torch.Tensor,
    *,
    huber_beta: float,
    bootstrap_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bootstrap-compatible fitted-Q regression for an ensemble."""

    if estimates.ndim != 2 or target.shape != estimates.shape[:1]:
        raise ValueError("AggreVaTe critic estimates and targets are misaligned")
    error = F.smooth_l1_loss(
        estimates, target[:, None].expand_as(estimates),
        beta=float(huber_beta), reduction="none",
    )
    if bootstrap_mask is None:
        mask = torch.ones_like(error)
    else:
        mask = bootstrap_mask.to(error)
        if mask.shape != error.shape:
            raise ValueError("AggreVaTe bootstrap mask is misaligned")
    loss = (error * mask).sum() / mask.sum().clamp_min(1.0)
    mean = estimates.mean(-1)
    return loss, {
        "critic_loss": loss.detach(),
        "critic_mae": (mean - target).abs().mean().detach(),
        "critic_bias": (mean - target).mean().detach(),
        "critic_ensemble_std": estimates.std(
            dim=-1, unbiased=False
        ).mean().detach(),
    }


def _critic_values(
    critic: AggreVaTeQCritic,
    latent: torch.Tensor,
    actions: torch.Tensor,
    remaining_fraction: torch.Tensor,
    speed_command: torch.Tensor,
    config: AggreVaTeConfig,
) -> torch.Tensor:
    return critic.conservative(
        latent, actions, remaining_fraction, speed_command,
        quantile=config.conservative_quantile,
    )


def aggrevate_actor_loss(
    *,
    policy_actions: torch.Tensor,
    latent: torch.Tensor,
    critic: AggreVaTeQCritic,
    expert_actions: torch.Tensor,
    sampled_actions: torch.Tensor,
    remaining_fraction: torch.Tensor,
    speed_command: torch.Tensor,
    config: AggreVaTeConfig,
    candidate_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """AggreVaTe/AggreVaTeD actor objective for a deterministic CTBR actor.

    ``aggrevate`` is the cost-sensitive reduction: select a bounded candidate
    with the fitted expert-Q and regress to that candidate. ``aggrevated``
    directly minimizes the fitted expert advantage. Hybrid variants retain an
    explicit MPCC action anchor.
    """

    batch = len(policy_actions)
    for name, value in {
        "policy_actions": policy_actions,
        "expert_actions": expert_actions,
        "sampled_actions": sampled_actions,
    }.items():
        if value.shape != (batch, 4):
            raise ValueError(f"{name} must have shape [B,4]")
    q_expert = _critic_values(
        critic, latent, expert_actions, remaining_fraction, speed_command, config
    ).detach()
    q_policy = _critic_values(
        critic, latent, policy_actions, remaining_fraction, speed_command, config
    )
    normalized_advantage = (q_policy - q_expert).clamp(
        -config.actor_advantage_clip, config.actor_advantage_clip
    )
    bc_per_sample = F.smooth_l1_loss(
        policy_actions, expert_actions,
        beta=config.actor_huber_beta, reduction="none",
    ).mean(-1)

    selected_actions = expert_actions
    selected_q = q_expert
    if config.differentiable:
        value_loss = normalized_advantage.mean()
    else:
        candidates = [expert_actions, sampled_actions, policy_actions.detach()]
        if candidate_noise is not None:
            if candidate_noise.ndim != 3 or candidate_noise.shape[:2] != (
                batch, len(config.candidate_noise_scales)
            ) or candidate_noise.shape[-1] != 4:
                raise ValueError("candidate noise has the wrong shape")
            for index, scale in enumerate(config.candidate_noise_scales):
                candidates.append(torch.clamp(
                    expert_actions + float(scale) * candidate_noise[:, index],
                    -1.0, 1.0,
                ))
        stacked = torch.stack(candidates, dim=1)
        flat_actions = stacked.reshape(-1, 4)
        flat_latent = latent[:, None].expand(
            -1, stacked.shape[1], -1
        ).reshape(-1, latent.shape[-1])
        flat_remaining = remaining_fraction[:, None].expand(
            -1, stacked.shape[1]
        ).reshape(-1)
        flat_speed = speed_command[:, None].expand(
            -1, stacked.shape[1]
        ).reshape(-1)
        with torch.no_grad():
            candidate_q = _critic_values(
                critic, flat_latent, flat_actions, flat_remaining, flat_speed,
                config,
            ).reshape(batch, -1)
            selected_index = candidate_q.argmin(dim=1)
            selected_q = candidate_q.gather(1, selected_index[:, None])[:, 0]
            selected_actions = stacked[
                torch.arange(batch, device=stacked.device), selected_index
            ]
        value_loss = F.smooth_l1_loss(
            policy_actions, selected_actions,
            beta=config.actor_huber_beta,
        )

    bc_loss = bc_per_sample.mean()
    sampled_loss = F.smooth_l1_loss(
        policy_actions, sampled_actions,
        beta=config.actor_huber_beta,
    )
    total = (
        config.value_weight * value_loss
        + (config.behavior_cloning_weight if config.hybrid else 0.0) * bc_loss
        + config.sampled_action_weight * sampled_loss
    )
    return total, {
        "actor_loss": total.detach(),
        "actor_value_loss": value_loss.detach(),
        "actor_bc_loss": bc_loss.detach(),
        "actor_sampled_loss": sampled_loss.detach(),
        "expert_q": q_expert.mean().detach(),
        "policy_q": q_policy.mean().detach(),
        "expert_advantage": normalized_advantage.mean().detach(),
        "selected_q": selected_q.mean().detach(),
        "selected_expert_fraction": (
            (selected_actions - expert_actions).abs().amax(-1) < 1.0e-6
        ).float().mean().detach(),
        "actor_action_shift": (
            policy_actions - expert_actions
        ).square().mean().sqrt().detach(),
    }


class AggreVaTeReplayBuffer:
    """Small bounded query replay; one row represents one expensive rollout."""

    def __init__(self, capacity: int) -> None:
        if int(capacity) < 1:
            raise ValueError("AggreVaTe replay capacity must be positive")
        self.capacity = int(capacity)
        self._batches: list[AggreVaTeQueryBatch] = []

    def __len__(self) -> int:
        return sum(len(item.histories) for item in self._batches)

    def append(self, batch: AggreVaTeQueryBatch) -> None:
        if not len(batch.histories):
            return
        self._batches.append(batch)
        while len(self) > self.capacity and self._batches:
            excess = len(self) - self.capacity
            first = self._batches[0]
            if excess >= len(first.histories):
                self._batches.pop(0)
                continue
            keep = np.arange(excess, len(first.histories))
            self._batches[0] = self._index(first, keep)

    @staticmethod
    def _index(batch: AggreVaTeQueryBatch, indices: np.ndarray) -> AggreVaTeQueryBatch:
        return AggreVaTeQueryBatch(**{
            name: np.asarray(getattr(batch, name))[indices]
            for name in batch.__dataclass_fields__
        })

    def arrays(self) -> AggreVaTeQueryBatch:
        if not self._batches:
            raise ValueError("AggreVaTe replay is empty")
        return AggreVaTeQueryBatch(**{
            name: np.concatenate([
                np.asarray(getattr(batch, name)) for batch in self._batches
            ], axis=0)
            for name in AggreVaTeQueryBatch.__dataclass_fields__
        })

    def sample(
        self, rng: np.random.Generator, count: int,
    ) -> AggreVaTeQueryBatch:
        arrays = self.arrays()
        indices = rng.choice(
            len(arrays.histories), size=int(count),
            replace=len(arrays.histories) < int(count),
        )
        return self._index(arrays, indices)

    def sample_stratified(
        self, rng: np.random.Generator, count: int,
        *, include_query_mode: bool = True,
    ) -> AggreVaTeQueryBatch:
        """Sample every represented track/mode cell at equal expected mass.

        Cost-to-go rollouts have track-dependent duration and failure rates.
        Uniform row sampling would therefore let the easiest-to-query track
        define the critic and actor updates even when collection attempts were
        family balanced.
        """

        arrays = self.arrays()
        if include_query_mode:
            width = int(np.max(arrays.query_modes)) + 1
            strata = arrays.track_ids.astype(np.int64) * width + arrays.query_modes
        else:
            strata = arrays.track_ids.astype(np.int64)
        values = np.unique(strata)
        if not len(values):
            raise ValueError("AggreVaTe replay has no represented strata")
        base, extra = divmod(int(count), len(values))
        selected: list[np.ndarray] = []
        for index, value in enumerate(values):
            candidates = np.flatnonzero(strata == value)
            amount = base + int(index < extra)
            if amount:
                selected.append(rng.choice(
                    candidates, size=amount,
                    replace=len(candidates) < amount,
                ))
        indices = np.concatenate(selected)
        rng.shuffle(indices)
        return self._index(arrays, indices)

    def state_dict(self) -> dict[str, Any]:
        if not self._batches:
            return {"capacity": self.capacity, "arrays": None}
        arrays = self.arrays()
        return {
            "capacity": self.capacity,
            "arrays": {
                name: np.asarray(getattr(arrays, name))
                for name in arrays.__dataclass_fields__
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "AggreVaTeReplayBuffer":
        result = cls(int(state["capacity"]))
        arrays = state.get("arrays")
        if arrays is not None:
            result.append(AggreVaTeQueryBatch(**dict(arrays)))
        return result


class AggreVaTeImitationReplayBuffer:
    """Bounded all-step MPCC replay used only by hybrid variants."""

    def __init__(self, capacity: int) -> None:
        if int(capacity) < 1:
            raise ValueError("hybrid imitation replay capacity must be positive")
        self.capacity = int(capacity)
        self._batches: list[AggreVaTeImitationBatch] = []

    def __len__(self) -> int:
        return sum(len(item.histories) for item in self._batches)

    @staticmethod
    def _index(
        batch: AggreVaTeImitationBatch, indices: np.ndarray,
    ) -> AggreVaTeImitationBatch:
        return AggreVaTeImitationBatch(**{
            name: np.asarray(getattr(batch, name))[indices]
            for name in batch.__dataclass_fields__
        })

    def append(self, batch: AggreVaTeImitationBatch) -> None:
        if not len(batch.histories):
            return
        self._batches.append(batch)
        while len(self) > self.capacity and self._batches:
            excess = len(self) - self.capacity
            first = self._batches[0]
            if excess >= len(first.histories):
                self._batches.pop(0)
            else:
                self._batches[0] = self._index(
                    first, np.arange(excess, len(first.histories))
                )

    def arrays(self) -> AggreVaTeImitationBatch:
        if not self._batches:
            raise ValueError("hybrid imitation replay is empty")
        return AggreVaTeImitationBatch(**{
            name: np.concatenate([
                np.asarray(getattr(batch, name)) for batch in self._batches
            ], axis=0)
            for name in AggreVaTeImitationBatch.__dataclass_fields__
        })

    def sample(
        self, rng: np.random.Generator, count: int,
    ) -> AggreVaTeImitationBatch:
        arrays = self.arrays()
        indices = rng.choice(
            len(arrays.histories), size=int(count),
            replace=len(arrays.histories) < int(count),
        )
        return self._index(arrays, indices)

    def state_dict(self) -> dict[str, Any]:
        if not self._batches:
            return {"capacity": self.capacity, "arrays": None}
        arrays = self.arrays()
        return {
            "capacity": self.capacity,
            "arrays": {
                name: np.asarray(getattr(arrays, name))
                for name in arrays.__dataclass_fields__
            },
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any],
    ) -> "AggreVaTeImitationReplayBuffer":
        result = cls(int(state["capacity"]))
        arrays = state.get("arrays")
        if arrays is not None:
            result.append(AggreVaTeImitationBatch(**dict(arrays)))
        return result
