"""Self-supervised, Dreamer world-model, actor, and critic objectives."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
import torch.nn.functional as F

from dreamerv4 import TwoHot
from .tokenizer import macro_last


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = logits.sigmoid()
    dimensions = tuple(range(2, logits.ndim))
    intersection = (probability * target).sum(dimensions)
    denominator = probability.sum(dimensions) + target.sum(dimensions)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Precision-biased overlap loss; alpha weights false positives."""

    probability = logits.sigmoid()
    dimensions = tuple(range(2, logits.ndim))
    true_positive = (probability * target).sum(dimensions)
    false_positive = (probability * (1.0 - target)).sum(dimensions)
    false_negative = ((1.0 - probability) * target).sum(dimensions)
    index = (true_positive + eps) / (
        true_positive + float(alpha) * false_positive + float(beta) * false_negative + eps
    )
    return (1.0 - index).pow(float(gamma)).mean()


def gatenet_loss(outputs: list[torch.Tensor], target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SkyDreamer Appendix A deep-supervision objective."""

    weights = (4.0, 2.0, 1.0, 1.0, 1.0)
    if len(outputs) != len(weights):
        raise ValueError("GateNet must return five supervised output maps")
    losses = []
    metrics: dict[str, torch.Tensor] = {}
    for index, (output, weight) in enumerate(zip(outputs, weights)):
        resized = F.interpolate(target, size=output.shape[-2:], mode="nearest")
        bce = F.binary_cross_entropy_with_logits(output, resized)
        dice = dice_loss(output, resized)
        loss = dice + 2.0 * bce
        losses.append(weight * loss)
        metrics[f"scale_{index}"] = loss.detach()
    total = torch.stack(losses).sum() / sum(weights)
    metrics["loss"] = total.detach()
    return total, metrics


@torch.no_grad()
def binary_segmentation_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    prediction = logits.sigmoid() >= 0.5
    truth = target >= 0.5
    dimensions = tuple(range(1, prediction.ndim))
    intersection = (prediction & truth).sum(dimensions).float()
    union = (prediction | truth).sum(dimensions).float()
    true_positive = intersection
    false_positive = (prediction & ~truth).sum(dimensions).float()
    false_negative = (~prediction & truth).sum(dimensions).float()
    iou = torch.where(union > 0, intersection / union.clamp_min(1), (~prediction).all(dimensions).float())
    nonempty = truth.any(dimensions)
    foreground_iou = iou[nonempty].mean() if nonempty.any() else logits.new_zeros(())
    foreground_precision = (
        (true_positive / (true_positive + false_positive).clamp_min(1))[nonempty].mean()
        if nonempty.any() else logits.new_zeros(())
    )
    foreground_recall = (
        (true_positive / (true_positive + false_negative).clamp_min(1))[nonempty].mean()
        if nonempty.any() else logits.new_zeros(())
    )
    empty = ~nonempty
    return {
        "iou": iou.mean(),
        "precision": (true_positive / (true_positive + false_positive).clamp_min(1)).mean(),
        "recall": (true_positive / (true_positive + false_negative).clamp_min(1)).mean(),
        "foreground_iou": foreground_iou,
        "foreground_precision": foreground_precision,
        "foreground_recall": foreground_recall,
        "empty_accuracy": (~prediction).all(dimensions)[empty].float().mean() if empty.any() else logits.new_zeros(()),
        "empty_false_positive": false_positive[empty].mean() if empty.any() else logits.new_zeros(()),
    }


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    first = F.normalize(value[..., 0:3], dim=-1)
    raw_second = value[..., 3:6]
    second = F.normalize(raw_second - (first * raw_second).sum(-1, keepdim=True) * first, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def rotation_geodesic_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = rotation_6d_to_matrix(prediction).transpose(-1, -2) @ rotation_6d_to_matrix(target)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cosine).mean()


def _masked_rotation_geodesic_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    relative = rotation_6d_to_matrix(prediction).transpose(-1, -2) @ rotation_6d_to_matrix(target)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1 + 1e-6, 1 - 1e-6
    )
    error = torch.acos(cosine)
    if valid.shape != error.shape:
        raise ValueError("rotation transition mask must align with rotation errors")
    weights = valid.to(error.dtype)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def heteroscedastic_loss(
    mean: torch.Tensor, log_scale: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    inverse_variance = torch.exp(-2.0 * log_scale)
    return (0.5 * (mean - target).square() * inverse_variance + log_scale).mean()


def gaussian_nll(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    include_constant: bool = True,
) -> torch.Tensor:
    """Diagonal Gaussian NLL parameterized by log standard deviation.

    NLL = 0.5 * ((x-mu)^2 / sigma^2 + 2 log(sigma) + log(2 pi)).
    Computing the inverse variance from ``-2*log_std`` avoids an unnecessary
    square root and is stable for the bounded log scales emitted by the models.
    """

    if mean.shape != target.shape or log_std.shape != target.shape:
        raise ValueError("mean, log_std, and target must have identical shapes")
    loss = 0.5 * (mean - target).square() * torch.exp(-2.0 * log_std) + log_std
    if include_constant:
        loss = loss + 0.5 * torch.log(mean.new_tensor(2.0 * torch.pi))
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    return loss.mean()


def _macro_targets(batch: dict[str, torch.Tensor], patch_size: int):
    mask = macro_last(batch["mask"], patch_size)
    proprio_source = batch["proprio"] if "proprio" in batch else batch["vector"][..., :11]
    proprio = macro_last(proprio_source, patch_size)
    route_source = batch.get("route_target", batch.get("route"))
    if route_source is None:
        route_source = batch["vector"][..., 11:].reshape(*batch["vector"].shape[:-1], 3, 13)
    route = macro_last(route_source, patch_size)
    state = macro_last(batch["task_state"], patch_size)
    return mask, proprio, route, state


def _route_orientation_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Cosine loss for the normal/up basis in a 13D gate record.

    Gate records contain position(3), normal(3), up(3), size(2), opposite,
    valid. They do not contain quaternions; treating columns 3:7 as one was a
    labeling bug in the previous tokenizer objective.
    """

    if prediction.shape != target.shape or prediction.shape[-1] != 13:
        raise ValueError("route orientation loss requires matching (...,13) records")
    normal = 1.0 - F.cosine_similarity(
        prediction[..., 3:6].float(), target[..., 3:6].float(), dim=-1
    )
    up = 1.0 - F.cosine_similarity(
        prediction[..., 6:9].float(), target[..., 6:9].float(), dim=-1
    )
    valid = target[..., 12].float().clamp(0.0, 1.0)
    return ((normal + up) * 0.5 * valid).sum() / valid.sum().clamp_min(1.0)


def _loss_scale(reference: torch.Tensor, values, width: int) -> torch.Tensor:
    scale = reference.new_tensor(values if values is not None else [1.0] * width)
    if scale.shape != (width,) or torch.any(scale <= 0):
        raise ValueError(f"loss scale must contain {width} positive values")
    return scale


def _weighted_task_state_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    group_weights,
) -> torch.Tensor:
    """Group-balanced 19D task-state regression.

    Position, velocity, attitude, rates, and motor state have very different
    widths and observability. Group weighting prevents directly measured rates
    and motors from hiding weak visual position/velocity estimates.
    """

    if group_weights is None:
        return F.smooth_l1_loss(prediction.float() / scale, target.float() / scale)
    weights = tuple(float(value) for value in group_weights)
    if len(weights) != 5 or any(value < 0 for value in weights) or sum(weights) <= 0:
        raise ValueError("state_group_weights must contain five nonnegative values")
    groups = ((0, 3), (3, 6), (6, 12), (12, 15), (15, 19))
    losses = [
        F.smooth_l1_loss(
            prediction[..., start:stop].float() / scale[start:stop],
            target[..., start:stop].float() / scale[start:stop],
        )
        for start, stop in groups
    ]
    return sum(weight * loss for weight, loss in zip(weights, losses)) / sum(weights)


def _masked_weighted_task_state_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    group_weights,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Group-balanced state loss over transitions that retain one gate frame."""

    if valid.shape != prediction.shape[:-1] or target.shape != prediction.shape:
        raise ValueError("state transition mask must align with prediction and target")
    valid = valid.to(prediction.dtype)
    denominator = valid.sum().clamp_min(1.0)
    weights = tuple(float(value) for value in (group_weights or (1, 1, 1, 1, 1)))
    if len(weights) != 5 or any(value < 0 for value in weights) or sum(weights) <= 0:
        raise ValueError("state_group_weights must contain five nonnegative values")
    groups = ((0, 3), (3, 6), (6, 12), (12, 15), (15, 19))
    losses = []
    for start, stop in groups:
        elementwise = F.smooth_l1_loss(
            prediction[..., start:stop].float() / scale[start:stop],
            target[..., start:stop].float() / scale[start:stop],
            reduction="none",
        ).mean(dim=-1)
        losses.append((elementwise * valid).sum() / denominator)
    return sum(weight * loss for weight, loss in zip(weights, losses)) / sum(weights)


def _macro_gate_index(batch: dict[str, torch.Tensor], patch_size: int) -> torch.Tensor | None:
    gate_index = batch.get("gate_index")
    if gate_index is None:
        return None
    if gate_index.ndim == 3:
        gate_index = gate_index[..., 0]
    if gate_index.ndim != 2:
        raise ValueError("gate_index must have shape (B,T) or (B,T,K)")
    return macro_last(gate_index.unsqueeze(-1), patch_size).squeeze(-1)


def _representation_regularizers(
    latents: torch.Tensor, variance_target: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg-style variance/covariance guards over the complete token code."""

    features = latents.float().flatten(0, 1).flatten(1)
    if features.shape[0] < 2:
        zero = features.new_zeros(())
        return zero, zero
    centered = features - features.mean(dim=0, keepdim=True)
    std = centered.var(dim=0, unbiased=False).add(1e-4).sqrt()
    variance = F.relu(float(variance_target) - std).mean()
    normalized = centered / std.clamp_min(1e-4)
    covariance = normalized.T @ normalized / max(1, normalized.shape[0] - 1)
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    decorrelation = off_diagonal.square().sum() / max(
        1, covariance.numel() - covariance.shape[0]
    )
    return variance, decorrelation


def _grouped_representation_regularizers(
    latents: torch.Tensor, group_sizes: tuple[int, ...], variance_target: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Guard each reserved modality bank instead of allowing cross-bank collapse."""

    if not group_sizes:
        zero = latents.new_zeros(())
        return zero, zero
    if sum(group_sizes) != latents.shape[-2]:
        raise ValueError("latent group sizes must match the latent token count")
    values = [
        _representation_regularizers(group, variance_target)
        for group in latents.split(group_sizes, dim=-2)
    ]
    return (
        torch.stack([value[0] for value in values]).mean(),
        torch.stack([value[1] for value in values]).mean(),
    )


def tokenizer_loss(
    output,
    batch: dict[str, torch.Tensor],
    *,
    state_huber_weight: float = 1.0,
    state_objective: str = "smooth_l1",
    state_nll_weight: float = 0.05,
    state_log_scale_min: float = -2.3,
    mask_positive_weight_max: float = 20.0,
    proprio_channel_scale=None,
    route_channel_scale=None,
    state_channel_scale=None,
    state_group_weights=None,
    mask_objective: str = "balanced_bce_dice",
    tversky_alpha: float = 0.7,
    tversky_beta: float = 0.3,
    tversky_gamma: float = 0.75,
    mask_geometry_weight: float = 0.0,
    mask_weight: float = 1.0,
    visual_state_auxiliary_weight: float = 0.0,
    fusion_state_auxiliary_weight: float = 0.0,
    regime_state_weights=None,
    tail_state_cvar_fraction: float = 0.0,
    tail_state_cvar_weight: float = 0.0,
    tail_phase_weights=None,
    tail_vector_auxiliary_weight: float = 0.0,
    state_delta_weight: float = 0.0,
    future_state_weight: float = 0.0,
    future_state_horizon_weights=None,
    inverse_action_weight: float = 0.0,
    progress_weight: float = 0.0,
    progress_center=None,
    progress_scale=None,
    action_state_delta_weight: float = 0.0,
    gate_frame_safe_dynamics: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Auxiliary reconstruction/probe loss for the multi-rate state tokenizer."""

    mask_target, proprio_target, route_target, state_target = _macro_targets(batch, output.patch_size)
    mask_logits = output.mask_logits.float()
    mask_target = mask_target.float()
    positive = mask_target.sum()
    negative = mask_target.numel() - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(
        1.0, float(mask_positive_weight_max)
    )
    mask_bce = F.binary_cross_entropy_with_logits(
        mask_logits, mask_target, pos_weight=positive_weight
    )
    mask_dice = dice_loss(output.mask_logits.float(), mask_target.float())
    mask_tversky = focal_tversky_loss(
        mask_logits, mask_target, alpha=tversky_alpha, beta=tversky_beta,
        gamma=tversky_gamma,
    )
    if mask_objective == "balanced_bce_dice":
        mask_structure = mask_dice
    elif mask_objective == "precision_tversky":
        mask_structure = mask_tversky
    else:
        raise ValueError("mask_objective must be 'balanced_bce_dice' or 'precision_tversky'")
    proprio_scale = _loss_scale(output.proprio, proprio_channel_scale, output.proprio.shape[-1])
    route_scale = _loss_scale(output.route, route_channel_scale, output.route.shape[-1])
    state_scale = _loss_scale(output.state_mean, state_channel_scale, output.state_mean.shape[-1])
    proprio_error = F.smooth_l1_loss(
        output.proprio.float() / proprio_scale, proprio_target.float() / proprio_scale,
        reduction="none",
    ).mean(dim=-1)
    route_error = F.smooth_l1_loss(
        output.route.float() / route_scale, route_target.float() / route_scale,
        reduction="none",
    ).mean(dim=(-1, -2))
    proprio = proprio_error.mean()
    route = route_error.mean()
    phase_weights = None
    mean_phase_weight = mask_bce.new_ones(())
    if tail_phase_weights is not None and "tail_phase" in batch:
        configured_phase_weights = proprio_error.new_tensor(tail_phase_weights)
        phase = macro_last(
            batch["tail_phase"].unsqueeze(-1), output.patch_size
        ).squeeze(-1).long()
        if phase.shape != proprio_error.shape:
            raise ValueError("macro tail phases must align with reconstruction steps")
        if phase.numel() and int(phase.max()) >= len(configured_phase_weights):
            raise ValueError("tail_phase_weights does not cover every phase index")
        phase_weights = configured_phase_weights[phase]
        mean_phase_weight = phase_weights.mean()
    tail_vector = mask_bce.new_zeros(())
    if phase_weights is not None:
        tail_vector = (
            ((proprio_error + route_error) * phase_weights).sum()
            / phase_weights.sum().clamp_min(1e-6)
        )
    route_rotation = _route_orientation_loss(output.route, route_target)
    normalized_state_mean = output.state_mean.float() / state_scale
    normalized_state_target = state_target.float() / state_scale
    if state_objective == "smooth_l1":
        state_mean = _weighted_task_state_loss(
            output.state_mean, state_target, state_scale, state_group_weights
        )
    elif state_objective == "mse":
        state_mean = F.mse_loss(normalized_state_mean, normalized_state_target)
    else:
        raise ValueError("state_objective must be 'smooth_l1' or 'mse'")
    standardized_log_scale = output.state_log_scale.float() - state_scale.log()
    bounded_state_log_scale = standardized_log_scale.clamp_min(float(state_log_scale_min))
    state_nll = heteroscedastic_loss(
        output.state_mean.float() / state_scale,
        bounded_state_log_scale,
        state_target.float() / state_scale,
    )
    rotation = rotation_geodesic_loss(
        output.state_mean[..., 6:12].float(), state_target[..., 6:12].float()
    )
    visual_state = mask_bce.new_zeros(())
    visual_state_rotation = mask_bce.new_zeros(())
    if output.visual_state_mean is not None:
        visual_state = F.smooth_l1_loss(
            output.visual_state_mean.float() / state_scale,
            state_target.float() / state_scale,
        )
        visual_state_rotation = rotation_geodesic_loss(
            output.visual_state_mean[..., 6:12].float(), state_target[..., 6:12].float()
        )
    fusion_state = mask_bce.new_zeros(())
    fusion_state_unweighted = mask_bce.new_zeros(())
    fusion_state_cvar = mask_bce.new_zeros(())
    fusion_state_rotation = mask_bce.new_zeros(())
    mean_regime_weight = mask_bce.new_ones(())
    if output.fusion_state_mean is not None:
        fusion_error = F.smooth_l1_loss(
            output.fusion_state_mean.float() / state_scale,
            state_target.float() / state_scale,
            reduction="none",
        ).mean(dim=-1)
        fusion_state_unweighted = fusion_error.mean()
        if phase_weights is not None:
            weights = phase_weights
            fusion_state = (fusion_error * weights).sum() / weights.sum().clamp_min(1e-6)
            mean_regime_weight = weights.mean()
        elif regime_state_weights is not None and "distribution_regime" in batch:
            configured_weights = fusion_error.new_tensor(regime_state_weights)
            regime = batch["distribution_regime"].long()
            if regime.ndim != 1 or regime.shape[0] != fusion_error.shape[0]:
                raise ValueError("distribution_regime must contain one index per batch sample")
            if regime.numel() and int(regime.max()) >= len(configured_weights):
                raise ValueError("regime_state_weights does not cover every regime index")
            weights = configured_weights[regime].view(-1, 1).expand_as(fusion_error)
            mean_regime_weight = weights.mean()
            fusion_state = (fusion_error * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            fusion_state = fusion_state_unweighted
        fraction = float(tail_state_cvar_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("tail_state_cvar_fraction must be between zero and one")
        if fraction > 0.0:
            flat_error = fusion_error.flatten()
            count = max(1, int(round(fraction * flat_error.numel())))
            fusion_state_cvar = flat_error.topk(count, sorted=False).values.mean()
        fusion_state_rotation = rotation_geodesic_loss(
            output.fusion_state_mean[..., 6:12].float(), state_target[..., 6:12].float()
        )
    state_delta = mask_bce.new_zeros(())
    macro_gate_index = _macro_gate_index(batch, output.patch_size)
    transition_valid = (
        macro_gate_index[:, 1:] == macro_gate_index[:, :-1]
        if gate_frame_safe_dynamics and macro_gate_index is not None
        else torch.ones(
            state_target.shape[0], max(0, state_target.shape[1] - 1),
            device=state_target.device, dtype=torch.bool,
        )
    )
    if output.state_delta_mean is not None:
        if state_target.shape[1] < 2:
            raise ValueError("state-delta objective requires at least two macro steps")
        delta_target = state_target[:, 1:] - state_target[:, :-1]
        state_delta = _masked_weighted_task_state_loss(
            output.state_delta_mean[:, 1:], delta_target, state_scale,
            state_group_weights, transition_valid,
        )
    future_state = mask_bce.new_zeros(())
    future_state_rotation = mask_bce.new_zeros(())
    if output.future_state_mean is not None:
        horizons = output.future_state_horizons
        configured = future_state_horizon_weights or [1.0] * len(horizons)
        if len(configured) != len(horizons):
            raise ValueError("future_state_horizon_weights must match future_state_horizons")
        horizon_losses = []
        horizon_rotations = []
        horizon_weights = []
        for index, (horizon, weight) in enumerate(zip(horizons, configured)):
            horizon = int(horizon)
            if horizon >= state_target.shape[1]:
                continue
            prediction = output.future_state_mean[:, :-horizon, index]
            target = state_target[:, horizon:]
            valid = (
                macro_gate_index[:, :-horizon] == macro_gate_index[:, horizon:]
                if gate_frame_safe_dynamics and macro_gate_index is not None
                else torch.ones(prediction.shape[:-1], device=prediction.device, dtype=torch.bool)
            )
            horizon_losses.append(_masked_weighted_task_state_loss(
                prediction, target, state_scale, state_group_weights, valid
            ))
            horizon_rotations.append(_masked_rotation_geodesic_loss(
                prediction[..., 6:12].float(), target[..., 6:12].float(), valid
            ))
            horizon_weights.append(float(weight))
        if horizon_losses:
            weight_tensor = state_target.new_tensor(horizon_weights)
            future_state = (
                torch.stack(horizon_losses) * weight_tensor
            ).sum() / weight_tensor.sum().clamp_min(1e-6)
            future_state_rotation = (
                torch.stack(horizon_rotations) * weight_tensor
            ).sum() / weight_tensor.sum().clamp_min(1e-6)
    inverse_action = mask_bce.new_zeros(())
    if output.inverse_action_mean is not None:
        if "normalized_action" not in batch:
            raise ValueError("inverse-action objective requires include_actions: true")
        actions = batch["normalized_action"].reshape(
            batch["normalized_action"].shape[0], -1, output.patch_size,
            batch["normalized_action"].shape[-1],
        )
        inverse_action = F.smooth_l1_loss(
            output.inverse_action_mean.float(), actions[:, 1:].float()
        )
    progress = mask_bce.new_zeros(())
    if output.progress_mean is not None:
        if "privileged_progress" not in batch:
            raise ValueError("progress objective requires include_privileged: true")
        progress_target = macro_last(batch["privileged_progress"], output.patch_size)
        width = output.progress_mean.shape[-1]
        center = output.progress_mean.new_tensor(
            progress_center if progress_center is not None else [0.0] * width
        )
        scale = output.progress_mean.new_tensor(
            progress_scale if progress_scale is not None else [1.0] * width
        )
        if center.shape != (width,) or scale.shape != (width,) or torch.any(scale <= 0):
            raise ValueError("invalid progress normalization")
        progress = F.smooth_l1_loss(
            output.progress_mean.float(),
            (progress_target[..., :width].float() - center) / scale,
        )
    action_state_delta = mask_bce.new_zeros(())
    if output.action_state_delta_mean is not None:
        if state_target.shape[1] < 2:
            raise ValueError("action-conditioned dynamics requires at least two steps")
        action_state_delta = _masked_weighted_task_state_loss(
            output.action_state_delta_mean[:, :-1],
            state_target[:, 1:] - state_target[:, :-1],
            state_scale,
            state_group_weights,
            transition_valid,
        )
    geometry = mask_bce.new_zeros(())
    geometry_presence = mask_bce.new_zeros(())
    geometry_shape = mask_bce.new_zeros(())
    if output.mask_geometry is not None:
        truth = (mask_target >= 0.5).float()
        height, width = truth.shape[-2:]
        y = torch.linspace(-1.0, 1.0, height, device=truth.device, dtype=truth.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=truth.device, dtype=truth.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        mass = truth.sum(dim=(-3, -2, -1))
        safe_mass = mass.clamp_min(1.0)
        centroid_x = (truth * xx).sum(dim=(-3, -2, -1)) / safe_mass
        centroid_y = (truth * yy).sum(dim=(-3, -2, -1)) / safe_mass
        spread_x = ((truth * (xx - centroid_x[..., None, None, None]).square()).sum(
            dim=(-3, -2, -1)
        ) / safe_mass).sqrt()
        spread_y = ((truth * (yy - centroid_y[..., None, None, None]).square()).sum(
            dim=(-3, -2, -1)
        ) / safe_mass).sqrt()
        area = (mass / float(height * width)).sqrt()
        presence = mass > 0
        geometry_target = torch.stack(
            [centroid_x, centroid_y, spread_x, spread_y, area], dim=-1
        )
        geometry_presence = F.binary_cross_entropy_with_logits(
            output.mask_geometry[..., 0].float(), presence.float()
        )
        if presence.any():
            geometry_shape = F.smooth_l1_loss(
                output.mask_geometry[..., 1:].float().tanh()[presence],
                geometry_target[presence],
            )
        geometry = geometry_presence + geometry_shape
    total = (
        float(mask_weight) * (mask_bce + mask_structure) + proprio + route + 0.1 * route_rotation
        + float(state_huber_weight) * state_mean
        + float(state_nll_weight) * state_nll
        + 0.1 * rotation
        + float(mask_geometry_weight) * geometry
        + float(visual_state_auxiliary_weight) * (
            visual_state + 0.1 * visual_state_rotation
        )
        + float(fusion_state_auxiliary_weight) * (
            fusion_state
            + float(tail_state_cvar_weight) * fusion_state_cvar
            + 0.1 * fusion_state_rotation
        )
        + float(tail_vector_auxiliary_weight) * tail_vector
        + float(state_delta_weight) * state_delta
        + float(future_state_weight) * (future_state + 0.1 * future_state_rotation)
        + float(inverse_action_weight) * inverse_action
        + float(progress_weight) * progress
        + float(action_state_delta_weight) * action_state_delta
    )
    return total, {
        "loss": total.detach(),
        "mask_bce": mask_bce.detach(),
        "mask_dice": mask_dice.detach(),
        "mask_tversky": mask_tversky.detach(),
        "proprio": proprio.detach(),
        "route": route.detach(),
        "route_rotation": route_rotation.detach(),
        "state_huber": state_mean.detach(),
        "state_nll": state_nll.detach(),
        "state_log_scale": bounded_state_log_scale.mean().detach(),
        "mask_positive_weight": positive_weight.detach(),
        "mask_geometry": geometry.detach(),
        "mask_geometry_presence": geometry_presence.detach(),
        "mask_geometry_shape": geometry_shape.detach(),
        "rotation": rotation.detach(),
        "visual_state_huber": visual_state.detach(),
        "visual_state_rotation": visual_state_rotation.detach(),
        "fusion_state_huber": fusion_state.detach(),
        "fusion_state_unweighted": fusion_state_unweighted.detach(),
        "fusion_state_cvar": fusion_state_cvar.detach(),
        "fusion_state_rotation": fusion_state_rotation.detach(),
        "mean_regime_weight": mean_regime_weight.detach(),
        "tail_vector": tail_vector.detach(),
        "mean_phase_weight": mean_phase_weight.detach(),
        "state_delta": state_delta.detach(),
        "future_state": future_state.detach(),
        "future_state_rotation": future_state_rotation.detach(),
        "inverse_action": inverse_action.detach(),
        "progress": progress.detach(),
        "action_state_delta": action_state_delta.detach(),
        "gate_consistent_transition_fraction": transition_valid.float().mean().detach(),
    }


def observation_jepa_loss(
    output,
    batch: dict[str, torch.Tensor],
    *,
    prediction_weight: float = 1.0,
    prediction_l1_weight: float = 0.25,
    unconditioned_prediction_weight: float = 0.0,
    reconstruction_weight: float = 0.5,
    variance_weight: float = 0.1,
    covariance_weight: float = 0.01,
    variance_target: float = 0.5,
    state_huber_weight: float = 1.0,
    state_objective: str = "smooth_l1",
    state_nll_weight: float = 0.05,
    state_log_scale_min: float = -2.3,
    mask_positive_weight_max: float = 20.0,
    proprio_channel_scale=None,
    route_channel_scale=None,
    state_channel_scale=None,
    state_group_weights=None,
    mask_objective: str = "balanced_bce_dice",
    tversky_alpha: float = 0.7,
    tversky_beta: float = 0.3,
    tversky_gamma: float = 0.75,
    mask_geometry_weight: float = 0.0,
    mask_weight: float = 1.0,
    visual_state_auxiliary_weight: float = 0.0,
    fusion_state_auxiliary_weight: float = 0.0,
    regime_state_weights=None,
    tail_state_cvar_fraction: float = 0.0,
    tail_state_cvar_weight: float = 0.0,
    tail_phase_weights=None,
    tail_vector_auxiliary_weight: float = 0.0,
    state_delta_weight: float = 0.0,
    future_state_weight: float = 0.0,
    future_state_horizon_weights=None,
    inverse_action_weight: float = 0.0,
    progress_weight: float = 0.0,
    progress_center=None,
    progress_scale=None,
    action_state_delta_weight: float = 0.0,
    grouped_variance_weight: float = 0.0,
    grouped_covariance_weight: float = 0.0,
    prediction_warmup_steps: int = 0,
    selection_prediction_weight: float = 0.25,
    selection_mask_weight: float = 1.0,
    selection_state_weight: float = 1.0,
    selection_belief_weight: float = 1.0,
    gate_frame_safe_dynamics: bool = False,
    gate_consistent_prediction: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """EMA-target representation prediction plus geometry-preserving auxiliaries."""

    reconstruction, reconstruction_metrics = tokenizer_loss(
        output.reconstruction,
        batch,
        state_huber_weight=state_huber_weight,
        state_objective=state_objective,
        state_nll_weight=state_nll_weight,
        state_log_scale_min=state_log_scale_min,
        mask_positive_weight_max=mask_positive_weight_max,
        proprio_channel_scale=proprio_channel_scale,
        route_channel_scale=route_channel_scale,
        state_channel_scale=state_channel_scale,
        state_group_weights=state_group_weights,
        mask_objective=mask_objective,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        tversky_gamma=tversky_gamma,
        mask_geometry_weight=mask_geometry_weight,
        mask_weight=mask_weight,
        visual_state_auxiliary_weight=visual_state_auxiliary_weight,
        fusion_state_auxiliary_weight=fusion_state_auxiliary_weight,
        regime_state_weights=regime_state_weights,
        tail_state_cvar_fraction=tail_state_cvar_fraction,
        tail_state_cvar_weight=tail_state_cvar_weight,
        tail_phase_weights=tail_phase_weights,
        tail_vector_auxiliary_weight=tail_vector_auxiliary_weight,
        state_delta_weight=state_delta_weight,
        future_state_weight=future_state_weight,
        future_state_horizon_weights=future_state_horizon_weights,
        inverse_action_weight=inverse_action_weight,
        progress_weight=progress_weight,
        progress_center=progress_center,
        progress_scale=progress_scale,
        action_state_delta_weight=action_state_delta_weight,
        gate_frame_safe_dynamics=gate_frame_safe_dynamics,
    )
    prediction_l1_steps = F.smooth_l1_loss(
        output.prediction.float(), output.target.float(), reduction="none"
    ).mean(dim=(-1, -2))
    prediction_cosine_steps = (
        1.0 - F.cosine_similarity(
            output.prediction.float(), output.target.float(), dim=-1
        ).mean(dim=-1)
    )
    prediction_valid = torch.ones_like(prediction_l1_steps, dtype=torch.bool)
    if gate_consistent_prediction:
        macro_gate_index = _macro_gate_index(batch, output.reconstruction.patch_size)
        if macro_gate_index is not None:
            prediction_valid = macro_gate_index[:, 1:] == macro_gate_index[:, :-1]
    prediction_weights = prediction_valid.to(prediction_l1_steps.dtype)
    prediction_denominator = prediction_weights.sum().clamp_min(1.0)
    prediction_l1 = (prediction_l1_steps * prediction_weights).sum() / prediction_denominator
    prediction_cosine = (
        prediction_cosine_steps * prediction_weights
    ).sum() / prediction_denominator
    prediction = prediction_cosine + float(prediction_l1_weight) * prediction_l1
    if output.unconditioned_prediction is not None:
        unconditioned_l1_steps = F.smooth_l1_loss(
            output.unconditioned_prediction.float(), output.target.float(), reduction="none"
        ).mean(dim=(-1, -2))
        unconditioned_cosine_steps = 1.0 - F.cosine_similarity(
            output.unconditioned_prediction.float(), output.target.float(), dim=-1
        ).mean(dim=-1)
        unconditioned_prediction_l1 = (
            unconditioned_l1_steps * prediction_weights
        ).sum() / prediction_denominator
        unconditioned_prediction_cosine = (
            unconditioned_cosine_steps * prediction_weights
        ).sum() / prediction_denominator
        unconditioned_prediction = (
            unconditioned_prediction_cosine
            + float(prediction_l1_weight) * unconditioned_prediction_l1
        )
    else:
        unconditioned_prediction_l1 = prediction.new_zeros(())
        unconditioned_prediction_cosine = prediction.new_zeros(())
        unconditioned_prediction = prediction.new_zeros(())
    variance, covariance = _representation_regularizers(
        output.reconstruction.latents, variance_target
    )
    grouped_variance, grouped_covariance = _grouped_representation_regularizers(
        output.reconstruction.latents,
        output.reconstruction.latent_group_sizes,
        variance_target,
    )
    total = (
        float(prediction_weight) * prediction
        + float(unconditioned_prediction_weight) * unconditioned_prediction
        + float(reconstruction_weight) * reconstruction
        + float(variance_weight) * variance
        + float(covariance_weight) * covariance
        + float(grouped_variance_weight) * grouped_variance
        + float(grouped_covariance_weight) * grouped_covariance
    )
    mask_target = macro_last(batch["mask"], output.reconstruction.patch_size)
    mask_metrics = binary_segmentation_metrics(
        output.reconstruction.mask_logits.float(), mask_target.float()
    )
    selection_score = (
        float(selection_prediction_weight) * prediction_cosine.detach()
        + 0.25 * float(unconditioned_prediction_weight) * unconditioned_prediction_cosine.detach()
        + float(selection_mask_weight) * (1.0 - mask_metrics["foreground_iou"])
        + reconstruction_metrics["proprio"]
        + reconstruction_metrics["route"]
        + float(selection_state_weight) * reconstruction_metrics["state_huber"]
        + float(visual_state_auxiliary_weight) * reconstruction_metrics["visual_state_huber"]
        + float(fusion_state_auxiliary_weight) * reconstruction_metrics["fusion_state_huber"]
        + float(fusion_state_auxiliary_weight) * float(tail_state_cvar_weight)
        * reconstruction_metrics["fusion_state_cvar"]
        + float(tail_vector_auxiliary_weight) * reconstruction_metrics["tail_vector"]
        + float(selection_belief_weight) * (
            float(state_delta_weight) * reconstruction_metrics["state_delta"]
            + float(future_state_weight) * reconstruction_metrics["future_state"]
            + float(inverse_action_weight) * reconstruction_metrics["inverse_action"]
            + float(progress_weight) * reconstruction_metrics["progress"]
            + float(action_state_delta_weight)
            * reconstruction_metrics["action_state_delta"]
        )
        + 0.1 * reconstruction_metrics["rotation"]
        + 0.1 * variance.detach()
        + 0.01 * covariance.detach()
        + float(grouped_variance_weight) * grouped_variance.detach()
        + float(grouped_covariance_weight) * grouped_covariance.detach()
    )
    metrics = {f"reconstruction_{key}": value for key, value in reconstruction_metrics.items() if key != "loss"}
    metrics.update(
        loss=total.detach(),
        jepa=prediction.detach(),
        jepa_cosine=prediction_cosine.detach(),
        jepa_l1=prediction_l1.detach(),
        unconditioned_jepa=unconditioned_prediction.detach(),
        unconditioned_jepa_cosine=unconditioned_prediction_cosine.detach(),
        unconditioned_jepa_l1=unconditioned_prediction_l1.detach(),
        reconstruction=reconstruction.detach(),
        latent_variance_floor=variance.detach(),
        latent_covariance=covariance.detach(),
        grouped_latent_variance_floor=grouped_variance.detach(),
        grouped_latent_covariance=grouped_covariance.detach(),
        mask_foreground_iou=mask_metrics["foreground_iou"],
        mask_foreground_precision=mask_metrics["foreground_precision"],
        mask_foreground_recall=mask_metrics["foreground_recall"],
        prediction_gate_consistent_fraction=prediction_valid.float().mean().detach(),
        selection_score=selection_score,
    )
    return total, metrics


def action_tokenizer_loss(
    output,
    batch: dict[str, torch.Tensor],
    *,
    prediction_weight: float = 1.0,
    prediction_l1_weight: float = 0.25,
    reconstruction_weight: float = 1.0,
    reconstruction_objective: str = "smooth_l1",
    fixed_reconstruction_std: float = 0.03,
    uncertainty_weight: float = 0.05,
    delta_weight: float = 0.5,
    delta_objective_type: str = "match",
    variance_weight: float = 0.1,
    covariance_weight: float = 0.01,
    variance_target: float = 0.5,
    minimum_std: float = 0.03,
    geometry_weight: float = 0.5,
    order_weight: float = 0.0,
    order_margin: float = 0.25,
    order_min_action_delta: float = 0.01,
    state_effect_weight: float = 0.0,
    state_channel_scale=None,
    state_group_weights=None,
    gate_frame_safe_state_effect: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Action JEPA plus ordered-chunk Gaussian likelihood and transient geometry."""

    reconstruction = output.reconstruction
    actions = batch["applied_action"]
    batch_size, polls = actions.shape[:2]
    patch = reconstruction.patch_size
    if polls % patch:
        raise ValueError("action sequence length must be divisible by the action patch size")
    target = actions.reshape(batch_size, polls // patch, patch, actions.shape[-1])
    if minimum_std <= 0:
        raise ValueError("minimum_std must be positive")
    bounded_log_std = reconstruction.log_std.float().clamp_min(
        torch.log(reconstruction.log_std.new_tensor(float(minimum_std)))
    )
    gnll = (
        gaussian_nll(reconstruction.mean.float(), bounded_log_std, target.float())
        if uncertainty_weight else reconstruction.mean.new_zeros(())
    )
    mean_reconstruction = F.smooth_l1_loss(reconstruction.mean.float(), target.float())
    if fixed_reconstruction_std <= 0:
        raise ValueError("fixed_reconstruction_std must be positive")
    fixed_log_std = reconstruction.mean.new_full(
        reconstruction.mean.shape, math.log(float(fixed_reconstruction_std))
    ).float()
    fixed_precision_reconstruction = gaussian_nll(
        reconstruction.mean.float(), fixed_log_std, target.float()
    )
    if reconstruction_objective == "smooth_l1":
        reconstruction_objective_loss = mean_reconstruction
    elif reconstruction_objective == "mse":
        reconstruction_objective_loss = F.mse_loss(
            reconstruction.mean.float(), target.float()
        )
    elif reconstruction_objective == "fixed_gaussian":
        reconstruction_objective_loss = fixed_precision_reconstruction
    else:
        raise ValueError(
            "reconstruction_objective must be 'smooth_l1', 'mse', or 'fixed_gaussian'"
        )
    if patch > 1:
        predicted_delta = reconstruction.mean[:, :, 1:] - reconstruction.mean[:, :, :-1]
        target_delta = target[:, :, 1:] - target[:, :, :-1]
        delta = F.smooth_l1_loss(predicted_delta.float(), target_delta.float())
        if delta_objective_type == "smooth_l1":
            delta_objective_loss = delta
        elif delta_objective_type == "mse":
            delta_objective_loss = F.mse_loss(
                predicted_delta.float(), target_delta.float()
            )
        elif delta_objective_type == "match" and reconstruction_objective == "fixed_gaussian":
            delta_objective_loss = gaussian_nll(
                predicted_delta.float(),
                predicted_delta.new_full(
                    predicted_delta.shape,
                    math.log(float(fixed_reconstruction_std)),
                ).float(),
                target_delta.float(),
            )
        elif delta_objective_type == "match" and reconstruction_objective == "mse":
            delta_objective_loss = F.mse_loss(predicted_delta.float(), target_delta.float())
        elif delta_objective_type == "match":
            delta_objective_loss = delta
        else:
            raise ValueError("delta_objective_type must be 'match', 'smooth_l1', or 'mse'")
    else:
        predicted_delta = reconstruction.mean[:, 1:, 0] - reconstruction.mean[:, :-1, 0]
        target_delta = target[:, 1:, 0] - target[:, :-1, 0]
        delta = F.smooth_l1_loss(predicted_delta.float(), target_delta.float())
        if delta_objective_type == "mse":
            delta_objective_loss = F.mse_loss(
                predicted_delta.float(), target_delta.float()
            )
        elif delta_objective_type in {"match", "smooth_l1"}:
            delta_objective_loss = delta
        else:
            raise ValueError("delta_objective_type must be 'match', 'smooth_l1', or 'mse'")
    if output.prediction is not None and output.target is not None:
        prediction_l1 = F.smooth_l1_loss(output.prediction.float(), output.target.float())
        prediction_cosine = (
            1.0
            - F.cosine_similarity(output.prediction.float(), output.target.float(), dim=-1)
        ).mean()
        prediction = prediction_cosine + float(prediction_l1_weight) * prediction_l1
    else:
        prediction_l1 = prediction_cosine = prediction = reconstruction.mean.new_zeros(())
    predicted_mean = reconstruction.mean.mean(dim=2)
    target_mean = target.mean(dim=2)
    if patch > 1:
        predicted_endpoint = reconstruction.mean[:, :, -1] - reconstruction.mean[:, :, 0]
        target_endpoint = target[:, :, -1] - target[:, :, 0]
        predicted_variation = (reconstruction.mean[:, :, 1:] - reconstruction.mean[:, :, :-1]).abs().mean((2, 3))
        target_variation = (target[:, :, 1:] - target[:, :, :-1]).abs().mean((2, 3))
        geometry = (
            F.smooth_l1_loss(predicted_mean, target_mean)
            + F.smooth_l1_loss(predicted_endpoint, target_endpoint)
            + F.smooth_l1_loss(predicted_variation, target_variation)
        )
    else:
        geometry = F.smooth_l1_loss(predicted_mean, target_mean)
    if output.reversed_tokens is not None:
        per_patch_order_distance = (
            reconstruction.tokens.float() - output.reversed_tokens.float()
        ).square().mean(dim=(-1, -2)).add(1e-8).sqrt()
        target_order_delta = (
            target.float() - target.float().flip(2)
        ).square().mean(dim=(-1, -2)).sqrt()
        informative_order = target_order_delta >= float(order_min_action_delta)
        order_informative_fraction = informative_order.float().mean()
        if informative_order.any():
            order_distance = per_patch_order_distance[informative_order].mean()
            order_margin_loss = F.relu(
                float(order_margin) - per_patch_order_distance[informative_order]
            ).mean()
        else:
            order_distance = reconstruction.mean.new_zeros(())
            order_margin_loss = reconstruction.mean.new_zeros(())
    else:
        order_distance = reconstruction.mean.new_zeros(())
        order_margin_loss = reconstruction.mean.new_zeros(())
        order_informative_fraction = reconstruction.mean.new_zeros(())
    variance, covariance = _representation_regularizers(reconstruction.tokens, variance_target)
    state_effect = reconstruction.mean.new_zeros(())
    state_effect_valid_fraction = reconstruction.mean.new_ones(())
    if reconstruction.state_delta_mean is not None:
        if "task_state" not in batch or batch["task_state"].shape[1] != polls + 1:
            raise ValueError("state-effect objective requires T+1 task_state samples")
        state_scale = _loss_scale(
            reconstruction.state_delta_mean, state_channel_scale, reconstruction.state_delta_mean.shape[-1]
        )
        valid = torch.ones(
            reconstruction.state_delta_mean.shape[:-1],
            device=reconstruction.state_delta_mean.device, dtype=torch.bool,
        )
        if gate_frame_safe_state_effect and "gate_index" in batch:
            gate_index = batch["gate_index"]
            if gate_index.ndim == 3:
                gate_index = gate_index[..., 0]
            valid = gate_index[:, 1:] == gate_index[:, :-1]
        state_effect_valid_fraction = valid.float().mean()
        state_effect = _masked_weighted_task_state_loss(
            reconstruction.state_delta_mean,
            batch["task_state"][:, 1:] - batch["task_state"][:, :-1],
            state_scale, state_group_weights, valid,
        )
    total = (
        float(prediction_weight) * prediction
        + float(reconstruction_weight) * reconstruction_objective_loss
        + float(uncertainty_weight) * gnll
        + float(delta_weight) * delta_objective_loss
        + float(geometry_weight) * geometry
        + float(order_weight) * order_margin_loss
        + float(variance_weight) * variance
        + float(covariance_weight) * covariance
        + float(state_effect_weight) * state_effect
    )
    selection_score = (
        mean_reconstruction.detach()
        + float(delta_weight) * delta.detach()
        + float(geometry_weight) * geometry.detach()
        + float(order_weight) * order_margin_loss.detach()
        + float(variance_weight) * variance.detach()
        + float(covariance_weight) * covariance.detach()
        + float(state_effect_weight) * state_effect.detach()
    )
    return total, {
        "loss": total.detach(),
        "jepa": prediction.detach(),
        "jepa_cosine": prediction_cosine.detach(),
        "jepa_l1": prediction_l1.detach(),
        "reconstruction": mean_reconstruction.detach(),
        "reconstruction_objective": reconstruction_objective_loss.detach(),
        "fixed_precision_reconstruction": fixed_precision_reconstruction.detach(),
        "gnll": gnll.detach(),
        "delta": delta.detach(),
        "delta_objective": delta_objective_loss.detach(),
        "geometry": geometry.detach(),
        "order_distance": order_distance.detach(),
        "order_margin_loss": order_margin_loss.detach(),
        "order_informative_fraction": order_informative_fraction.detach(),
        "mean_log_std": bounded_log_std.detach().mean(),
        "latent_variance_floor": variance.detach(),
        "latent_covariance": covariance.detach(),
        "state_effect": state_effect.detach(),
        "state_effect_gate_consistent_fraction": state_effect_valid_fraction.detach(),
        "selection_score": selection_score,
    }


def actor_bc_gnll_loss(output, target_action: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Behavior cloning objective for a diagonal-Gaussian action policy."""

    loss = gaussian_nll(output.mean.float(), output.log_std.float(), target_action.float())
    mse = F.mse_loss(output.mean.float(), target_action.float())
    return loss, {
        "loss": loss.detach(),
        "mean_mse": mse.detach(),
        "mean_std": output.log_std.detach().exp().mean(),
    }


def critic_twohot_loss(critic, state: torch.Tensor, target_return: torch.Tensor):
    logits = critic(state)
    loss = critic.coder.loss(logits, target_return)
    return loss, {"loss": loss.detach(), "value": critic.coder.mean(logits).detach().mean()}


def actor_value_loss(value: torch.Tensor, *, entropy: torch.Tensor | None = None, entropy_scale: float = 0.0):
    """Dreamer-style actor objective; gradients flow through imagined values."""

    objective = value.mean()
    if entropy is not None and entropy_scale:
        objective = objective + float(entropy_scale) * entropy.mean()
    loss = -objective
    return loss, {"loss": loss.detach(), "imagined_value": value.detach().mean()}


@dataclass(frozen=True)
class ShortcutBatch:
    noisy: torch.Tensor
    target: torch.Tensor
    step_indices: torch.Tensor
    signal_indices: torch.Tensor


def make_shortcut_batch(clean: torch.Tensor, k_max: int = 8) -> ShortcutBatch:
    """Construct diffusion-forcing inputs with per-time independent noise levels."""

    batch, time = clean.shape[:2]
    source = torch.randn_like(clean)
    signal = torch.randint(0, k_max + 1, (batch, time), device=clean.device)
    tau = signal.to(clean.dtype) / k_max
    while tau.ndim < clean.ndim:
        tau = tau.unsqueeze(-1)
    noisy = source + tau * (clean - source)
    max_step_bin = int(torch.tensor(k_max).log2().item())
    step = torch.randint(0, max_step_bin + 1, (batch, time), device=clean.device)
    return ShortcutBatch(noisy=noisy, target=clean, step_indices=step, signal_indices=signal)


def shortcut_forcing_objective(
    model,
    clean: torch.Tensor,
    actions: torch.Tensor,
    *,
    initial_context: torch.Tensor | None = None,
    action_context: torch.Tensor | None = None,
    direct_weight: float = 0.25,
    bootstrap_weight: float = 1.0,
):
    """Dreamer-4 shortcut forcing with half-step bootstrap velocity targets."""

    batch, time = clean.shape[:2]
    bins = int(torch.tensor(model.k_max).log2().item()) + 1
    step_indices = torch.randint(0, bins, (batch, time), device=clean.device)
    step_size = (2.0 ** step_indices.to(clean.dtype)) / model.k_max
    cells = (1.0 / step_size).to(torch.long).clamp_min(1)
    cell = (torch.rand(batch, time, device=clean.device) * cells).to(torch.long)
    tau_scalar = cell.to(clean.dtype) * step_size
    signal = (tau_scalar * model.k_max).round().to(torch.long).clamp(0, model.k_max)
    tau = tau_scalar
    while tau.ndim < clean.ndim:
        tau = tau.unsqueeze(-1)
    source = torch.randn_like(clean)
    noisy = source + tau * (clean - source)
    output = model(
        noisy, actions, step_indices, signal,
        initial_context=initial_context, action_context=action_context,
    )
    # torch.compile's CUDA Graphs reuse output buffers across invocations. This
    # objective invokes the same model again for the half-step teacher, so keep
    # the trainable first-pass heads and latents in independent storage. Without
    # this boundary clone inference-mode validation reads overwritten tensors.
    output = replace(
        output,
        **{
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in vars(output).items()
        },
    )
    direct_error = (output.predicted_packed_latents - clean).square().flatten(2).mean(-1)

    bootstrap_mask = step_indices > 0
    if bootstrap_mask.any():
        half_indices = (step_indices - 1).clamp_min(0)
        with torch.no_grad():
            first = model(
                noisy, actions, half_indices, signal,
                initial_context=initial_context, action_context=action_context,
            ).predicted_packed_latents.clone()
            denominator = (1.0 - tau).clamp_min(1.0 / model.k_max)
            first_velocity = (first - noisy) / denominator
            midpoint = noisy + (step_size / 2.0).view(batch, time, *([1] * (clean.ndim - 2))) * first_velocity
            midpoint_tau_scalar = tau_scalar + step_size / 2.0
            midpoint_signal = (midpoint_tau_scalar * model.k_max).round().to(torch.long).clamp(0, model.k_max)
            second = model(
                midpoint, actions, half_indices, midpoint_signal,
                initial_context=initial_context, action_context=action_context,
            ).predicted_packed_latents
            midpoint_tau = midpoint_tau_scalar
            while midpoint_tau.ndim < clean.ndim:
                midpoint_tau = midpoint_tau.unsqueeze(-1)
            second_velocity = (second - midpoint) / (1.0 - midpoint_tau).clamp_min(1.0 / model.k_max)
            target_velocity = 0.5 * (first_velocity + second_velocity)
        predicted_velocity = (output.predicted_packed_latents - noisy) / denominator
        bootstrap_error = (
            (1.0 - tau_scalar).square()
            * (predicted_velocity - target_velocity).square().flatten(2).mean(-1)
        )
        # The bootstrap target teaches shortcut consistency, but is not an
        # observation target. Retaining a direct endpoint term at every step
        # prevents a self-consistent teacher from drifting away from data.
        error = float(direct_weight) * direct_error + torch.where(
            bootstrap_mask,
            float(bootstrap_weight) * bootstrap_error,
            direct_error,
        )
    else:
        error = direct_error
    return output, error.mean()


def dynamics_loss(
    output,
    target_packed: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    latent_loss: torch.Tensor | None = None,
    reconstruction=None,
    reconstruction_weight: float = 0.1,
    continuation_negative_weight: float = 10.0,
    continuation_focal_gamma: float = 0.0,
    continuation_loss_mode: str = "weighted_bce",
    continuation_brier_weight: float = 0.0,
    continuation_balanced_weight: float = 1.0,
    continuation_calibration_weight: float = 0.0,
    continuation_ranking_weight: float = 0.0,
    continuation_ranking_margin: float = 0.0,
    continuation_label_smoothing: float = 0.0,
    latent_weight: float = 1.0,
    reward_weight: float = 1.0,
    continuation_weight: float = 1.0,
    task_state_weight: float = 0.1,
    task_state_objective: str = "absolute_uncertainty",
    task_state_group_weights=None,
    reward_tail_threshold: float = 0.0,
    reward_tail_weight: float = 0.0,
    reward_decoded_weight: float = 0.0,
    reward_distribution_weight: float = 1.0,
    reward_direct_weight: float = 0.0,
    reward_direct_tail_weight: float = 0.0,
    privileged_state_weight: float = 0.0,
    estimator_state_weight: float = 0.0,
    estimator_std_weight: float = 0.0,
    privileged_state_center=None,
    privileged_state_scale=None,
    estimator_state_center=None,
    estimator_state_scale=None,
    estimator_std_center=None,
    estimator_std_scale=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    latent = F.mse_loss(output.predicted_packed_latents, target_packed) if latent_loss is None else latent_loss
    reward_coder = TwoHot(output.reward_logits.shape[-1]).to(output.reward_logits.device)
    reward_ce = reward_coder.loss(output.reward_logits, batch["reward"])
    reward_distribution_prediction = reward_coder.mean(output.reward_logits)
    reward_tail = latent.new_zeros(())
    tail_mask = batch["reward"].abs() >= float(reward_tail_threshold)
    if float(reward_tail_weight) > 0 and tail_mask.any():
        reward_tail = reward_coder.loss(output.reward_logits, batch["reward"], mask=tail_mask)
    reward_distribution = reward_ce + float(reward_tail_weight) * reward_tail
    reward_decoded = F.smooth_l1_loss(reward_distribution_prediction, batch["reward"])
    reward = float(reward_distribution_weight) * reward_distribution
    if float(reward_decoded_weight) > 0:
        reward = reward + float(reward_decoded_weight) * reward_decoded
    reward_direct = latent.new_zeros(())
    if output.reward_value is not None:
        direct_error = F.smooth_l1_loss(
            output.reward_value.float(), batch["reward"].float(), reduction="none"
        )
        direct_weights = 1.0 + float(reward_direct_tail_weight) * tail_mask.to(direct_error.dtype)
        reward_direct = (direct_error * direct_weights).sum() / direct_weights.sum().clamp_min(1.0)
        reward = reward + float(reward_direct_weight) * reward_direct
        reward_prediction = output.reward_value.float()
    else:
        reward_prediction = reward_distribution_prediction
    reward_mae = (reward_prediction - batch["reward"]).abs().mean()
    reward_distribution_mae = (
        reward_distribution_prediction - batch["reward"]
    ).abs().mean()
    continue_target = batch["continue"].to(output.continue_logits.dtype)
    smoothing = float(continuation_label_smoothing)
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("continuation_label_smoothing must be in [0,1)")
    continuation_loss_target = (
        continue_target * (1.0 - smoothing) + 0.5 * smoothing
    )
    continuation_raw = F.binary_cross_entropy_with_logits(
        output.continue_logits, continue_target, reduction="none"
    )
    continuation_loss_raw = F.binary_cross_entropy_with_logits(
        output.continue_logits, continuation_loss_target, reduction="none"
    )
    terminal_mask = continue_target < 0.5
    nonterminal_mask = ~terminal_mask
    continuation_brier = continuation_raw.new_zeros(())
    continuation_balanced = continuation_raw.new_zeros(())
    continuation_calibration = continuation_loss_raw.mean()
    continuation_ranking = continuation_raw.new_zeros(())
    if continuation_loss_mode in {"balanced_hazard", "calibrated_hazard"}:
        class_losses = []
        class_briers = []
        probability = output.continue_logits.sigmoid()
        for mask in (terminal_mask, nonterminal_mask):
            if mask.any():
                class_losses.append(continuation_loss_raw[mask].mean())
                class_briers.append((probability[mask] - continue_target[mask]).square().mean())
        continuation_balanced = torch.stack(class_losses).mean()
        continuation_brier = torch.stack(class_briers).mean()
        if continuation_loss_mode == "balanced_hazard":
            continuation = continuation_balanced
        else:
            terminal_logits = output.continue_logits[terminal_mask]
            nonterminal_logits = output.continue_logits[nonterminal_mask]
            if terminal_logits.numel() and nonterminal_logits.numel():
                # Continue logits should rank every nonterminal above every
                # terminal. Softplus is a smooth pairwise AUC surrogate.
                continuation_ranking = F.softplus(
                    terminal_logits[:, None] - nonterminal_logits[None, :]
                    + float(continuation_ranking_margin)
                ).mean()
            continuation = (
                float(continuation_balanced_weight) * continuation_balanced
                + float(continuation_calibration_weight) * continuation_calibration
                + float(continuation_ranking_weight) * continuation_ranking
            )
        continuation = continuation + float(continuation_brier_weight) * continuation_brier
    elif continuation_loss_mode == "weighted_bce":
        continuation_weights = torch.where(
            terminal_mask,
            continuation_raw.new_tensor(continuation_negative_weight),
            continuation_raw.new_tensor(1.0),
        )
        if float(continuation_focal_gamma) > 0:
            probability_target = torch.where(
                continue_target > 0.5,
                output.continue_logits.sigmoid(),
                (-output.continue_logits).sigmoid(),
            )
            continuation_weights = continuation_weights * (
                1.0 - probability_target
            ).pow(float(continuation_focal_gamma))
        continuation = (
            continuation_loss_raw * continuation_weights
        ).sum() / continuation_weights.sum().clamp_min(1.0)
    else:
        raise ValueError(f"unknown continuation_loss_mode: {continuation_loss_mode}")
    state_target = batch["task_state"][:, 1:]
    if task_state_objective == "structured_delta":
        anchor = batch["task_state"][:, :1]
        state_prediction = output.task_state_mean.clone()
        # Position, velocity, rates and motor speed are residual dynamics. The
        # gate-frame orientation is already a continuous 6D rotation and is
        # predicted absolutely so that a geodesic loss remains well-defined.
        state_prediction[..., 0:6] = anchor[..., 0:6] + output.task_state_mean[..., 0:6]
        state_prediction[..., 12:19] = anchor[..., 12:19] + output.task_state_mean[..., 12:19]
        groups = ((0, 3), (3, 6), (6, 12), (12, 15), (15, 19))
        group_weights = task_state_group_weights or (1.0, 1.0, 1.0, 1.0, 0.5)
        if len(group_weights) != len(groups):
            raise ValueError("task_state_group_weights must contain five values")
        group_losses = []
        for index, ((start, stop), weight) in enumerate(zip(groups, group_weights)):
            if index == 2:
                value = rotation_geodesic_loss(
                    state_prediction[..., start:stop], state_target[..., start:stop]
                )
            else:
                value = F.smooth_l1_loss(
                    state_prediction[..., start:stop], state_target[..., start:stop]
                )
            group_losses.append(float(weight) * value)
        state = torch.stack(group_losses).sum() / max(sum(map(float, group_weights)), 1e-6)
        state_nll = latent.new_zeros(())
        state_mean = F.smooth_l1_loss(state_prediction, state_target)
    elif task_state_objective == "absolute_uncertainty":
        state_prediction = output.task_state_mean
        state_nll = heteroscedastic_loss(
            output.task_state_mean, output.task_state_log_scale, state_target
        )
        state_mean = F.smooth_l1_loss(output.task_state_mean, state_target)
        state = 0.5 * state_nll + 0.5 * state_mean
    else:
        raise ValueError(f"unknown task_state_objective: {task_state_objective}")
    auxiliary = latent.new_zeros(())
    auxiliary_metrics: dict[str, torch.Tensor] = {}

    def normalized_auxiliary(
        prediction, target_key: str, center_values, scale_values, metric_name: str
    ):
        if prediction is None or target_key not in batch:
            return latent.new_zeros(()), {}
        target = batch[target_key][:, 1:, : prediction.shape[-1]].float()
        center = target.new_tensor(center_values if center_values is not None else [0.0] * target.shape[-1])
        scale = target.new_tensor(scale_values if scale_values is not None else [1.0] * target.shape[-1])
        if center.shape != target.shape[-1:] or scale.shape != target.shape[-1:] or torch.any(scale <= 0):
            raise ValueError(f"invalid normalization for {metric_name}")
        normalized_target = (target - center) / scale
        objective = F.smooth_l1_loss(prediction.float(), normalized_target)
        decoded = prediction.float() * scale + center
        return objective, {
            f"{metric_name}_loss": objective.detach(),
            f"{metric_name}_mae": (decoded - target).abs().mean().detach(),
        }

    privileged_state, values = normalized_auxiliary(
        output.privileged_state, "privileged_state", privileged_state_center,
        privileged_state_scale, "privileged_state",
    )
    auxiliary_metrics.update(values)
    estimator_state, values = normalized_auxiliary(
        output.estimator_state, "privileged_state_estimate", estimator_state_center,
        estimator_state_scale, "estimator_state",
    )
    auxiliary_metrics.update(values)
    estimator_std, values = normalized_auxiliary(
        output.estimator_std, "privileged_state_estimate_std", estimator_std_center,
        estimator_std_scale, "estimator_std",
    )
    auxiliary_metrics.update(values)
    auxiliary = (
        float(privileged_state_weight) * privileged_state
        + float(estimator_state_weight) * estimator_state
        + float(estimator_std_weight) * estimator_std
    )
    reconstruction_loss = latent.new_zeros(())
    reconstruction_metrics: dict[str, torch.Tensor] = {}
    if reconstruction is not None:
        patch = int(reconstruction.patch_size)
        mask_target = macro_last(batch["mask"], patch)[:, 1:]
        proprio_target = macro_last(batch["proprio"], patch)[:, 1:]
        route_target = macro_last(batch["route"], patch)[:, 1:]
        mask_bce = F.binary_cross_entropy_with_logits(reconstruction.mask_logits.float(), mask_target.float())
        mask_dice = dice_loss(reconstruction.mask_logits.float(), mask_target.float())
        proprio_reconstruction = F.smooth_l1_loss(reconstruction.proprio.float(), proprio_target.float())
        route_reconstruction = F.smooth_l1_loss(reconstruction.route.float(), route_target.float())
        reconstruction_loss = mask_bce + mask_dice + proprio_reconstruction + route_reconstruction
        reconstruction_metrics = {
            "reconstruction_mask_bce": mask_bce.detach(),
            "reconstruction_mask_dice": mask_dice.detach(),
            "reconstruction_proprio": proprio_reconstruction.detach(),
            "reconstruction_route": route_reconstruction.detach(),
        }
    total = (
        float(latent_weight) * latent
        + float(reward_weight) * reward
        + float(continuation_weight) * continuation
        + float(task_state_weight) * state
        + auxiliary
        + float(reconstruction_weight) * reconstruction_loss
    )
    probability = output.continue_logits.sigmoid()
    terminal = terminal_mask
    nonterminal = nonterminal_mask
    terminal_probability = (1.0 - probability)[terminal].mean() if terminal.any() else probability.new_zeros(())
    terminal_bce = continuation_raw[terminal].mean() if terminal.any() else probability.new_zeros(())
    terminal_recall = (probability[terminal] < 0.5).float().mean() if terminal.any() else probability.new_zeros(())
    nonterminal_bce = continuation_raw[nonterminal].mean() if nonterminal.any() else probability.new_zeros(())
    nonterminal_false_stop_probability = (
        (1.0 - probability)[nonterminal].mean()
        if nonterminal.any() else probability.new_zeros(())
    )
    metrics = {
        "loss": total.detach(),
        "latent": latent.detach(),
        "reward_ce": reward_ce.detach(),
        "reward_objective": reward.detach(),
        "reward_tail_ce": reward_tail.detach(),
        "reward_decoded": reward_decoded.detach(),
        "reward_direct": reward_direct.detach(),
        "reward_mae": reward_mae.detach(),
        "reward_distribution_mae": reward_distribution_mae.detach(),
        "continue_bce": continuation.detach(),
        "continue_balanced_bce": continuation_balanced.detach(),
        "continue_calibration_bce": continuation_calibration.detach(),
        "continue_ranking": continuation_ranking.detach(),
        "continue_brier": continuation_brier.detach(),
        "continue_probability": probability.mean().detach(),
        "terminal_probability": terminal_probability.detach(),
        "terminal_bce": terminal_bce.detach(),
        "terminal_recall": terminal_recall.detach(),
        "terminal_count": terminal.sum().detach(),
        "nonterminal_bce": nonterminal_bce.detach(),
        "nonterminal_false_stop_probability": nonterminal_false_stop_probability.detach(),
        "nonterminal_count": nonterminal.sum().detach(),
        "state": state.detach(),
        "state_mae": (state_prediction - state_target).abs().mean().detach(),
        "state_mean_log_std": output.task_state_log_scale.mean().detach(),
        "reconstruction": reconstruction_loss.detach(),
        "auxiliary": auxiliary.detach(),
    }
    metrics.update(reconstruction_metrics)
    metrics.update(auxiliary_metrics)
    return total, metrics


def flow_matching_loss(predicted_velocity: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted_velocity, target_velocity)


def lambda_returns(
    reward: torch.Tensor,
    value: torch.Tensor,
    continuation: torch.Tensor,
    *,
    discount: float = 0.997,
    lambda_: float = 0.95,
) -> torch.Tensor:
    if value.shape[-1] != reward.shape[-1] + 1:
        raise ValueError("value must contain one bootstrap step beyond reward")
    returns = torch.empty_like(reward)
    carry = value[..., -1]
    for index in range(reward.shape[-1] - 1, -1, -1):
        bootstrap = (1.0 - lambda_) * value[..., index + 1] + lambda_ * carry
        carry = reward[..., index] + discount * continuation[..., index] * bootstrap
        returns[..., index] = carry
    return returns
