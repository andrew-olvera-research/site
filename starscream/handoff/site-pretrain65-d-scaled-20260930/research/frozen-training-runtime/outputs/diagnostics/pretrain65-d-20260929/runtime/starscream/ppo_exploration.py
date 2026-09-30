"""Explicit, bounded direct-Gaussian exploration controls (latent action units)."""
import numpy as np
import torch


def noise_bounds(settings, policy):
    config = settings.get('ppo_noise_bounds', {})
    low = np.broadcast_to(np.asarray(config.get('minimum', policy.minimum_log_std), float), (4,)).copy()
    high = np.broadcast_to(np.asarray(config.get('maximum', policy.maximum_log_std), float), (4,)).copy()
    if (not np.isfinite(low).all() or not np.isfinite(high).all()
            or np.any(low > high) or np.any(low < policy.minimum_log_std)
            or np.any(high > policy.maximum_log_std)):
        raise ValueError('noise bounds must be finite, ordered and inside policy log-std bounds')
    return low, high


@torch.no_grad()
def prepare_noise_bounds(settings, policy):
    """Upload bounds ONCE, not four CPU scalars after every GPU optimizer step."""
    low, high = noise_bounds(settings, policy)
    p = policy.log_std_parameter
    policy._ppo_noise_bounds_tensors = (
        torch.tensor(low.tolist(),device=p.device,dtype=p.dtype),
        torch.tensor(high.tolist(),device=p.device,dtype=p.dtype))


@torch.no_grad()
def project_noise_bounds(settings, policy):
    if not hasattr(policy, '_ppo_noise_bounds_tensors'):
        prepare_noise_bounds(settings, policy)
    lo, hi = policy._ppo_noise_bounds_tensors
    policy.log_std_parameter.clamp_(min=lo,max=hi)


def noise_optimizer_group(settings, parameter, actor_lr):
    scale = float(settings.get('exploration_learning_rate_scale', 1.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('exploration_learning_rate_scale must be finite and positive')
    parameter.requires_grad_(True)
    return dict(params=[parameter], lr=actor_lr*scale, lr_scale=scale,
                weight_decay=0.0)
