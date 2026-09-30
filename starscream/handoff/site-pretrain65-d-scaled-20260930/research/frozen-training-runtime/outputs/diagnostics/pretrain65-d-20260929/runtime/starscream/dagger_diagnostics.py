"""Read-only fitting diagnostics with an independent sampling RNG."""
import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def fresh_action_metrics(policy, histories, actions, speeds, settings, *, seed):
    count = min(int(settings.get('dagger_fresh_loss_rows', 0)), len(histories))
    if count <= 0:
        return {}
    from starscream.imitation_objective import action_dimension_weights
    from scripts.train_privileged_racing import normalized_ctbr_tensor
    indices = np.random.default_rng(seed).choice(len(histories), count, replace=False)
    device = next(policy.parameters()).device
    was_training = policy.training
    policy.eval()
    errors, physical = [], []
    try:
        for start in range(0, count, 256):
            rows = indices[start:start+256]
            x = torch.as_tensor(np.asarray(histories[rows], np.float32), device=device)
            y = torch.as_tensor(actions[rows], device=device)
            speed = torch.as_tensor(speeds[rows], device=device)
            pred = policy(x, speed).float()
            errors.append(F.smooth_l1_loss(pred, y, beta=float(settings.get('huber_beta', .05)),
                                          reduction='none').cpu())
            physical.append((normalized_ctbr_tensor(pred, settings) - normalized_ctbr_tensor(y, settings)).abs().cpu())
    finally:
        policy.train(was_training)
    losses = torch.cat(errors)
    mae = torch.cat(physical).mean(0)
    weighted = losses * action_dimension_weights(settings, losses)
    return {'fresh/rows': count, 'fresh/action_loss_unweighted': float(losses.mean()),
            'fresh/action_loss_weighted': float(weighted.mean()),
            **{f'fresh/{name}_loss': float(losses[:, i].mean()) for i, name in enumerate(('thrust', 'roll', 'pitch', 'yaw'))},
            **{f'fresh/{name}_physical_mae': float(mae[i]) for i, name in enumerate(('thrust', 'roll', 'pitch', 'yaw'))}}
