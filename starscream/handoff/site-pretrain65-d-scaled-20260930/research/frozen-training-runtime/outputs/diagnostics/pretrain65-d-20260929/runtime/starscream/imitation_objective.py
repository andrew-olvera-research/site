"""Shared, capture-safe reductions for supervised DAgger objectives."""
import math

import torch


def action_dimension_weights(settings, reference):
    values = settings.get('action_dimension_weights', [1., 1., 1., 1.])
    if (len(values) != 4 or any(not math.isfinite(float(v)) or float(v) < 0 for v in values)
            or sum(values) <= 0):
        raise ValueError('action_dimension_weights requires four finite nonnegative values with positive sum')
    # CPU validation and normalization; no GPU readback inside CUDA capture.
    values = [4. * float(v) / sum(values) for v in values]
    return torch.as_tensor(settings.get('_dagger_action_dimension_weights', values),
                           device=reference.device, dtype=reference.dtype)


def masked_objective(values, valid):
    """Keep the scalar AND its gradient equal to a mean over valid rows.

    Detached mean imputation alone preserves the displayed value but dilutes
    gradients by valid_count / batch_size. Preserve imputation for robust-group
    scoring while correcting its gradient, including the all-invalid case.
    """
    mask = valid.to(values.dtype).reshape(-1)
    if mask.shape != values.shape:
        raise ValueError('objective validity mask must align with rows')
    selected = torch.where(mask > 0, values, torch.zeros_like(values))
    mean = selected.sum() / mask.sum().clamp_min(1.)
    corrected = selected * (len(values) / mask.sum().clamp_min(1.))
    display = torch.where(mask > 0, values, mean.detach())
    return mean, display.detach() + corrected - corrected.detach()
