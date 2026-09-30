"""Classical local inverse step over a learned chart. No learned planner."""
import torch


def local_inverse_step(jacobian,desired_change,metric=None,radius=.2,damping=.01):
    """Damped local least-squares direction with trust-region norm projection.

    Caller must align/count-match semantic target features, re-decode, validate,
    and admit with MPCC/policy. This solves a local surrogate, not a geodesic or
    globally optimal path. The desired change is NOT inserted into a track.
    """
    if radius<=0 or damping<=0:raise ValueError('positive trust radius/damping required')
    j=jacobian;eye=torch.eye(j.shape[-1],device=j.device,dtype=j.dtype)
    g=eye if metric is None else metric
    # A pullback can be rank deficient (e.g. latent64, six gates with only
    # 48 independent continuous DOFs). Metric-only trust permits arbitrarily
    # large null-space jumps. Euclidean damping AND trust are both required.
    step=torch.linalg.solve(j.T@j+damping*(g+eye),j.T@desired_change)
    norm=(step@g@step).clamp_min(0).sqrt()
    bound=torch.maximum(norm,step.norm()).clamp_min(1e-12)
    return step*torch.clamp(torch.as_tensor(radius,device=j.device)/bound,max=1)
