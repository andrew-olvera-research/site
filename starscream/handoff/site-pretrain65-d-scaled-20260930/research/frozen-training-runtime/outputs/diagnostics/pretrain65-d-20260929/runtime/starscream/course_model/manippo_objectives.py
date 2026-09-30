"""Opt-in task metric and online latent-neighborhood static regularization.

Not policy learning, MPCC supervision, or a guarantee of flyability. All
reconstruction anchors remain active; generated examples are never expert labels.
"""
import torch
from torch.nn import functional as F
from .model import rotation6


def task_metric_loss(mu,target):
    """Match paired task distances within homogeneous-count batches.

    Absolute physical coordinates, proper frames, sizes and crossing flags
    contribute. This declared proxy is NOT an intrinsic control metric.
    """
    mu=mu.float();t=target.float()
    other=torch.roll(torch.arange(len(mu),device=mu.device),1)
    rot=rotation6(t[...,3:9]).flatten(-2)
    features=torch.cat([t[...,:3]/20,rot*.5,t[...,9:11]*.5,t[...,11:17]*.5],-1)
    distance=(features-features[other]).square().mean((1,2)).detach()
    latent=(mu-mu[other]).square().mean(-1)
    return F.smooth_l1_loss(latent,distance,beta=.1)


def decoded_constraints(raw,context):
    """Physical static barriers on raw decoder samples, FP32 reductions."""
    raw=raw.float();context=context.float()
    rot=rotation6(raw[...,3:9]);size=raw[...,9:11].clamp(-3,3).exp()
    half_extent=.5*(rot[...,2,1:].abs()*size).sum(-1)
    floor=(half_extent-raw[...,2]).clamp_min(0).square().mean()
    # Fixed generic aperture range; no target-specific geometry.
    aperture=(.5-size).clamp_min(0).square().mean()+(size-4.).clamp_min(0).square().mean()
    spacing=(raw[:,1:,:3]-raw[:,:-1,:3]).norm(dim=-1)
    separation=(.5-spacing).clamp_min(0).square().mean()
    closure=(raw[:,0,:3]-raw[:,-1,:3]).norm(dim=-1)
    separation=separation+((.5-closure).clamp_min(0).square()*context[:,12]).mean()
    return floor+aperture+separation


def auxiliary_losses(model,out,target,context,settings):
    result={}
    metric_weight=float(settings.get('latent_metric_weight',0.))
    generated_weight=float(settings.get('generated_constraint_weight',0.))
    if metric_weight<0 or generated_weight<0:raise ValueError('negative auxiliary weight')
    if metric_weight:
        result['latent_metric']=metric_weight*task_metric_loss(out['mu'],target)
    if generated_weight:
        # Stop-gradient center prevents encoder escape from sampled violations.
        # Fresh samples each batch, not a frozen bank or offline pseudo-labels.
        radius=float(settings.get('generated_radius',.25))
        if radius<=0:raise ValueError('generated radius must be positive')
        z=out['mu'].detach()+radius*F.normalize(torch.randn_like(out['mu']),dim=-1)
        counts=torch.full((len(z),),target.shape[1],device=z.device,dtype=torch.long)
        raw,_=model.decode(z,counts,context,_length=target.shape[1])
        result['generated_constraints']=generated_weight*decoded_constraints(raw,context)
    return result
