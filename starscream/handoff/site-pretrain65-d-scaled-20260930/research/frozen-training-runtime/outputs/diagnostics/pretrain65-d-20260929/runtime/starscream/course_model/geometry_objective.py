"""Optional directed-opening fidelity terms; zero weights preserve base VAE."""
import torch
from torch.nn import functional as F
from .model import rotation6


def geometry_terms(raw,target,mask,context):
    raw=raw.float();target=target.float();context=context.float()
    delta=raw[...,:3]-target[...,:3]
    frame=rotation6(target[...,3:9])
    local=torch.einsum('bnij,bni->bnj',frame,delta)
    half=.5*target[...,9:11].exp().clamp_min(.1)
    transverse=F.smooth_l1_loss(local[...,1:]/half,torch.zeros_like(local[...,1:]),reduction='none').mean(-1)
    mean=lambda x:((x*mask).sum(-1)/mask.sum(-1).clamp_min(1)).mean()
    # Smooth maximum via max of nonnegative per-gate Huber errors, including
    # longitudinal displacement. Not an unbounded squared outlier penalty.
    scaled=delta/half.amin(-1,keepdim=True)
    per_gate=F.smooth_l1_loss(scaled,torch.zeros_like(scaled),reduction='none').mean(-1)
    worst=per_gate.masked_fill(~mask,0).amax(-1).mean()
    last=(mask.sum(-1)-1).clamp_min(0);batch=torch.arange(len(raw),device=raw.device)
    pred_edge=raw[:,0,:3]-raw[batch,last,:3];true_edge=target[:,0,:3]-target[batch,last,:3]
    cyclic=(((pred_edge-true_edge)/20).square().mean(-1)*context[:,12]).mean()
    pred_frame=rotation6(raw[...,3:9]);sizes=raw[...,9:11].clamp(-3,3).exp()
    extent=.5*(pred_frame[...,2,1:].abs()*sizes).sum(-1)
    penetration=(extent-raw[...,2]).clamp_min(0)
    floor=mean(F.smooth_l1_loss(penetration,torch.zeros_like(penetration),reduction='none'))
    return dict(transverse=mean(transverse),worst_gate=worst,cyclic=cyclic,floor=floor)
