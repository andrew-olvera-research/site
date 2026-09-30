"""Ordered relation supervision: explicit physical scales, not grammar claims."""
import torch
from torch.nn import functional as F
from .model import rotation6


def relational_terms(raw,target,mask,context):
    raw=raw.float().masked_fill(~mask[...,None],0);target=target.float().masked_fill(~mask[...,None],0)
    p=raw[...,:3];t=target[...,:3];b,n,_=p.shape
    idx=torch.arange(n,device=p.device)[None].expand(b,-1)
    nxt=(idx+1)%mask.sum(1)[:,None]
    gather=lambda x:x.gather(1,nxt[...,None].expand(-1,-1,x.shape[-1]))
    pe=gather(p)-p;te=gather(t)-t
    valid=mask & ((idx<mask.sum(1)[:,None]-1)|(context[:,12:13]>.5))
    den=te.norm(dim=-1,keepdim=True).clamp_min(.5)
    mean=lambda x:(x*valid).sum()/valid.sum().clamp_min(1)
    edge=mean(F.smooth_l1_loss(pe/den,te/den,reduction='none',beta=.1).mean(-1))
    pr=rotation6(raw[...,3:9].float());tr=rotation6(target[...,3:9].float())
    # Ordered edge components in each gate frame encode sidedness and relative
    # vertical/lateral relation. These are chords, NOT MPCC approach velocities.
    a=(pr.transpose(-1,-2)@pe[...,None]).squeeze(-1)/den
    c=(tr.transpose(-1,-2)@te[...,None]).squeeze(-1)/den
    frame=mean(F.smooth_l1_loss(a,c,reduction='none',beta=.1).mean(-1))
    pa=F.normalize(pe,dim=-1);ta=F.normalize(te,dim=-1)
    predturn=torch.cross(pa,gather(pa),dim=-1);trueturn=torch.cross(ta,gather(ta),dim=-1)
    # Require both edges; exclude last non-loop successor turn.
    turnmask=valid & valid.gather(1,nxt)
    turn=(F.smooth_l1_loss(predturn,trueturn,reduction='none',beta=.1).mean(-1)*turnmask).sum()/turnmask.sum().clamp_min(1)
    return dict(edge=edge,gate_frame=frame,signed_turn=turn)
