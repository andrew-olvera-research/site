"""Small ordered-geometry demand and policy-conditioned readiness predictors."""
import torch
from torch import nn
from torch.nn import functional as F
from .models import SwiBlock


class GateBlock(nn.Module):
    def __init__(self,width):
        super().__init__();self.width=width;self.norm=nn.RMSNorm(width)
        self.qkv=nn.Linear(width,3*width);self.out=nn.Linear(width,width);self.ff=SwiBlock(width)
    def forward(self,x,mask):
        b,n,w=x.shape
        q,k,v=self.qkv(self.norm(x)).reshape(b,n,3,4,w//4).permute(2,0,3,1,4)
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:])
        return self.ff(x+self.out(y.transpose(1,2).reshape(b,n,w)))


class DemandReadiness(nn.Module):
    def __init__(self,width=96,policy_context=True):
        super().__init__()
        if width%4:raise ValueError('width must divide four heads')
        self.policy_context=policy_context
        self.embed=nn.Linear(18,width);self.context=nn.Linear(19,width)
        self.blocks=nn.ModuleList([GateBlock(width),GateBlock(width)])
        self.norm=nn.RMSNorm(width);self.demand=nn.Linear(width,972)
        self.readiness=nn.Sequential(nn.Linear(width+7,width),nn.SiLU(),nn.Linear(width,2))
    def forward(self,x,mask,context,policy):
        if not bool(mask.any(1).all()):raise ValueError('empty course')
        h=self.embed(x)+self.context(context)[:,None]
        for block in self.blocks:h=block(h,mask)
        h=self.norm(h);pooled=(h*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
        logits=self.readiness(torch.cat([pooled,policy if self.policy_context else torch.zeros_like(policy)],-1))
        return self.demand(h),logits


def module_loss(pred,logits,target,mask,demand_available,outcomes,episodes,objective='mse'):
    """Binomial readiness likelihood plus masked teacher-demand regression.

    k/n labels retain n; failed/unqualified expert flights have no demand target.
    Gate progress is a separate bounded regression, not a survival likelihood.
    """
    if objective not in ['mse','huber']:raise ValueError('unknown demand objective')
    channel=torch.tensor([.625/7]*7+[0.]*3+[.375/8]*8,device=pred.device).repeat(54)
    errors=(pred.float()-target.float()).square() if objective=='mse' else F.smooth_l1_loss(pred.float(),target.float(),reduction='none',beta=.1)
    active=mask*demand_available[:,None]
    demand=((errors*channel).sum(-1)*active).sum()/(active.sum().clamp_min(1)*54)
    success=(F.binary_cross_entropy_with_logits(logits[:,0].float(),outcomes[:,0],reduction='none')*episodes).sum()/episodes.sum().clamp_min(1)
    progress=F.mse_loss(logits[:,1].float().sigmoid(),outcomes[:,1])
    return demand+success+.25*progress,dict(demand=demand,success_nll=success,gate_progress=progress)
