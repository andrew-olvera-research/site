"""Straight-path conditional flow matching with canonical RMSNorm/SwiGLU blocks.

No shortcut targets, diffusion scheduler, or hidden one-step distillation.
Time runs from Gaussian source (0) to data (1). All integration is FP32.
"""
import math
import torch
from torch import nn
from .sequence_architecture import AdaRMSNormSwiGLUBlock


def time_features(t, width=64):
    frequencies=torch.exp(-math.log(10000)*torch.arange(width//2,device=t.device)/(width//2-1))
    phase=t.float()[:,None]*frequencies[None]*2*math.pi
    return torch.cat([phase.sin(),phase.cos()],-1)


def flow_pair(target, noise=None, time=None):
    target=target.float()
    noise=torch.randn_like(target) if noise is None else noise.float()
    time=torch.rand(len(target),device=target.device) if time is None else time.float()
    shape=(len(target),)+(1,)*(target.ndim-1)
    return noise+(target-noise)*time.reshape(shape),time,target-noise


def integrate(field, source, steps, method='heun'):
    if steps<1 or method not in ('euler','heun'):raise ValueError('invalid flow integration')
    x=source.float();dt=1./steps
    for i in range(steps):
        t=x.new_full((len(x),),i*dt);v=field(x,t).float()
        if method=='heun':
            v=(v+field(x+dt*v,t+dt).float())*.5
        x=x+dt*v
    return x


class RMSFlowField(nn.Module):
    """Joint attention over optional condition tokens and noisy output tokens."""
    def __init__(self, dim, width, heads, depth, ff, max_length, condition_dim,
                 condition_tokens=0, causal=False):
        super().__init__();self.causal=causal;self.condition_tokens=condition_tokens
        self.input=nn.Linear(dim,width);self.position=nn.Parameter(torch.randn(1,max_length,width)*.02)
        self.time=nn.Sequential(nn.Linear(64,width),nn.SiLU(),nn.Linear(width,width))
        self.condition=nn.Linear(condition_dim,width)
        self.memory=nn.Linear(condition_dim,condition_tokens*width) if condition_tokens else None
        self.blocks=nn.ModuleList([AdaRMSNormSwiGLUBlock(width,heads,ff,dropout=0.,adaln_zero=True) for _ in range(depth)])
        self.norm=nn.RMSNorm(width);self.output=nn.Linear(width,dim)
        # Nonzero small output lets AdaLN gates learn on the first update.
        nn.init.normal_(self.output.weight,std=.001);nn.init.zeros_(self.output.bias)

    def forward(self,x,t,condition,memory=None):
        n=x.shape[1];cond=self.condition(condition)+self.time(time_features(t))
        h=self.input(x)+self.position[:,:n]
        if memory is None and self.memory is not None:
            memory=self.memory(condition).reshape(len(x),self.condition_tokens,-1)
        if memory is not None:h=torch.cat([memory,h],1)
        for block in self.blocks:h=block(h,condition=cond,is_causal=self.causal)
        return self.output(self.norm(h[:,-n:]))
