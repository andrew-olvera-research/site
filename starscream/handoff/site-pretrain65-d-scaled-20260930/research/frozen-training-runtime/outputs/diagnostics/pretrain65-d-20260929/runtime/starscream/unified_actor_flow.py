"""v5.5 token contract, one shared modern DiT, single CTBR conditional flow.

Reuses DAgger's flow API but explicitly prohibits shortcut bootstrapping.
History is clean conditioning, never a denoising target. Current route tokens
follow history; the noisy action is last and attends to the whole prefix.
"""
import torch
from torch import nn
from .continuous_flow import time_features,flow_pair,integrate
from .sequence_architecture import AdaRMSNormSwiGLUBlock


class UnifiedActorFlow(nn.Module):
    def __init__(self,width,heads,depth,ff,source_noise,deterministic_source='fixed_gaussian'):
        super().__init__();self.source_noise=source_noise;self.action_horizon=1;self.action_dim=4
        if deterministic_source not in ('fixed_gaussian','zero'):raise ValueError('invalid deterministic flow source')
        self.deterministic_source=deterministic_source
        self.action=nn.Linear(4,width)
        self.time=nn.Sequential(nn.Linear(64,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList([AdaRMSNormSwiGLUBlock(width,heads,ff,dropout=0.,adaln_zero=True) for _ in range(depth)])
        self.norm=nn.RMSNorm(width);self.output=nn.Linear(width,4)
        nn.init.normal_(self.output.weight,std=.001);nn.init.zeros_(self.output.bias)
        self.register_buffer('fixed_source',torch.randn(1,1,4,generator=torch.Generator().manual_seed(55508))*source_noise)

    def hidden(self,context,noisy=None,t=None):
        if t is None:t=context.new_zeros(len(context))
        condition=context.mean(1)+self.time(time_features(t))
        h=context if noisy is None else torch.cat([context,self.action(noisy)],1)
        for block in self.blocks:h=block(h,condition=condition,is_causal=True)
        return self.norm(h)

    def velocity_from_context(self,noisy,t,step_size,context):
        return self.output(self.hidden(context,noisy,t)[:,-1:])

    def shortcut_training_loss(self,target,observations,actions,*,context,direct_weight=1.,bootstrap_weight=0.,reduction='none',**unused):
        if bootstrap_weight!=0 or direct_weight!=1:raise ValueError('unified actor uses pure CFM, no shortcuts')
        xt,t,v=flow_pair(target,torch.randn_like(target)*self.source_noise)
        prediction=self.velocity_from_context(xt,t,torch.zeros_like(t),context).float()
        loss=(prediction-v).square().mean((1,2))
        endpoint=xt+(1-t[:,None,None])*prediction
        metrics=dict(flow_matching_loss=loss.mean().detach(),shortcut_bootstrap_loss=loss.new_zeros(()),
                     endpoint_mse=(endpoint-target).square().mean().detach(),step_size=loss.new_zeros(()))
        return (loss.mean() if reduction=='mean' else loss),metrics

    def integrate(self,observations,actions,*,steps,method,deterministic,context,source_noise=None,**unused):
        scale=self.source_noise if source_noise is None else source_noise
        source=self.fixed_source.expand(len(context),-1,-1) if deterministic else torch.randn(len(context),1,4,device=context.device)*scale
        if deterministic and self.deterministic_source=='zero':source=torch.zeros_like(source)
        return integrate(lambda x,t:self.velocity_from_context(x,t,torch.zeros_like(t),context),source,steps,method).clamp(-1,1)

    @torch.no_grad()
    def sample(self,*args,**kwargs):return self.integrate(*args,**kwargs)
