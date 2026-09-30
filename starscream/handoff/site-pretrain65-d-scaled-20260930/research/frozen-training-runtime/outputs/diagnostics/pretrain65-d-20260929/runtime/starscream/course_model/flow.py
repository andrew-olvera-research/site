"""Variational course encoder with flow decoder and optional learned flow prior.

CFM plus KL/reconstruction regularizers is a surrogate, not an exact ELBO.
The learned prior is count-conditioned and fits detached aggregated posterior
samples. Decoder noise is explicit: fixed common noise defines reproducible
chart probes, random noise measures conditional diversity.
"""
import torch
from torch import nn
from .model import CourseVAE
from ..continuous_flow import RMSFlowField,flow_pair,integrate


class FlowCourseVAE(CourseVAE):
    def __init__(self,config):
        super().__init__(config);c=config
        # Remove unused deterministic decoder; encoder is exactly shared.
        del self.decoder,self.output,self.query,self.latent,self.count
        context_dim=19 if c.decoder_context else 0
        self.field=RMSFlowField(17,c.width,c.heads,c.depth,c.ff,c.max_gates,
                               c.latent+1+context_dim,c.flow_condition_tokens)
        self.prior=RMSFlowField(c.latent,c.width,c.heads,2,c.ff,1,1) if c.flow_prior else None
        rng=torch.Generator().manual_seed(90817)
        self.register_buffer('decode_noise',torch.randn(1,c.max_gates,17,generator=rng))

    def condition(self,z,counts,context):
        values=[z,counts[:,None].to(z)/self.config.max_gates]
        if self.config.decoder_context:values.append(context/self.context_scale)
        return torch.cat(values,-1)

    def target_values(self,gates):
        x=gates.float().clone()
        x[...,15:17]=torch.logit(x[...,15:17].clamp(.05,.95))
        return x/self.gate_scale

    def training_forward(self,inputs,target,context):
        mask=torch.ones(inputs.shape[:2],device=inputs.device,dtype=torch.bool)
        mu,lv=self.encode(inputs,mask,context,_uniform=True)
        z=mu+torch.randn_like(mu)*(.5*lv).exp() if self.config.sample_posterior else mu
        n=inputs.shape[1];counts=torch.full((len(z),),n,device=z.device,dtype=torch.long)
        x1=self.target_values(target);xt,t,v=flow_pair(x1)
        predicted=self.field(xt,t,self.condition(z,counts,context)).float()
        loss=(predicted-v).square().mean()
        # A supervised local endpoint estimate, NOT a sampled deployment endpoint.
        raw=(xt+(1-t[:,None,None])*predicted)*self.gate_scale
        result=dict(raw=raw,mu=mu,logvar=lv,z=z,flow_matching=loss)
        if self.prior is not None:
            p,pt,pv=flow_pair(z.detach()[:,None])
            result['prior_flow_matching']=(self.prior(p,pt,counts[:,None].float()/self.config.max_gates).float()-pv).square().mean()
        return result

    def decode(self,z,counts,context,*,_length=None,noise=None,steps=None):
        n=int(counts.max()) if _length is None else _length
        if n<1 or n>self.config.max_gates:raise ValueError('invalid decode count')
        # Homogeneous buckets in training/eval; mixed count handled independently.
        if _length is None and not bool((counts==n).all()):
            rows=[]
            for i,count in enumerate(counts.tolist()):
                row,_=self.decode(z[i:i+1],counts[i:i+1],context[i:i+1],_length=count,steps=steps)
                rows.append(torch.nn.functional.pad(row,(0,0,0,n-count)))
            return torch.cat(rows),torch.arange(n,device=z.device)[None]<counts[:,None]
        cond=self.condition(z,counts,context)
        source=self.decode_noise[:,:n].expand(len(z),-1,-1) if noise is None else noise
        raw=integrate(lambda x,t:self.field(x,t,cond),source,
                      steps or self.config.flow_steps,self.config.flow_method)*self.gate_scale
        return raw,torch.arange(n,device=z.device)[None]<counts[:,None]

    def sample_prior(self,noise,counts,context):
        if self.prior is None:return noise
        return integrate(lambda x,t:self.prior(x,t,counts[:,None].float()/self.config.max_gates),
                         noise[:,None],self.config.flow_steps,self.config.flow_method)[:,0]

    def forward_uniform(self,gates,context,sample=True):
        mask=torch.ones(gates.shape[:2],device=gates.device,dtype=torch.bool)
        mu,lv=self.encode(gates,mask,context,_uniform=True)
        z=mu+torch.randn_like(mu)*(.5*lv).exp() if sample and self.config.sample_posterior else mu
        counts=torch.full((len(z),),gates.shape[1],device=z.device,dtype=torch.long)
        raw,_=self.decode(z,counts,context,_length=gates.shape[1])
        return dict(raw=raw,mu=mu,logvar=lv,z=z)


def build_course_model(config):
    if config.latent_tokens>1:
        from .structured import StructuredCourseVAE
        return StructuredCourseVAE(config)
    return FlowCourseVAE(config) if config.decoder_type=='flow' else CourseVAE(config)
