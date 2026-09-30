"""Independent latent summary tokens, not reshaped copies of one pooled code.

Fixed slot count permits a stable flattened chart. Slots summarize the ordered
course with bidirectional attention. This is inspired by latent-set geometry
representations, not an exact reproduction or per-gate latent alignment claim.
"""
import torch
from torch import nn
from .model import CourseVAE
from ..continuous_flow import RMSFlowField,flow_pair,integrate


class StructuredCourseVAE(CourseVAE):
    def __init__(self,c):
        super().__init__(c)
        self.slot_queries=nn.Parameter(torch.randn(1,c.latent_tokens,c.width)*.02)
        self.posterior=nn.Sequential(nn.RMSNorm(c.width),nn.Linear(c.width,2*(c.latent//c.latent_tokens)))
        del self.latent
        self.memory=nn.Linear(c.latent//c.latent_tokens,c.width)
        if c.decoder_type=='flow':
            del self.decoder,self.query,self.output,self.count
            self.field=RMSFlowField(17,c.width,c.heads,c.depth,c.ff,c.max_gates,
                c.latent+1+(19 if c.decoder_context else 0))
            self.register_buffer('decode_noise',torch.randn(1,c.max_gates,17,generator=torch.Generator().manual_seed(90817)))

    def encode(self,gates,mask,context,*,_uniform=False):
        n=gates.shape[1];k=self.config.latent_tokens
        if not 1<=n<=self.config.max_gates:raise ValueError('invalid gate count')
        if not _uniform and (not mask.any(1).all() or not torch.equal(mask,torch.arange(n,device=mask.device)[None]<mask.sum(1)[:,None])):
            raise ValueError('mask must be nonempty contiguous valid prefix')
        cond=self.context(context/self.context_scale)
        g=self.input(gates.masked_fill(~mask[...,None],0)/self.gate_scale)+self.position[:,1:n+1]
        slots=self.slot_queries.expand(len(g),-1,-1)+cond[:,None]
        x=torch.cat([slots,g],1)
        valid=torch.cat([torch.ones(len(g),k,device=g.device,dtype=torch.bool),mask],1)
        for block in self.encoder:x=block(x,condition=cond,attention_bias=None if _uniform else valid[:,None,None,:],is_causal=False)
        mu,lv=self.posterior(x[:,:k]).float().chunk(2,-1)
        return mu.flatten(1),lv.flatten(1).clamp(-12,8)

    def memory_tokens(self,z):return self.memory(z.reshape(len(z),self.config.latent_tokens,-1))

    def condition(self,z,counts,context):
        values=[z,counts[:,None].to(z)/self.config.max_gates]
        if self.config.decoder_context:values.append(context/self.context_scale)
        return torch.cat(values,-1)

    def decode(self,z,counts,context,*,_length=None,noise=None,steps=None):
        n=int(counts.max()) if _length is None else _length
        if not 1<=n<=self.config.max_gates:raise ValueError('invalid count')
        # Do not silently attend to padded noisy output tokens in mixed buckets.
        if _length is None and not bool((counts==n).all()):
            rows=[]
            for i,c in enumerate(counts.tolist()):
                row,_=self.decode(z[i:i+1],counts[i:i+1],context[i:i+1],_length=c,
                    noise=None if noise is None else noise[i:i+1,:c],steps=steps)
                rows.append(torch.nn.functional.pad(row,(0,0,0,n-c)))
            return torch.cat(rows),torch.arange(n,device=z.device)[None]<counts[:,None]
        mem=self.memory_tokens(z);mask=torch.arange(n,device=z.device)[None]<counts[:,None]
        if self.config.decoder_type=='flow':
            cond=self.condition(z,counts,context)
            source=self.decode_noise[:,:n].expand(len(z),-1,-1) if noise is None else noise
            raw=integrate(lambda x,t:self.field(x,t,cond,memory=mem),source,
                steps or self.config.flow_steps,self.config.flow_method)*self.gate_scale
        else:
            cond=mem.mean(1)+self.count(counts)
            if self.config.decoder_context:cond=cond+self.context(context/self.context_scale)
            x=torch.cat([mem,self.query[:,:n].expand(len(z),-1,-1)+cond[:,None]],1)
            for block in self.decoder:x=block(x,condition=cond,is_causal=False)
            raw=self.output(x[:,-n:])*self.gate_scale
        return raw.masked_fill(~mask[...,None],0),mask

    def training_forward(self,inputs,target,context):
        if self.config.decoder_type=='direct':return self.forward_uniform(inputs,context,sample=True)
        mask=torch.ones(inputs.shape[:2],device=inputs.device,dtype=torch.bool)
        mu,lv=self.encode(inputs,mask,context,_uniform=True)
        z=mu+torch.randn_like(mu)*(.5*lv).exp() if self.config.sample_posterior else mu
        n=inputs.shape[1];counts=torch.full((len(z),),n,device=z.device,dtype=torch.long)
        values=target.float().clone();values[...,15:17]=torch.logit(values[...,15:17].clamp(.05,.95))
        xt,t,v=flow_pair(values/self.gate_scale)
        pred=self.field(xt,t,self.condition(z,counts,context),memory=self.memory_tokens(z)).float()
        return dict(raw=(xt+(1-t[:,None,None])*pred)*self.gate_scale,mu=mu,logvar=lv,z=z,
            flow_matching=(pred-v).square().mean())
