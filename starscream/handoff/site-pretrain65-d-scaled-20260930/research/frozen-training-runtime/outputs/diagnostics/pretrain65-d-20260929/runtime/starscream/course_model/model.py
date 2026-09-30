"""Count-conditioned bidirectional Transformer VAE using canonical policy blocks."""
from dataclasses import dataclass,asdict
import torch
from torch import nn
from torch.nn import functional as F
from starscream.sequence_architecture import AdaRMSNormSwiGLUBlock
from .schema import GATE_DIM,CONTEXT_DIM,VERSION


@dataclass(frozen=True)
class VAEConfig:
    width:int=256
    heads:int=8
    depth:int=3
    ff:int=768
    latent:int=32
    max_gates:int=32
    dropout:float=0.
    decoder_context:bool=True
    decoder_type:str='direct'
    flow_steps:int=16
    flow_method:str='heun'
    flow_prior:bool=False
    flow_condition_tokens:int=0
    latent_tokens:int=1
    sample_posterior:bool=True

    def __post_init__(self):
        if self.decoder_type not in ('direct','flow') or self.flow_steps<1 or self.flow_method not in ('euler','heun'):
            raise ValueError('invalid course decoder contract')
        if self.flow_prior and self.decoder_type!='flow':raise ValueError('flow prior requires flow decoder')
        if self.latent_tokens<1 or self.latent%self.latent_tokens:raise ValueError('latent must divide into whole tokens')
        if self.latent_tokens>1 and self.flow_prior:raise ValueError('structured learned prior not implemented')


class CourseVAE(nn.Module):
    def __init__(self,config=VAEConfig()):
        super().__init__();self.config=config;c=config
        self.input=nn.Linear(GATE_DIM,c.width)
        self.context=nn.Linear(CONTEXT_DIM,c.width)
        self.position=nn.Parameter(torch.randn(1,c.max_gates+1,c.width)*.02)
        self.query=nn.Parameter(torch.randn(1,c.max_gates,c.width)*.02)
        self.count=nn.Embedding(c.max_gates+1,c.width)
        self.encoder=nn.ModuleList([AdaRMSNormSwiGLUBlock(c.width,c.heads,c.ff,dropout=c.dropout,adaln_zero=False) for _ in range(c.depth)])
        self.decoder=nn.ModuleList([AdaRMSNormSwiGLUBlock(c.width,c.heads,c.ff,dropout=c.dropout,adaln_zero=True) for _ in range(c.depth)])
        self.posterior=nn.Sequential(nn.RMSNorm(c.width),nn.Linear(c.width,2*c.latent))
        self.latent=nn.Linear(c.latent,c.width)
        self.output=nn.Sequential(nn.RMSNorm(c.width),nn.Linear(c.width,GATE_DIM))
        # Fixed physical normalization retains differences in course scale.
        scale=torch.ones(GATE_DIM);scale[:3]=20
        cs=torch.ones(CONTEXT_DIM);cs[:3]=20;cs[9:12]=10;cs[13:]=30
        self.register_buffer('gate_scale',scale);self.register_buffer('context_scale',cs)

    def encode(self,gates,mask,context,*,_uniform=False):
        if gates.shape[1]>self.config.max_gates or (not _uniform and not mask.any(1).all()):raise ValueError('invalid gate count')
        if not _uniform and not torch.equal(mask,torch.arange(mask.shape[1],device=mask.device)[None]<mask.sum(1)[:,None]):
            raise ValueError('mask must be a contiguous valid prefix')
        # Replace pads before projection; even NaN pads must not contaminate keys.
        g=gates.masked_fill(~mask[...,None],0)/self.gate_scale
        cond=self.context(context/self.context_scale)
        x=torch.cat([cond[:,None],self.input(g)],1)+self.position[:,:g.shape[1]+1]
        valid=torch.cat([torch.ones_like(mask[:,:1]),mask],1)
        bias=None if _uniform else valid[:,None,None,:]  # Uniform count batches permit FlashAttention.
        for block in self.encoder:x=block(x,condition=cond,attention_bias=bias,is_causal=False)
        mu,logvar=self.posterior(x[:,0]).float().chunk(2,-1)
        return mu,logvar.clamp(-12,8)

    def decode(self,z,counts,context,*,_length=None):
        if _length is None and ((counts<1)|(counts>self.config.max_gates)).any():raise ValueError('invalid decoder count')
        length=int(counts.max()) if _length is None else _length
        mask=torch.arange(length,device=z.device)[None]<counts[:,None]
        cond=self.latent(z)+self.count(counts)
        if self.config.decoder_context:
            cond=cond+self.context(context/self.context_scale)
        x=self.query[:,:length].expand(len(z),-1,-1)+cond[:,None]
        for block in self.decoder:x=block(x,condition=cond,attention_bias=None if _length is not None else mask[:,None,None,:],is_causal=False)
        raw=self.output(x)*self.gate_scale
        return raw.masked_fill(~mask[...,None],0),mask

    def forward(self,gates,mask,context,sample=True):
        mu,logvar=self.encode(gates,mask,context)
        z=mu+torch.randn_like(mu)*torch.exp(.5*logvar) if sample and self.config.sample_posterior else mu
        raw,_=self.decode(z,mask.sum(1),context)
        return {'raw':raw,'mu':mu,'logvar':logvar,'z':z}

    def forward_uniform(self,gates,context,sample=True):
        """Exact same model for homogeneous-count batches, without padding/sync.

        Takes no external mask: every provided gate is real. The count-bucket
        loader guarantees this contract. Mixed-count callers use forward().
        """
        b,n,_=gates.shape
        if not 1<=n<=self.config.max_gates:raise ValueError('invalid uniform count')
        mask=torch.ones(b,n,dtype=torch.bool,device=gates.device)
        counts=torch.full((b,),n,dtype=torch.long,device=gates.device)
        mu,logvar=self.encode(gates,mask,context,_uniform=True)
        z=mu+torch.randn_like(mu)*torch.exp(.5*logvar) if sample and self.config.sample_posterior else mu
        raw,_=self.decode(z,counts,context,_length=n)
        return {'raw':raw,'mu':mu,'logvar':logvar,'z':z}

    def checkpoint(self):return {'schema':VERSION,'config':asdict(self.config),'model':self.state_dict()}

    @torch.no_grad()
    def generate_tracks(self,z,counts,context):
        """Decode proposals; reject invalid geometry instead of silent repairs.

        Geometric acceptance here is NOT flight feasibility or RL admission.
        """
        from .schema import unpack,static_reasons
        raw,_=self.decode(z,counts,context)
        proposals=[]
        for i,n in enumerate(counts.tolist()):
            values=raw[i,:n].float().cpu().numpy().copy()
            values[:,15:17]=raw[i,:n,15:17].sigmoid().float().cpu().numpy()
            try:
                if (abs(values[:,9:11])>5).any():raise ValueError('aperture outside proposal range')
                track=unpack(values,context[i].float().cpu().numpy(),name=f'vae_{i}')
                reasons=static_reasons(track)
            except ValueError as error:track=None;reasons=[str(error)]
            proposals.append({'track':track,'geometry_valid':not reasons,'reasons':reasons,'flight_feasibility':'unknown'})
        return proposals


def rotation6(raw):
    a=F.normalize(raw[...,:3],dim=-1,eps=1e-6)
    b=F.normalize(raw[...,3:]-a*(a*raw[...,3:]).sum(-1,keepdim=True),dim=-1,eps=1e-6)
    return torch.stack([a,b,torch.cross(a,b,dim=-1)],-1)


def vae_loss(out,target,mask,beta=.001,weights=None):
    # Reductions, rotations and Gaussian KL remain FP32 under BF16 AMP.
    raw=out['raw'].float();target=target.float()[:,:raw.shape[1]].masked_fill(~mask[:,:raw.shape[1],None],0)
    mask=mask[:,:raw.shape[1]]
    def mean(x,m=mask):return ((x*m).sum(1)/m.sum(1).clamp_min(1)).mean()
    pos=mean(((raw[...,:3]-target[...,:3])/20).square().mean(-1))
    rotation=mean((rotation6(raw[...,3:9])-rotation6(target[...,3:9])).square().mean((-1,-2)))
    # Discourage invalid 6D vectors in addition to the proper-rotation loss.
    a,b=raw[...,3:6],raw[...,6:9]
    ortho=mean((a.norm(dim=-1)-1).square()+(b.norm(dim=-1)-1).square()+(a*b).sum(-1).square())
    size=mean((raw[...,9:11]-target[...,9:11]).square().mean(-1))
    kind=mean(F.cross_entropy(raw[...,11:15].transpose(1,2),target[...,11:15].argmax(-1),reduction='none'))
    flags=mean(F.binary_cross_entropy_with_logits(raw[...,15:17],target[...,15:17],reduction='none').mean(-1))
    pair=mask[:,1:]&mask[:,:-1]
    relative=mean(((torch.diff(raw[...,:3],dim=1)-torch.diff(target[...,:3],dim=1))/20).square().mean(-1),pair)
    logvar=out['logvar'].float();mu=out['mu'].float()
    kl=(-.5*(1+logvar-mu.square()-logvar.exp()).sum(-1)).mean()
    parts=dict(position=pos,rotation=rotation,orthogonality=ortho,aperture=size,kind=kind,flags=flags,relative=relative,kl=kl)
    w=dict(position=1.,rotation=1.,orthogonality=.05,aperture=1.,kind=1.,flags=1.,relative=.5)
    if weights is not None:
        if set(weights)-set(w):raise ValueError('unknown reconstruction objective')
        w.update(weights)
    if any(v<0 for v in w.values()) or beta<0:raise ValueError('negative loss weight')
    parts['loss']=sum(w[k]*parts[k] for k in w)+beta*kl
    return parts


class FeasibilityHead(nn.Module):
    """Optional contract-conditioned surrogate; untrained until MPCC labels exist."""
    def __init__(self,latent=32,contract_dim=8):
        super().__init__();self.net=nn.Sequential(nn.Linear(latent+CONTEXT_DIM+1+contract_dim,128),nn.SiLU(),nn.Linear(128,1))
    def forward(self,z,context,counts,contract):
        return self.net(torch.cat([z,context,counts[:,None].to(z)/32,contract],-1)).squeeze(-1)


def feasibility_loss(logits,labels):
    """Unknown (-1) controller outcomes do not become negative examples."""
    known=labels>=0
    return F.binary_cross_entropy_with_logits(logits[known],labels[known].to(logits)) if known.any() else logits.sum()*0
