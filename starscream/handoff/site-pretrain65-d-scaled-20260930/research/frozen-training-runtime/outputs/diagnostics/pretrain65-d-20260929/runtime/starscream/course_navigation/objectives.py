"""Paired local-edit supervision. Not a Riemannian-VAE reproduction."""
import torch
from torch.nn import functional as F
from starscream.course_model.model import rotation6


def edit_courses(gates, position_std=.35, yaw_std=.08):
    """One non-anchor gate, fixed count/start/frame gauge and discrete semantics.

    Inputs are already canonical. Never recanonicalize paired edits, since that
    would spread a local edit over every gate and contaminate the delta target.
    Synthetic edited courses are reconstruction examples, NOT flight-qualified.
    """
    edited=gates.clone(); b,n,_=gates.shape
    if n<2:raise ValueError('local edit requires two gates')
    row=torch.arange(b,device=gates.device)
    index=torch.randint(1,n,(b,),device=gates.device)
    edited[row,index,:3]+=torch.randn(b,3,device=gates.device)*position_std
    angle=torch.randn(b,device=gates.device)*yaw_std
    cs,sn=angle.cos(),angle.sin()
    for a,k in ((3,4),(6,7)):
        x,y=gates[row,index,a],gates[row,index,k]
        edited[row,index,a]=cs*x-sn*y;edited[row,index,k]=sn*x+cs*y
    return edited,index


def edge_loss(raw,target):
    # All successive directed edges, plus cyclic closure. Direction is crucial:
    # unordered center distances can be correct while the traversal is reversed.
    p=raw.float()[...,:3];t=target.float()[...,:3]
    pe=p.roll(-1,1)-p;te=t.roll(-1,1)-t
    direction=(1-F.cosine_similarity(pe,te,dim=-1,eps=1e-5)).mean()
    length=F.smooth_l1_loss(pe.norm(dim=-1)/20,te.norm(dim=-1)/20)
    return direction+length


def navigation_losses(model,gates,context,settings):
    if model.config.decoder_type!='direct':
        raise ValueError('v1 paired objective requires actual direct decode; not a flow endpoint proxy')
    take=min(len(gates),int(settings.get('samples',32)))
    # Rotate subset through stochastic selection, not always the first family.
    ids=torch.randperm(len(gates),device=gates.device)[:take]
    g,c=gates[ids],context[ids]
    e,index=edit_courses(g,settings.get('position_std_m',.35),settings.get('yaw_std_rad',.08))
    base=model.forward_uniform(g,c,sample=False)['raw'].float()
    changed=model.forward_uniform(e,c,sample=False)['raw'].float()
    # Work in metres/rotation6/log-aperture units, not the 20m tokenizer scale.
    pred=changed[...,:11]-base[...,:11];truth=e[...,:11]-g[...,:11]
    error=F.smooth_l1_loss(pred,truth,reduction='none',beta=.1).mean(-1)
    row=torch.arange(take,device=g.device)
    if 'spill_sum_weight' in settings:
        other=error.clone();other[row,index]=0
        local=error[row,index].mean()+float(settings['spill_sum_weight'])*other.sum(-1).mean()
    else:
        local=error[row,index].mean()+.1*error.mean()
    # Preserve absolute identities as well: delta alone admits constant shifts.
    scale=model.gate_scale[:11]
    anchor=F.smooth_l1_loss(changed[...,:11]/scale,e[...,:11]/scale)
    return {'navigation_edit':float(settings.get('edit_weight',1.))*local,
            'navigation_anchor':float(settings.get('anchor_weight',.1))*anchor,
            'navigation_edges':float(settings.get('edge_weight',.1))*edge_loss(base,g)}
