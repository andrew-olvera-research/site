"""Decoder-space local proposals, with explicit fixed-count trust regions.

Not a policy-transfer compass. Cross-count target geometry is compared as an
ordered arc-length polyline; no gate birth/removal or full target equivalence.
"""
import torch


def ordered_curve(points,samples=64):
    """Closed ordered center polyline at normalized arc length, differentiable
    within segment-index regions. It is not a time-optimal flown trajectory.
    """
    end=torch.roll(points,-1,1);length=(end-points).norm(dim=-1).clamp_min(1e-6)
    cumulative=torch.cat([torch.zeros_like(length[:,:1]),length.cumsum(1)],1)
    t=torch.arange(samples,device=points.device,dtype=points.dtype)[None]/samples*cumulative[:,-1:]
    index=(torch.searchsorted(cumulative.contiguous(),t.contiguous(),right=True)-1).clamp(0,points.shape[1]-1)
    start=points.gather(1,index[...,None].expand(-1,-1,3));finish=end.gather(1,index[...,None].expand(-1,-1,3))
    fraction=(t-cumulative.gather(1,index))/length.gather(1,index)
    return start+fraction[...,None]*(finish-start)


def curve_distance(a,b):
    return (ordered_curve(a)-ordered_curve(b)).square().sum(-1).mean(-1).sqrt()


def project_ball(z,center,radius):
    d=z-center
    return center+d*(radius/d.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp_max(1)
