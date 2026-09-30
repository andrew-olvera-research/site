"""Independent, count/shape-balanced codec corpus; no target-ray construction."""
from dataclasses import replace
import numpy as np
from .generation import FAMILIES,PRIMITIVES,shape_parent,perturb,bounds_for
from starscream.env.racing_manifold.primitives import graft_primitive
from starscream.env.tracks import forward_up_quaternion
from starscream.env.racing_manifold.generator import SplineManifoldGenerator


def proposal(index,seed,references,attempt=0):
    parent_id=index//5
    # Structural degeneracies (e.g. a flat self-intersection sampled twice)
    # need a fresh parent, not endless millimeter perturbations of that parent.
    prng=np.random.default_rng(np.random.SeedSequence([seed,parent_id,attempt//20]))
    rng=np.random.default_rng(np.random.SeedSequence([seed,index,attempt,711]))
    family=parent_id%6
    n=4+(parent_id//6)%21
    parent=shape_parent(FAMILIES[family],n,prng)
    flat=bool(prng.random()<.3)
    if flat:
        points=np.array([g.position for g in parent.gates]);points[:,2]=prng.uniform(1.5,4.)
        tangents=SplineManifoldGenerator._spline_tangents(points)
        gates=tuple(replace(g,position=p,quaternion_wxyz=forward_up_quaternion(t)) for g,p,t in zip(parent.gates,points,tangents))
        parent=replace(parent,gates=gates,bounds=bounds_for(gates))
    strength=(.025,.15,.5,1.,2.)[index%5]
    track=perturb(parent,rng,strength)
    primitive=0
    if not flat and rng.random()<.3:
        primitive=int(rng.integers(1,len(PRIMITIVES)))
        track=graft_primitive(track,PRIMITIVES[primitive],anchor=int(rng.integers(len(track.gates))),severity=float(rng.uniform(.25,.8)))
    if flat:
        z=float(parent.gates[0].position[2])
        gates=tuple(replace(g,position=np.r_[g.position[:2],z]) for g in track.gates)
        track=replace(track,gates=gates,bounds=bounds_for(gates))
    bottom=min(g.position[2]-abs(g.rotation[2,1])*g.size[0]/2-abs(g.rotation[2,2])*g.size[1]/2 for g in track.gates)
    lift=max(0.,.05-float(bottom))
    if lift:
        gates=tuple(replace(g,position=g.position+np.array([0,0,lift])) for g in track.gates)
        track=replace(track,gates=gates,bounds=bounds_for(gates))
    return track,dict(parent_id=parent_id,family=family,reference=-1,strength=strength,
        primitive=primitive,attempt=attempt,ground_lift_m=lift,
        split=int(np.random.default_rng(np.random.SeedSequence([seed,parent_id,901])).integers(10)==0))
