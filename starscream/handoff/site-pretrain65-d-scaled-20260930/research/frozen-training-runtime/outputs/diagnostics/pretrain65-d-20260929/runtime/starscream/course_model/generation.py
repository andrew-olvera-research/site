"""Reproducible shape-first proposal mixture; no learned policy required."""
from dataclasses import replace
import numpy as np
from scipy.interpolate import splprep,splev
from starscream.env.tracks import Track,Gate,forward_up_quaternion,matrix_quaternion
from starscream.env.racing_manifold.generator import SplineManifoldGenerator
from starscream.env.racing_manifold.primitives import graft_primitive

FAMILIES=('oval','lobed','figure_eight','stadium','switchback','free_spline')
PRIMITIVES=('none','slalom','split_s','corkscrew','inverted_gate')


def bounds_for(gates):
    p=np.array([g.position for g in gates]); b=np.stack([p.min(0)-5,p.max(0)+5],1)
    b[2,0]=0
    return b.astype(np.float32)


def shape_parent(family,n,rng):
    t=np.linspace(0,2*np.pi,257)
    aspect=rng.uniform(.3,1.5)
    if family=='oval':x,y=np.cos(t),aspect*np.sin(t)
    elif family=='lobed':
        r=1+rng.uniform(.15,.4)*np.cos(int(rng.integers(2,5))*t+rng.uniform(-np.pi,np.pi))
        x,y=r*np.cos(t),aspect*r*np.sin(t)
    elif family=='figure_eight':x,y=np.sin(t),aspect*np.sin(2*t)
    elif family=='stadium':
        x=np.cos(t)+rng.uniform(.5,2)*np.tanh(4*np.cos(t));y=aspect*np.sin(t)
    elif family=='switchback':
        # Ordered sweeps, with an exterior return leg; changes whole-course
        # traversal rather than decorating an oval with local gate primitives.
        lanes=int(rng.integers(2,5))
        anchors=[]
        for lane in range(lanes):
            for side in ((-1,1) if lane%2==0 else (1,-1)):
                anchors.append([side,aspect*(lane/(lanes-1)*2-1)])
        anchors.extend([[1.5,aspect*1.4],[-1.5,aspect*1.4],[-1.5,-aspect*1.3],anchors[0]])
        points=np.array(anchors)
        spline,_=splprep(points.T.copy(),per=True,s=0,k=2)
        x,y=splev(np.linspace(0,1,len(t)),spline)
    else:
        knots=int(rng.integers(5,11));angle=np.linspace(0,2*np.pi,knots,endpoint=False)
        radius=rng.uniform(.55,1.5,knots)
        points=np.c_[radius*np.cos(angle),aspect*radius*np.sin(angle)]
        # Concave hub visits and crossed order are separate modes. Pure radial
        # ordering otherwise excluded Swift-like retraversal from the corpus.
        mode=int(rng.integers(3))
        if mode==1:points[1::3]*=rng.uniform(.08,.35)
        elif mode==2:
            ids=np.arange(knots);ids[1:4]=ids[1:4][::-1];points=points[ids]
        points=np.r_[points,points[:1]]
        spline,_=splprep(points.T.copy(),per=True,s=0,k=3)
        x,y=splev(np.linspace(0,1,len(t)),spline)
    z=rng.uniform(.03,.25)*np.sin(t+rng.uniform(0,6))+rng.uniform(0,.1)*np.sin(2*t)
    curve=np.c_[x,y,z]
    distances=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
    # Nonuniform gate placement adds straight/turn clustering without conflating
    # gate count and global shape. n and requested length are separately sampled.
    increments=rng.uniform(.65,1.35,n);phase=np.r_[0,np.cumsum(increments[:-1])]/increments.sum()
    points=np.stack([np.interp(phase*distances[-1],distances,curve[:,i]) for i in range(3)],1)
    length=float(np.exp(rng.uniform(np.log(max(25,n*2)),np.log(240))))
    points*=length/np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1).sum()
    # Vertical range is an independent physical coordinate, not tied to length.
    points[:,2]=(points[:,2]-points[:,2].min())/max(np.ptp(points[:,2]),1e-6)*rng.uniform(.2,9)+rng.uniform(1.4,3)
    tangents=SplineManifoldGenerator._spline_tangents(points)
    size=rng.uniform(1.1,2.8,2)
    gates=tuple(Gate(p,forward_up_quaternion(v),size.copy(),f'g{i}') for i,(p,v) in enumerate(zip(points,tangents)))
    return Track(family,gates,bounds_for(gates),bool(rng.random()>.15),{'source':'independent-shape'})


def perturb(parent,rng,strength):
    points=np.array([g.position for g in parent.gates],np.float64)
    center=points.mean(0)
    scale=np.exp(rng.normal(0,strength*.35,3));scale[2]=np.exp(rng.normal(0,strength*.15))
    shifted=(points-center)*scale+center
    noise=rng.normal(0,strength,len(points)*3).reshape(-1,3)
    noise=(np.roll(noise,1,0)+2*noise+np.roll(noise,-1,0))/4
    shifted+=noise
    old=SplineManifoldGenerator._spline_tangents(points)
    new=SplineManifoldGenerator._spline_tangents(shifted)
    gates=[]
    for g,p,a,b in zip(parent.gates,shifted,old,new):
        rot=SplineManifoldGenerator._align_vectors(a,b)@g.rotation
        gates.append(replace(g,position=p,quaternion_wxyz=matrix_quaternion(rot),size=g.size*np.exp(rng.normal(0,strength*.15,2))))
    return replace(parent,gates=tuple(gates),bounds=bounds_for(gates))


def proposal(index,seed,references,attempt=0):
    # Five siblings share a parent and split. Never scatter siblings across val.
    parent_id=index//5
    rng_parent=np.random.default_rng(np.random.SeedSequence([seed,parent_id]))
    rng=np.random.default_rng(np.random.SeedSequence([seed,index,attempt,117]))
    reference=parent_id%5<2  # 40% reference-neighborhood, 60% independent.
    origin=parent_id%len(references) if reference else -1
    family=6+origin if reference else (parent_id//5*3+(parent_id%5-2))%6
    if reference:parent=references[origin]
    else:
        ranges=((4,7),(7,11),(11,17),(17,25));low,high=ranges[(parent_id//10)%4]
        n=int(rng_parent.integers(low,high))
        parent=shape_parent(FAMILIES[family],n,rng_parent)
    strength=(.025,.15,.5,1.,2.)[index%5]
    # Resample a parent after repeated structural rejection; record attempts.
    if attempt>=20 and not reference:
        parent=shape_parent(FAMILIES[family],len(parent.gates),rng)
    track=perturb(parent,rng,strength)
    primitive=0
    if not reference and rng.random()<.35:
        primitive=int(rng.integers(1,len(PRIMITIVES)))
        track=graft_primitive(track,PRIMITIVES[primitive],anchor=int(rng.integers(len(track.gates))),severity=float(rng.uniform(.25,1)))
    # Real CDRA apertures touch the floor. Keep sampled apertures above it by
    # an explicit recorded vertical repair, rather than rejecting that family.
    bottom=min(g.position[2]-abs(g.rotation[2,1])*g.size[0]/2-abs(g.rotation[2,2])*g.size[1]/2 for g in track.gates)
    lift=max(0.,.02-float(bottom))
    if lift:
        gates=tuple(replace(g,position=g.position+np.array([0,0,lift])) for g in track.gates)
        track=replace(track,gates=gates,bounds=bounds_for(gates))
    # Gate shape is not drone roll: 180-degree rectangular aperture roll alone
    # is physically symmetric. Provenance labels do not imply a new skill.
    return track,dict(parent_id=parent_id,family=family,reference=origin,strength=strength,
        primitive=primitive,attempt=attempt,ground_lift_m=lift,split=int(np.random.default_rng(np.random.SeedSequence([seed,parent_id,901])).integers(10)==0))
