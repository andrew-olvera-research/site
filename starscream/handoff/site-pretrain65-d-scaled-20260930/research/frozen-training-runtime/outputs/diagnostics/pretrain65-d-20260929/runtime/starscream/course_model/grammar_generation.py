"""Design-first ordered racing geometry. Splines are one of eight families.

No benchmark coordinates enter proposals. Static qualification is not MPCC
qualification. Explicit primitives retain their physical relationships.
"""
import numpy as np
from starscream.env.tracks import Gate,Track,forward_up_quaternion,matrix_quaternion
from .generation import bounds_for,shape_parent

FAMILIES=('banking_polygon','concave_infield','chicane_lanes','hub_returns',
          'altitude_layers','stacked_reversal','mixed_technical','spline_free')


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def proposal(index,seed,attempt=0):
    parent=index//3;family=parent%len(FAMILIES);n=4+(parent//len(FAMILIES))%29
    rng=np.random.default_rng(np.random.SeedSequence([seed,parent,attempt//12]))
    aug=np.random.default_rng(np.random.SeedSequence([seed,index,attempt,817]))
    jitter=min(.18,.3*2*np.pi/n)
    phase=np.linspace(0,2*np.pi,n,endpoint=False)+rng.uniform(-jitter,jitter,n)
    radius=rng.uniform(7,24);aspect=rng.uniform(.45,1.4)
    p=np.c_[radius*np.cos(phase),radius*aspect*np.sin(phase),np.zeros(n)]
    motif=np.zeros(n,np.int8)
    if family==0:
        p[:,:2]*=rng.uniform(.88,1.12,(n,1))
    elif family==1:
        # Deep but spaced reentrant infield vertices; no smooth radial assumption.
        ids=np.arange(2,n,3);p[ids,:2]*=rng.uniform(.08,.5,(len(ids),1));motif[ids]=1
    elif family==2:
        # Outbound left-right chain, then a separated return lane.
        m=max(2,n//2);p[:m,0]=np.linspace(-radius,radius,m);p[:m,1]=np.where(np.arange(m)%2,1.,-1.)*radius*.15
        p[m:,0]=np.linspace(radius,-radius,n-m);p[m:,1]=radius*aspect*(.7+rng.uniform(-.12,.12,n-m));motif[:m]=2
    elif family==3:
        p[1::2,:2]*=rng.uniform(.12,.4,(len(p[1::2]),1));motif[1::2]=3
    elif family==4:
        p[:,2]=np.where(np.arange(n)<n//2,0,rng.uniform(2,7));motif[n//2:]=4
    elif family in (5,6):
        p[:,:2]*=rng.uniform(.75,1.2,(n,1))
        if family==6:
            ids=np.arange(2,n,4);p[ids,:2]*=.35;motif[ids]=1
    else:
        source=shape_parent('free_spline',n,rng);p=np.array([g.position for g in source.gates]);p[:,2]-=p[:,2].min()
    # Overall length independent of count, within conservative schema bounds.
    requested=np.exp(rng.uniform(np.log(max(25,n*2.5)),np.log(280)))
    p*=requested/np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1).sum()
    spacing=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)
    pair=np.linalg.norm(p[:,None]-p[None,:],axis=-1)+np.eye(n)*1e9
    p*=max(1.,.8/spacing.min(),.55/pair.min())
    p[:,:2]*=aug.uniform(.96,1.04,2)
    if family not in (4,7) and parent%3!=0:
        p[:,2]+=rng.uniform(0,3)*(.5+.5*np.sin(phase+rng.uniform(0,6)))
    # Clear physical stack, not a gate-roll label. Broad placement across families.
    stacks=[]
    if family in (5,6) or (parent%7==0 and n>=6):
        a=int(rng.integers(1,n-2));b=a+1
        p[a,2]=max(p[a,2],p[b,2])+rng.uniform(1.5,4.0)
        p[b,:2]=p[a,:2]+aug.normal(0,.08,2);p[b,2]=p[a,2]-rng.uniform(1.5,3.5)
        stacks.append((a,b));motif[a:b+1]=5
    inc=unit(p-np.roll(p,1,axis=0));out=unit(np.roll(p,-1,axis=0)-p)
    w=rng.uniform(.35,.65,(n,1));norm=unit(w*inc+(1-w)*out)
    # Upright bank/ground courses use horizontal bisectors, not pitched splines.
    if family!=7 and parent%3!=2:norm[:,2]=0;norm=unit(norm)
    if family not in (5,6):
        turn=np.arccos(np.clip(np.sum(inc*out,axis=1),-1,1))
        limit=np.minimum(.65,np.maximum(.05,np.deg2rad(92)-turn/2))
        yaw=rng.uniform(-1,1,n)*limit
        x=norm[:,0].copy();y=norm[:,1].copy()
        norm[:,0]=np.cos(yaw)*x-np.sin(yaw)*y;norm[:,1]=np.sin(yaw)*x+np.cos(yaw)*y
    # Approach/departure emphasis samples design choices without flipping travel.
    for a,b in stacks:
        v=inc[a].copy()-out[b];v[2]=0
        if np.linalg.norm(v)<.01:v=inc[a].copy();v[2]=0
        norm[a]=unit(v);norm[b]=-norm[a]
    sizes=np.tile(rng.uniform([1.1,1.2],[2.8,2.8]),(n,1))*aug.uniform(.97,1.03,(n,2))
    for a,b in stacks:
        # Apertures cannot overlap even when centers pass the spacing check.
        p[a,2]=max(p[a,2],p[b,2]+.5*(sizes[a,1]+sizes[b,1])+.15)
    rot=np.stack([np.array(Gate(q,forward_up_quaternion(v),s,'tmp').rotation) for q,v,s in zip(p,norm,sizes)])
    # True tilted portals as a separate mode, never confused with body roll.
    if family in (4,6,7) and not stacks and parent%4==1:
        j=int(rng.integers(n));angle=rng.uniform(-1.0,1.0)
        z=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        rot[j]=rot[j]@z;motif[j]=6
    halfz=(abs(rot[:,2,1])*sizes[:,0]+abs(rot[:,2,2])*sizes[:,1])/2
    # Explicit aperture-relative low/mid/high clearance strata.
    stratum=parent%3;clearance=(rng.uniform(.01,.25) if stratum==0 else rng.uniform(.4,1.5) if stratum==1 else rng.uniform(2,5))
    p[:,2]+=clearance-float(np.min(p[:,2]-halfz))
    opposite=np.zeros(n,bool)
    # Equivalent directed geometry with varied raw side encoding.
    if parent%4==0:
        opposite=aug.random(n)<.25;rot[opposite,:,:2]*=-1
    gates=tuple(Gate(q,matrix_quaternion(r),s,f'g{i}',enter_from_opposite_side=bool(o)) for i,(q,r,s,o) in enumerate(zip(p,rot,sizes,opposite)))
    loop=bool(parent%10!=9)
    track=Track(f'grammar_{index}',gates,bounds_for(gates),loop,{'family':FAMILIES[family],'motif':motif.tolist()})
    meta=dict(parent_id=parent,family=family,reference=-1,split=int(np.random.default_rng(np.random.SeedSequence([seed,parent,901])).integers(10)==0),
              primitive=int(bool(stacks)),attempt=attempt,strength=(.03,.06,.1)[index%3],ground_lift_m=0,clearance_stratum=stratum)
    return track,meta,motif
