"""v6.20.1: complete loop proposals with protected maneuver joins.

Unlike v6.20's graft, the return leg is built around the maneuver. No real
track coordinates are used. A static witness is not dynamic certification.
"""
import numpy as np
from ..tracks import Gate, Track, forward_up_quaternion
from .transition_corpus import static_admission, transition_metrics


def generate_course(grammar, count, seed, name, revision=1):
    if not 4 <= count <= 10:raise ValueError('gate count must be 4..10')
    rng=np.random.default_rng(seed)
    if grammar in ('stacked_reversal','go_around'):
        width=rng.uniform(9.,15.);depth=rng.uniform(7.,12.)
        if grammar=='stacked_reversal':
            low=rng.uniform(1.05,3.) if revision>=4 else (rng.uniform(1.45,3.) if revision>=2 else rng.uniform(2.5,3.8))
            drop=rng.uniform(1.8,6.) if revision>=4 else (rng.uniform(2.2,6.) if revision>=2 else rng.uniform(4.,6.))
            p=[[0,0,low+drop],[0,rng.uniform(-.3,.3),low]]
            normal=[np.array([-1.,0,0]),np.array([1.,0,0])]
            # Broad returning arc rises back to the upper crossing. Alternate
            # lateral sign is a genuine mirrored maneuver, not a shape clone
            # presented as independent validation.
            ts=np.linspace(-1.1,1.1,count-2)
            for j,t in enumerate(ts):
                p.append([width*np.cos(t)+width*.4,depth*np.sin(t),low+drop*(j+1)/(count-1)])
        else:
            z=rng.uniform(3.,6.)
            p=[[0,0,z],[width*(.12 if revision>=2 else .8),depth*(1. if revision>=2 else .55),z+rng.uniform(-.5,.5)]]
            normal=[np.array([1.,0,0]),np.array([-1.,0,0])]
            ts=np.linspace(1.2,3.6,count-2) if revision>=2 else np.linspace(.5,2.9,count-2)
            for t in ts:p.append([width*.7*np.cos(t)-width*.4,depth*np.sin(t)+depth*.4,z+rng.uniform(-.6,.6)])
        p=np.asarray(p,float)
    else:
        # Independent irregular radial loops, with actual turn magnitudes
        # measured afterwards. Whole geometry varies, not only gate jitter.
        increments=rng.uniform(.7,1.3,count);angles=np.cumsum(increments)/increments.sum()*2*np.pi
        radial=rng.uniform(8.,14.,count);aspect=rng.uniform(.65,1.4)
        p=np.column_stack((radial*np.cos(angles),aspect*radial*np.sin(angles),
            rng.uniform(1.05,5.8,count) if revision>=4 else (rng.uniform(1.3,5.8,count) if revision>=2 else rng.uniform(2.8,5.8,count))))
        if grammar=='slalom':
            p[1:3,:2]+=np.array([[0,3],[0,-3]])*rng.uniform(.6,1.2)
        elif grammar=='braking_reversal':
            p[1,:2]=p[0,:2]+[rng.uniform(6,10),rng.uniform(3,5)]
            p[2,:2]=p[0,:2]+[rng.uniform(-1,1),rng.uniform(7,10)]
        elif grammar!='banking':raise ValueError(grammar)
        normal=[]
    if revision>=3 and grammar=='slalom':
        # A real alternating signed-turn witness, not a label on a convex loop.
        run=rng.uniform(8.,12.);amplitude=rng.uniform(2.5,4.)
        height=rng.uniform(2.3,4.5)
        p=[[-run,-amplitude,height],[0,amplitude,height+rng.uniform(-.5,.5)],
           [run,-amplitude,height+rng.uniform(-.5,.5)]]
        for angle in np.linspace(0,np.pi,count-1)[1:-1]:
            p.append([run*np.cos(angle),rng.uniform(9.,13.)*np.sin(angle)+amplitude,
                      height+rng.uniform(-.7,.7)])
        p=np.asarray(p,float)
    normals=[]
    for i in range(count):
        a=p[i]-p[(i-1)%count];b=p[(i+1)%count]-p[i]
        a/=np.linalg.norm(a);b/=np.linalg.norm(b)
        n=a+b;n[2]=0
        if np.linalg.norm(n)<1e-7:n=b.copy();n[2]=0
        n/=np.linalg.norm(n)
        normals.append(normal[i] if i<len(normal) else n)
    if revision>=4:
        for i in range(len(normal),count):
            # Gate-plane incidence is not determined by centerline turn angle.
            # Keep protected opposing crossings exact; vary other plane yaws.
            angle=np.radians(rng.uniform(-25.,25.));c,s=np.cos(angle),np.sin(angle)
            normals[i]=np.array([[c,-s,0],[s,c,0],[0,0,1]])@normals[i]
    # Transform course and directed frames together; never transform just
    # positions. Proper yaw rotation preserves gravity and handedness.
    mirror=-1 if revision>=3 and seed%2 else 1
    # Reflect forward vectors, then rebuild proper right-handed frames below.
    # Reflecting an entire orientation matrix would incorrectly give det=-1.
    p[:,1]*=mirror
    normals=[np.asarray(n)*[1,mirror,1] for n in normals]
    yaw=rng.uniform(-np.pi,np.pi);c,s=np.cos(yaw),np.sin(yaw)
    R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    p=p@R.T
    gates=[]
    for i,(point,n) in enumerate(zip(p,normals)):
        reverse=grammar=='go_around' and i==1
        n=R@n
        gates.append(Gate(point.astype(np.float32),forward_up_quaternion(-n if reverse else n),
            np.array([rng.uniform(1.45,2.5) if revision>=2 else rng.uniform(2.1,2.5),
                      rng.uniform(1.45,2.3) if revision>=2 else rng.uniform(1.9,2.3)],np.float32),
            name=f'gate_{i:02}',enter_from_opposite_side=reverse))
    bounds=np.stack((p.min(0)-4,p.max(0)+4),axis=1);bounds[2,0]=0
    ids=[0,1] if grammar in ('stacked_reversal','go_around') else []
    if revision>=4 and ids:
        # The protected maneuver must not always occur at a cold initial spawn.
        # Move it within the single-lap sequence, never across the terminal cut.
        offset=int(rng.integers(0,count-1))
        gates=list(np.roll(np.asarray(gates,dtype=object),offset))
        ids=[offset,offset+1]
    return Track(name,tuple(gates),bounds.astype(np.float32),True,dict(
        source='v6201_complete_loop',revision=revision,seed=seed,mirror=mirror,intended_transition=grammar,
        transition_gate_indices=ids,behavior_certified=False))


def screen(track):
    # Reuse physical invariants without modifying the versioned v6.20 rules.
    reasons=[r for r in static_admission(track) if r!='gate_count']
    if (track.metadata or {}).get('revision',1)>=2:
        reasons=[r for r in reasons if r!='floor_clearance']
        if any(g.position[2]-abs(g.lateral[2])*g.size[0]/2-abs(g.up[2])*g.size[1]/2<.25 for g in track.gates):
            reasons.append('floor_clearance')
    if (track.metadata or {}).get('revision',1)>=4 and track.metadata['intended_transition']=='stacked_reversal':
        reasons=[r for r in reasons if r!='missing_stacked_reversal']
        a,b=[track.gates[i] for i in track.metadata['transition_gate_indices']]
        if a.normal@b.normal>-.9 or a.position[2]-b.position[2]<1.8:reasons.append('missing_stacked_reversal')
    if not 4<=len(track.gates)<=10:reasons.append('gate_count')
    if max(r['incoming_m'] for r in transition_metrics(track))>35:reasons.append('long_join')
    if (track.metadata or {}).get('revision',1)>=3 and track.metadata['intended_transition']=='slalom':
        p=np.asarray([g.position for g in track.gates]);a=p-np.roll(p,1,0);b=np.roll(p,-1,0)-p
        turns=np.degrees(np.arctan2(a[:,0]*b[:,1]-a[:,1]*b[:,0],(a[:,:2]*b[:,:2]).sum(1)))
        if not np.any((turns*np.roll(turns,-1)<0)&(abs(turns)>20)&(abs(np.roll(turns,-1))>20)):
            reasons.append('missing_alternating_turns')
    return reasons
