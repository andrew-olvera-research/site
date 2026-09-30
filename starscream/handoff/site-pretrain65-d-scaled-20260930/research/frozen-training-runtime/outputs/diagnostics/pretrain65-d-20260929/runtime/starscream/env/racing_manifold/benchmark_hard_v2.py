"""Behavior-first hard-course generator for the v6.22 hard-v2 experiment.

Unlike the original family generator, this samples a behavior program first
and realizes it geometrically second. The program is independent of the named
real60 family labels, making whole-course shape and behavioral requirements
separate sampling axes.
"""
from dataclasses import replace
import numpy as np

from starscream.env.tracks import Gate, Track, forward_up_quaternion

STRATA = ('long_low_braking', 'vertical_chain', 'wrong_side_incidence',
          'compound_reversal', 'radius_switch')


def _heading(angle): return np.array([np.cos(angle), np.sin(angle), 0.], float)


def _unit(value):
    value=np.asarray(value,float); return value/max(float(np.linalg.norm(value)),1e-9)


def _return_points(exit_point, exit_heading, entry_point, count, rng):
    if count <= 0: return np.empty((0,3),float)
    # Keep the closure inside the benchmark leg envelope.  The first version
    # let the Bézier handles grow with the motif chord; a long programmed edge
    # could therefore create a 60m+ *return* edge and fail admission for a
    # reason unrelated to the intended behavior cell.
    chord=np.linalg.norm(entry_point[:2]-exit_point[:2]); reach=min(0.55*chord, 24.+2.*count)
    c1=exit_point+reach*_heading(exit_heading)
    c2=entry_point-reach*_heading(0.)
    side=np.cross([0.,0.,1.],_heading(exit_heading)); c1+=rng.uniform(-10.,10.)*side; c2+=rng.uniform(-10.,10.)*side
    points=[]
    for t in np.linspace(0.,1.,count+2)[1:-1]:
        b=((1-t)**3)*exit_point+3*((1-t)**2)*t*c1+3*(1-t)*(t**2)*c2+(t**3)*entry_point
        points.append(b)
    return np.asarray(points)


def _program(stratum, rng, n):
    """Return edge lengths, signed heading changes, and vertical edge changes."""
    lengths=rng.uniform(6.,12.,n-1); turns=rng.uniform(-35.,35.,n-1); dz=rng.uniform(-.6,.6,n-1)
    if stratum == 'long_low_braking':
        slots=rng.choice(np.arange(1,n-2),2,replace=False); slots.sort()
        lengths[slots[0]-1]=rng.uniform(34.,50.); lengths[slots[1]-1]=rng.uniform(28.,44.)
        turns[slots[0]-1]=rng.choice([-1.,1.])*rng.uniform(145.,175.)
        turns[slots[1]-1]=rng.choice([-1.,1.])*rng.uniform(135.,170.)
        dz[slots[0]-1]=rng.uniform(-.2,.2); dz[slots[1]-1]=rng.uniform(-.2,.2)
        return lengths,turns,dz,dict(long_low_edges=slots.tolist(), braking_turns=[float(turns[s-1]) for s in slots])
    if stratum == 'vertical_chain':
        k=min(5,n-2); lengths[:k]=rng.uniform(8.,14.,k); turns[:k]=rng.choice([-1.,1.],k)*rng.uniform(112.,158.,k)
        # Preserve a multi-edge vertical chain without asking the teacher to
        # recover from four-metre discontinuities at racing speed.
        dz[:k]=np.array([-rng.uniform(2.3,3.5),-rng.uniform(2.2,3.3),rng.uniform(2.8,3.8),rng.uniform(-3.0,-2.0),rng.uniform(2.3,3.4)])[:k]
        return lengths,turns,dz,dict(vertical_chain_edges=list(range(k)), signed_dz=dz[:k].tolist())
    if stratum == 'wrong_side_incidence':
        k=min(5,n-2); lengths[:k]=rng.uniform(14.,30.,k); turns[:k]=rng.choice([-1.,1.],k)*rng.uniform(105.,165.,k); dz[:k]=rng.uniform(-1.5,1.5,k)
        return lengths,turns,dz,dict(wrong_side_edges=list(range(k)), incidence_deg=float(rng.uniform(25.,42.)))
    if stratum == 'compound_reversal':
        k=min(7,n-2); lengths[:k]=rng.uniform(5.,10.,k); turns[:k]=rng.choice([-1.,1.],k)*rng.uniform(102.,158.,k); dz[:k]=rng.uniform(-2.6,2.6,k)
        return lengths,turns,dz,dict(compound_edges=list(range(k)))
    # Alternating short radius and long approach segments forces changes in
    # braking/turning mode without inheriting a named family template.
    for i in range(min(7,n-2)):
        lengths[i]=rng.uniform(3.,5.) if i%2 else rng.uniform(24.,42.)
        turns[i]=rng.choice([-1.,1.])*rng.uniform(95.,175.)
        dz[i]=rng.uniform(-2.8,2.8)
    return lengths,turns,dz,dict(radius_switch_edges=list(range(min(7,n-2))))


def _generate_hard_v2_once(stratum, count, seed, name):
    if stratum not in STRATA: raise ValueError(stratum)
    if not 8 <= count <= 18: raise ValueError('hard-v2 count must be 8..18')
    rng=np.random.default_rng(seed); n=int(count); motif=n-int(rng.integers(1,4)); motif=max(6,motif)
    lengths,turns,dz,signature=_program(stratum,rng,motif)
    z0=float(rng.uniform(3.5,6.5)); points=[np.array([0.,0.,z0])]; heading=float(rng.uniform(-np.pi,np.pi))
    for length,turn,delta in zip(lengths,turns,dz):
        points.append(points[-1]+length*_heading(heading)+np.array([0.,0.,delta])); heading+=np.radians(turn)
    points=np.vstack([points,_return_points(points[-1],heading,points[0],n-len(points),rng)])
    # Keep all intended center heights in the raceable envelope while retaining
    # low centers for the long-low program.
    points[:,2]=np.clip(points[:,2],.9,9.5)
    if stratum == 'long_low_braking':
        # Decouple low-gate braking from the global course altitude. The two
        # selected long edges terminate at genuinely low centers, while the
        # remainder can stay high enough for a visually legible 3-D route.
        for target in signature['long_low_edges']:
            if 0 < target < len(points): points[target,2]=float(rng.uniform(.95,1.2))
    mirror=-1. if rng.integers(2) else 1.; points[:,1]*=mirror; heading*=mirror
    yaw=float(rng.uniform(-np.pi,np.pi)); c,s=np.cos(yaw),np.sin(yaw); R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    points=points@R.T
    gates=[]; incidence=float(signature.get('incidence_deg',0.))
    for i,point in enumerate(points):
        prev=points[(i-1)%n]; nxt=points[(i+1)%n]; incoming=_unit(point-prev); outgoing=_unit(nxt-point); normal=incoming+outgoing; normal[2]=0
        if np.linalg.norm(normal)<1e-6: normal=outgoing; normal[2]=0
        normal=_unit(normal)
        if stratum == 'wrong_side_incidence' and i in signature['wrong_side_edges']:
            angle=np.radians(incidence)*rng.choice([-1.,1.]); ca,sa=np.cos(angle),np.sin(angle); normal=np.array([ca*normal[0]-sa*normal[1],sa*normal[0]+ca*normal[1],0.])
        width=float(rng.uniform(1.20,1.58) if stratum in ('long_low_braking','radius_switch') else rng.uniform(1.25,1.85))
        height=float(rng.uniform(1.20,min(2.2,max(1.2,2.*(float(point[2])-.12)))))
        opposite=bool(stratum=='wrong_side_incidence' and i in signature['wrong_side_edges'] and i%2==0)
        gates.append(Gate(point.astype(np.float32),forward_up_quaternion(normal),np.array([width,height],np.float32),name=f'gate_{i:02d}',enter_from_opposite_side=opposite))
    bounds=np.stack((points.min(0)-7.,points.max(0)+7.),axis=1).astype(np.float32); bounds[2,0]=0.
    signature.update(stratum=stratum, seed=int(seed), count=n, signed_turns=[float(x) for x in turns], center_heights=points[:,2].tolist())
    return Track(name,tuple(gates),bounds,True,dict(source='v622-hard-v2-behavior-program',behavior_signature=signature,hard=True,behavior_certified=False))


def generate_hard_v2(stratum, count, seed, name):
    """Generate a behavior-program course with deterministic admission retries.

    Retries only change the realization seed; the requested stratum and its
    programmed requirement remain fixed. This prevents closure artefacts (or
    an unlucky near-coincident return gate) from silently becoming a selected
    hard cell.
    """
    from .benchmark_v22 import validate_geometry
    for attempt in range(48):
        realization_seed = int(seed) + attempt * 1000003
        track = _generate_hard_v2_once(stratum, count, realization_seed, name)
        if not validate_geometry(track):
            # Preserve the caller's logical seed in the signature while still
            # recording the realization seed for exact reproduction.
            sig = dict((track.metadata or {}).get('behavior_signature', {}))
            sig['program_seed'] = int(seed)
            sig['realization_attempt'] = int(attempt)
            metadata = dict(track.metadata or {})
            metadata['behavior_signature'] = sig
            return Track(track.name, track.gates, track.bounds, track.loop, metadata)
    raise RuntimeError(f'could not realize valid hard-v2 course: {stratum}/{count}/{seed}')
