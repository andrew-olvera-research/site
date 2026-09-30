"""Lossless ordered gate fields in a yaw/XY canonical frame (metres retained)."""
from dataclasses import replace
import numpy as np
from starscream.env.tracks import Gate, Track, matrix_quaternion

VERSION = 'course-v1-p3-rot6-logsize2-kind4-flags2-context19'
KINDS = ('gate', 'flag_route', 'over_route', 'maneuver_route')
GATE_DIM, CONTEXT_DIM = 17, 19


def pack(track):
    normal = track.gates[0].normal
    yaw = np.arctan2(normal[1], normal[0])
    c,s = np.cos(yaw),np.sin(yaw)
    frame = np.array([[c,-s,0],[s,c,0],[0,0,1]],np.float64)
    origin = np.array([*track.gates[0].position[:2],0.])
    rows=[]
    for g in track.gates:
        rot=frame.T@g.rotation
        rows.append(np.r_[(g.position-origin)@frame,rot[:,0],rot[:,1],
            np.log(g.size),np.eye(4)[KINDS.index(g.kind)],float(g.enter_from_opposite_side),float(g.render)])
    gates=np.array(rows,np.float32)
    corners=np.array(np.meshgrid(*track.bounds,indexing='ij')).reshape(3,-1).T
    corners=(corners-origin)@frame
    bounds=np.stack([corners.min(0),corners.max(0)],1)
    # Track files lack a universal initial-state schema. The debug corpus uses
    # an explicit gate-normal approach, not a claim to preserve benchmark reset.
    direction=frame.T@normal
    start=gates[0,:3]-3*direction
    start_rot=frame.T@track.gates[0].directed_rotation
    context=np.r_[start,start_rot[:,0],start_rot[:,1],3*direction,float(track.loop),bounds.ravel()].astype(np.float32)
    return gates,context,{'origin':origin.tolist(),'frame':frame.tolist()}


def rotations_numpy(raw):
    a=raw[...,:3];b=raw[...,3:6]
    if np.any(np.linalg.norm(a,axis=-1)<1e-6):raise ValueError('degenerate rotation')
    a=a/np.linalg.norm(a,axis=-1,keepdims=True)
    b=b-a*np.sum(a*b,axis=-1,keepdims=True)
    if np.any(np.linalg.norm(b,axis=-1)<1e-6):raise ValueError('collinear rotation')
    b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    return np.stack([a,b,np.cross(a,b)],-1)


def unpack(gates,context,name='decoded',transform=None):
    gates=np.asarray(gates);context=np.asarray(context)
    if gates.ndim!=2 or gates.shape[1]!=GATE_DIM or context.shape!=(CONTEXT_DIM,):
        raise ValueError('course tensor contract mismatch')
    if not np.isfinite(gates).all() or not np.isfinite(context).all():raise ValueError('nonfinite course')
    rot=rotations_numpy(gates[:,3:9])
    frame=np.eye(3) if transform is None else np.array(transform['frame'])
    origin=np.zeros(3) if transform is None else np.array(transform['origin'])
    out=tuple(Gate(position=frame@r[:3]+origin,quaternion_wxyz=matrix_quaternion(frame@q),
        size=np.exp(r[9:11]),name=f'gate_{i}',kind=KINDS[int(r[11:15].argmax())],
        enter_from_opposite_side=bool(r[15]>.5),render=bool(r[16]>.5)) for i,(r,q) in enumerate(zip(gates,rot)))
    bounds=context[13:19].reshape(3,2)
    corners=np.array(np.meshgrid(*bounds,indexing='ij')).reshape(3,-1).T@frame.T+origin
    bounds=np.stack([corners.min(0),corners.max(0)],1)
    return Track(name,out,bounds,bool(context[12]>.5),{'course_schema':VERSION})


def static_reasons(track):
    p=np.array([g.position for g in track.gates]);n=len(p)
    reasons=[]
    if not 4<=n<=32:return ['gate_count']
    segments=np.diff(np.r_[p,p[:1]] if track.loop else p,axis=0)
    spacing=np.linalg.norm(segments,axis=1)
    if spacing.min()<.6:reasons.append('adjacent_spacing')
    if not 15<=spacing.sum()<=450:reasons.append('length')
    d=np.linalg.norm(p[:,None]-p[None,:],axis=-1)+np.eye(n)*1e6
    if d.min()<.4:reasons.append('duplicate_centres')
    for g in track.gates:
        halfz=abs(g.rotation[2,1])*g.size[0]/2+abs(g.rotation[2,2])*g.size[1]/2
        if g.position[2]-halfz<-.00001:reasons.append('aperture_ground');break
    if not np.isfinite(p).all():reasons.append('nonfinite')
    return reasons
