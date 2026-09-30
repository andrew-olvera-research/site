"""Parent-preserving edits. Materialized geometry, not latent alpha, owns state.

Continuous displacement and discrete route changes have separate budgets.
Feasibility and policy reachability MUST still be measured by callers.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.spatial.transform import Rotation
from starscream.env.tracks import Track, Gate, matrix_quaternion, forward_up_quaternion, quaternion_matrix
from starscream.course_model.schema import rotations_numpy, static_reasons


@dataclass(frozen=True)
class EditBudget:
    position_m: float = 1.5
    rotation_deg: float = 20.
    curve_rms_m: float = 3.
    structural_events: int = 1

    def __post_init__(self):
        values=[self.position_m,self.rotation_deg,self.curve_rms_m]
        if not np.isfinite(values).all() or min(values)<0 or self.structural_events<0 or int(self.structural_events)!=self.structural_events:
            raise ValueError('finite nonnegative edit budgets required')


@dataclass
class CourseEdit:
    track: Track
    # Each child gate points at an original parent gate; -1 denotes insertion.
    correspondence: tuple[int, ...]
    removed: tuple[int, ...]
    operation: str
    metrics: dict
    reasons: list[str]


def curve(track, samples=256):
    p=np.asarray([g.position for g in track.gates], dtype=float)
    if track.loop:p=np.r_[p,p[:1]]
    lengths=np.linalg.norm(np.diff(p,axis=0),axis=1)
    if np.any(lengths<1e-8):raise ValueError('degenerate route edge')
    s=np.r_[0.,np.cumsum(lengths)]
    t=np.linspace(0,s[-1],samples,endpoint=not track.loop)
    return np.stack([np.interp(t,s,p[:,j]) for j in range(3)],-1)


def measured(parent, child, mapping, operation, budget=EditBudget()):
    mapping=tuple(int(i) for i in mapping)
    if len(mapping)!=len(child.gates) or any(i < -1 for i in mapping):raise ValueError('correspondence length/index')
    kept=[i for i in mapping if i>=0]
    if kept!=sorted(set(kept)) or any(i>=len(parent.gates) for i in kept):
        raise ValueError('correspondence must preserve ordering without duplicates')
    removed=tuple(i for i in range(len(parent.gates)) if i not in kept)
    # Preserve reset/config metadata, but never label an edited benchmark exact.
    metadata=dict(child.metadata or {})
    references={k:metadata.pop(k) for k in list(metadata) if k.startswith('published_')}
    if references:metadata['parent_published_reference']=references
    metadata.update(provenance='synthetic-parent-edit-v4',description='Derived synthetic course; not exact benchmark geometry.',
                    edit_parent=parent.name,edit_operation=operation)
    child=replace(child,metadata=metadata)
    distances=[];angles=[];semantic=0
    for g,i in zip(child.gates,mapping):
        if i<0:continue
        p=parent.gates[i]
        distances.append(float(np.linalg.norm(g.position-p.position)))
        angles.append(float(np.degrees(Rotation.from_matrix(g.rotation.astype(float)@p.rotation.T).magnitude())))
        semantic+=int((g.kind,g.enter_from_opposite_side,g.render)!=(p.kind,p.enter_from_opposite_side,p.render))
    rms=float(np.sqrt(np.mean(np.sum((curve(child)-curve(parent))**2,axis=1))))
    m=dict(preserved_position_max_m=max(distances,default=0.),preserved_rotation_max_deg=max(angles,default=0.),
           curve_rms_m=rms,removed=len(removed),inserted=mapping.count(-1),semantic_changes=semantic,
           gate_count=len(child.gates))
    reasons=list(static_reasons(child))
    if m['preserved_position_max_m']>budget.position_m+1e-5:reasons.append('position_budget')
    if m['preserved_rotation_max_deg']>budget.rotation_deg+1e-3:reasons.append('rotation_budget')
    if rms>budget.curve_rms_m:reasons.append('curve_budget')
    if len(removed)+mapping.count(-1)+semantic>budget.structural_events:reasons.append('structural_budget')
    if mapping[0]!=0:reasons.append('start_gate_changed')
    for g in child.gates:
        if np.any(g.position<child.bounds[:,0]) or np.any(g.position>child.bounds[:,1]):
            reasons.append('bounds');break
    return CourseEdit(child,mapping,removed,operation,m,reasons)


def remove_gate(parent, index, *, name='remove', budget=EditBudget()):
    if not 0<index<len(parent.gates) or len(parent.gates)<=4:raise ValueError('invalid removal; preserve start and minimum count')
    mapping=tuple(i for i in range(len(parent.gates)) if i!=index)
    return measured(parent,replace(parent,name=name,gates=tuple(parent.gates[i] for i in mapping)),mapping,'remove',budget)


def insert_gate(parent, after, fraction=.5, *, name='insert', budget=EditBudget()):
    if not 0<=after<len(parent.gates)-(not parent.loop) or not 0<fraction<1 or len(parent.gates)>=32:
        raise ValueError('invalid insertion')
    a=parent.gates[after];b=parent.gates[(after+1)%len(parent.gates)]
    delta=Rotation.from_matrix(b.rotation.astype(float)@a.rotation.T).as_rotvec()
    rot=Rotation.from_rotvec(fraction*delta).as_matrix()@a.rotation
    g=replace(a,position=(1-fraction)*a.position+fraction*b.position,
              quaternion_wxyz=matrix_quaternion(rot),name='inserted_gate')
    gates=parent.gates[:after+1]+(g,)+parent.gates[after+1:]
    mapping=tuple(range(after+1))+(-1,)+tuple(range(after+1,len(parent.gates)))
    return measured(parent,replace(parent,name=name,gates=gates),mapping,'insert',budget)


def deform(parent, translation=None, rotvec=None, *, strength=1., name='deform', budget=EditBudget()):
    n=len(parent.gates)
    t=np.zeros((n,3)) if translation is None else np.asarray(translation,float)
    r=np.zeros((n,3)) if rotvec is None else np.asarray(rotvec,float)
    if t.shape!=(n,3) or r.shape!=(n,3) or not np.isfinite([t,r]).all() or not np.isfinite(strength):
        raise ValueError('nonfinite or incorrect edit shape')
    if np.any(t[0]) or np.any(r[0]):raise ValueError('start gate is locked')
    gates=[]
    for i,g in enumerate(parent.gates):
        if strength==0 or (not np.any(t[i]) and not np.any(r[i])):gates.append(g);continue
        q=matrix_quaternion(Rotation.from_rotvec(strength*r[i]).as_matrix()@g.rotation)
        gates.append(replace(g,position=g.position+strength*t[i],quaternion_wxyz=q))
    return measured(parent,replace(parent,name=name,gates=tuple(gates)),range(n),'deform',budget)


def codec_delta(parent, base_decode, edited_decode, frame, *, strength=1., name='codec_delta', budget=EditBudget()):
    """Transport same-chart decoder DIFFERENCES onto the actual parent.

    Decode both with identical context/count/noise. After a structural edit,
    encode the materialized child and start a new chart; never compare n vs n-1
    decoded arrays. Discrete fields/sizes are untouched, not silently predicted.
    """
    a=np.asarray(base_decode);b=np.asarray(edited_decode);frame=np.asarray(frame)
    if a.shape!=b.shape or a.shape!=(len(parent.gates),17):raise ValueError('codec chart mismatch')
    if frame.shape!=(3,3) or not np.allclose(frame.T@frame,np.eye(3),atol=1e-5) or np.linalg.det(frame)<0:
        raise ValueError('invalid canonical frame')
    ra=rotations_numpy(a[:,3:9]);rb=rotations_numpy(b[:,3:9])
    t=(b[:,:3]-a[:,:3])@frame.T
    rv=Rotation.from_matrix(frame[None]@(rb@ra.transpose(0,2,1))@frame.T[None]).as_rotvec()
    t[0]=0;rv[0]=0
    if np.array_equal(a,b):rv[:]=0 # Exact identity, including float rotation noise.
    return deform(parent,t,rv,strength=strength,name=name,budget=budget)


def prepare_removal(parent, index, *, fraction=1., position_step_m=.5,
                    rotation_step_deg=10., name='prepare_removal', budget=EditBudget()):
    """Continuous predecessor to removal, not a promise of policy reachability.

    Move the gate toward the interior of the bypass chord and align its directed
    traversal normal with that chord. Explicit local geometry intent; no target
    course coordinates are used. Neighbors and gate count remain unchanged.
    """
    if not 0<index<len(parent.gates)-(not parent.loop) or not 0<=fraction<=1:
        raise ValueError('invalid preparation index/fraction')
    if not np.isfinite([position_step_m,rotation_step_deg]).all() or min(position_step_m,rotation_step_deg)<=0:
        raise ValueError('positive finite preparation step budgets required')
    gate=parent.gates[index];a=parent.gates[index-1].position.astype(float)
    b=parent.gates[(index+1)%len(parent.gates)].position.astype(float);chord=b-a
    if np.linalg.norm(chord)<.6:raise ValueError('bypass chord is degenerate')
    progress=np.clip(np.dot(gate.position-a,chord)/np.dot(chord,chord),.2,.8)
    displacement=a+progress*chord-gate.position
    normal=chord*(-1 if gate.enter_from_opposite_side else 1)
    target=quaternion_matrix(forward_up_quaternion(normal,gate.up))
    rotation=Rotation.from_matrix(target.astype(float)@gate.rotation.T).as_rotvec()
    scale=fraction*min(1.,position_step_m/max(np.linalg.norm(displacement),1e-9),
                       np.radians(rotation_step_deg)/max(np.linalg.norm(rotation),1e-9))
    t=np.zeros((len(parent.gates),3));rv=t.copy();t[index]=displacement;rv[index]=rotation
    result=deform(parent,t,rv,strength=scale,name=name,budget=budget)
    result.operation='prepare_removal'
    return result


def balanced_shortlist(rows, budget, seed):
    """Randomized strata + randomized within-stratum order; no signed-side bias."""
    rng=np.random.default_rng(seed);buckets={}
    for row in rows:
        if not row['reasons']:buckets.setdefault((row['operation'],row['gate_count']),[]).append(row)
    keys=list(buckets);rng.shuffle(keys)
    for values in buckets.values():rng.shuffle(values)
    selected=[]
    while len(selected)<budget and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected)<budget:selected.append(buckets[key].pop())
    return selected
