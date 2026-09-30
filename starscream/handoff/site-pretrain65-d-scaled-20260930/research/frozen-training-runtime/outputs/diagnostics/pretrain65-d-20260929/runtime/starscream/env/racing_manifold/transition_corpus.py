"""Explicit transition proposals and audits; labels never certify behavior.

All vectors are world-frame geometry. Directed normals, not gate appearance,
define crossing direction. Generated courses still require dynamic admission.
"""
from dataclasses import replace
import numpy as np
from ..tracks import Track, forward_up_quaternion

GRAMMARS = ('banking', 'slalom', 'braking_reversal', 'stacked_reversal', 'go_around')


def transition_metrics(track):
    p = np.asarray([g.position for g in track.gates], float)
    incoming = p - np.roll(p, 1, axis=0)
    outgoing = np.roll(p, -1, axis=0) - p
    rows = []
    for i, g in enumerate(track.gates):
        a, b = incoming[i], outgoing[i]
        n = np.asarray(g.normal, float)
        angle = np.degrees(np.arccos(np.clip(a @ b / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-9), -1, 1)))
        rows.append(dict(gate=i, incoming_m=float(np.linalg.norm(a)), outgoing_m=float(np.linalg.norm(b)),
            turn_deg=float(angle), height_change_m=float(a[2]),
            incoming_alignment=float(a @ n / max(np.linalg.norm(a), 1e-9)),
            preceding_gate_on_exit_side=bool((-a) @ n > .5),
            reverse_entry=bool(g.enter_from_opposite_side)))
    return rows


def graft_transition(track, grammar, seed, name):
    """One 3-gate motif on an independent spline parent, no geometry copying.

    Stacked reversal means opposing directed crossings at different heights;
    it is split-S-like geometry, NOT a guarantee of inverted body attitude.
    """
    if grammar not in GRAMMARS:
        raise ValueError(grammar)
    if len(track.gates) < 6:
        raise ValueError('need at least six gates')
    rng = np.random.default_rng(seed)
    gates = list(track.gates)
    anchor = int(rng.integers(1, len(gates)-3))
    ids = [anchor, anchor+1, anchor+2]
    first = np.asarray(gates[anchor].position, float)
    e = first - gates[anchor-1].position
    e[2] = 0
    e /= max(np.linalg.norm(e), 1e-9)
    side = np.cross([0., 0., 1.], e)
    side *= rng.choice([-1., 1.])
    run = float(rng.uniform(3.5, 5.5))
    lateral = float(rng.uniform(2.0, 3.5))
    dz = float(rng.uniform(2.5, 3.5))
    if grammar == 'banking':
        ids = []  # unmodified geometry control, measured banking only
    elif grammar == 'slalom':
        offsets = [(0,0,0), (run,lateral,0), (2*run,-lateral,.2)]
        directions = [e, e, e]
    elif grammar == 'braking_reversal':
        offsets = [(0,0,0), (run,lateral,.3), (0,2*lateral,0)]
        directions = [e, side, -e]
    elif grammar == 'stacked_reversal':
        first[2] = max(first[2], 2.0 + dz)
        offsets = [(0,0,0), (.2,lateral*.2,-dz), (-run,lateral,-dz)]
        directions = [e, -e, -e]
    elif grammar == 'go_around':
        offsets = [(0,0,0), (run,lateral,0), (0,2*lateral,0)]
        # Preceding gate lies on target's directed exit side. The aircraft
        # must reach the opposite half-space before its valid crossing.
        directions = [e, -e, -e]
    for j, i in enumerate(ids):
        x, y, z = offsets[j]
        position = first + x*e + y*side + np.array([0.,0.,z])
        reverse = grammar == 'go_around' and j == 1
        physical = -directions[j] if reverse else directions[j]
        gates[i] = replace(gates[i], position=position.astype(np.float32),
            quaternion_wxyz=forward_up_quaternion(physical), enter_from_opposite_side=reverse)
    p = np.asarray([g.position for g in gates])
    bounds = np.stack((p.min(0)-4, p.max(0)+4), axis=1)
    bounds[2,0] = 0
    meta = dict(track.metadata or {})
    meta.update(intended_transition=grammar, transition_gate_indices=ids,
        transition_parent=track.name, behavior_certified=False,
        maneuver_scope='directed geometry; inversion requires executed-state evidence')
    return Track(name, tuple(gates), bounds.astype(np.float32), track.loop, meta)


def static_admission(track):
    p = np.asarray([g.position for g in track.gates], float)
    reasons = []
    if not np.isfinite(p).all(): return ['nonfinite']
    if not 6 <= len(p) <= 8: reasons.append('gate_count')
    distance = np.linalg.norm(p[:,None]-p[None,:], axis=-1)
    np.fill_diagonal(distance, np.inf)
    if distance.min() < 1.5: reasons.append('gate_spacing')
    length = np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1).sum()
    if not 30 <= length <= 130: reasons.append('length')
    for g in track.gates:
        # Full physical aperture, not just the center, must clear the floor.
        extent = abs(g.lateral[2])*g.size[0]/2 + abs(g.up[2])*g.size[1]/2
        if g.position[2]-extent < .6: reasons.append('floor_clearance')
        if not np.allclose(g.directed_rotation.T@g.directed_rotation,np.eye(3),atol=1e-5):
            reasons.append('frame')
    kind = (track.metadata or {}).get('intended_transition')
    ids = (track.metadata or {}).get('transition_gate_indices', [])
    if kind == 'go_around' and not transition_metrics(track)[ids[1]]['preceding_gate_on_exit_side']:
        reasons.append('missing_go_around')
    if kind == 'stacked_reversal':
        a,b = [track.gates[i] for i in ids[:2]]
        if a.normal @ b.normal > -.9 or a.position[2]-b.position[2] < 2.:
            reasons.append('missing_stacked_reversal')
    return sorted(set(reasons))


def shape_distance(a, b):
    """Cyclic, yaw/translation/scale-invariant full-course RMS; no reflection.

    A diagnostic anti-clone metric, not a transfer predictor. Different gate
    counts are not compared here and must be audited separately.
    """
    if len(a.gates) != len(b.gates): return None
    x = np.asarray([g.position for g in a.gates], float)
    y = np.asarray([g.position for g in b.gates], float)
    x -= x.mean(0); y -= y.mean(0)
    x /= max(np.sqrt((x*x).sum(1).mean()), 1e-8)
    y /= max(np.sqrt((y*y).sum(1).mean()), 1e-8)
    values = []
    for shift in range(len(x)):
        z = np.roll(y, shift, axis=0)
        angle = np.arctan2(np.sum(x[:,0]*z[:,1]-x[:,1]*z[:,0]), np.sum(x[:,:2]*z[:,:2]))
        c,s = np.cos(angle),np.sin(angle)
        rotated = x @ np.array([[c,s,0],[-s,c,0],[0,0,1]])
        values.append(float(np.sqrt(np.mean(np.sum((rotated-z)**2,axis=1)))))
    return min(values)


def executed_transition_metrics(trace):
    """Phase-conditioned braking/turn evidence from world-state trajectories.

    Phase IDs are relative to the rollout start; all-gate prefix audits must
    join with the stored start_gate_index before labeling physical gate IDs.
    """
    s=np.asarray(trace['states'],float);ns=np.asarray(trace['next_states'],float)
    phase=np.asarray(trace['phase']);hz=float(trace['control_hz'])
    if s.ndim!=2 or s.shape[1]<13 or s.shape!=ns.shape or phase.shape!=(len(s),) or hz<=0:
        raise ValueError('invalid executed trajectory')
    if not np.isfinite(s).all() or not np.isfinite(ns).all():raise ValueError('nonfinite state')
    v=s[:,7:10];a=(ns[:,7:10]-v)*hz;speed=np.linalg.norm(v,axis=1)
    tangential=(a*v).sum(1)/np.maximum(speed,.5)
    heading=(v[:,0]*a[:,1]-v[:,1]*a[:,0])/np.maximum((v[:,:2]**2).sum(1),1.)
    rows=[]
    for p in np.unique(phase):
        mask=phase==p;ss=speed[mask]
        rows.append(dict(relative_gate_phase=int(p),duration_s=float(mask.sum()/hz),
            entry_speed_mps=float(ss[0]),exit_speed_mps=float(ss[-1]),minimum_speed_mps=float(ss.min()),
            peak_speed_mps=float(ss.max()),
            braking_p90_mps2=float(np.quantile(np.maximum(-tangential[mask],0),.9)),
            signed_horizontal_heading_change_rad=float(heading[mask].sum()/hz)))
    return rows
