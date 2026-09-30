"""Classical, interpretable course demand and geometry/behavior compass.

Not a learned transfer value. Features use actual simulator transitions, not
gate-relative finite differences or the MPCC horizon plan. All distances retain
episode ordering; local coverage additionally retains within-window ordering.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import cdist
from starscream.env.racing_manifold.compass import ControlWindows, ordered_distance


GROUPS = {
    'motion': ('speed', 'climb_speed', 'tangential_accel', 'lateral_accel',
               'signed_heading_rate', 'vertical_accel', 'jerk'),
    'effort': ('thrust', 'body_rate', 'command_body_rate'),
    'grammar': ('entry_alignment', 'gate_width', 'gate_height', 'gate_up_z',
                'next_turn', 'next_climb', 'next_distance', 'gate_duration'),
}
# Physical, fixed units: no fitting to validation or target observations.
SCALES = np.array([20, 10, 30, 30, 5, 30, 300, 40, 10, 10,
                   1, 3, 3, 1, np.pi, 10, 20, 2.])


@dataclass(frozen=True)
class Demand:
    samples: np.ndarray  # gate-phase bins x time-scales x three offsets x channels
    gates: np.ndarray
    summaries: dict


def extract_demand(trace, track, *, horizons=(.1, .3, .6), bins=6):
    """Canonical full one-lap traces only, with terminal windows shifted earlier.

    End windows are never wrapped or padded. Anchors are restricted to their
    gate interval; shifted window starts may lie in a preceding gate, preserving
    the real approach context. No derivative crosses an episode reset.
    """
    s, ns, a = (np.asarray(trace[k], float) for k in ('states', 'next_states', 'actions'))
    phase = np.asarray(trace['phase'])
    hz = float(trace['control_hz'])
    horizons = np.asarray(horizons, float)
    if (not bool(trace['success']) or hz <= 0 or not np.isfinite(hz)
            or horizons.ndim != 1 or not len(horizons) or not np.isfinite(horizons).all()
            or (horizons < 0).any() or bins < 2):
        raise ValueError('need successful full episode, finite positive Hz and valid windows')
    if (s.ndim != 2 or s.shape[1] < 13 or ns.shape != s.shape or a.shape != (len(s), 4)
            or phase.shape != (len(s),) or not all(np.isfinite(x).all() for x in (s, ns, a, phase))
            or not np.equal(phase, phase.astype(int)).all() or (np.diff(phase) < 0).any()
            or set(phase) != set(range(len(track.gates)))):
        raise ValueError('invalid canonical one-lap transition/phase contract')
    if int(round(horizons.max()*hz)) >= len(s):
        raise ValueError('episode shorter than requested temporal window')
    v = s[:, 7:10]; nv = ns[:, 7:10]
    speed = np.linalg.norm(v, axis=1)
    accel = uniform_filter1d((nv-v)*hz, size=max(1, round(.05*hz)), axis=0, mode='nearest')
    tangent = np.sum(accel*v, axis=1)/np.maximum(speed, .5)
    lateral = np.linalg.norm(accel-v/np.maximum(speed[:,None], .5)*tangent[:,None], axis=1)
    heading = (v[:,0]*accel[:,1]-v[:,1]*accel[:,0])/np.maximum(np.sum(v[:,:2]**2, axis=1), 1.)
    jerk = np.linalg.norm(np.gradient(accel, 1/hz, axis=0), axis=1)
    x = np.zeros((len(s), len(SCALES)))
    x[:,:10] = np.column_stack([speed, v[:,2], tangent, lateral, heading, accel[:,2], jerk,
                               a[:,0], np.linalg.norm(s[:,10:13],axis=1), np.linalg.norm(a[:,1:],axis=1)])
    for i, gate in enumerate(track.gates):
        mask = phase == i
        nxt = track.gates[min(i+1, len(track.gates)-1)]  # no final-lap wrap
        edge = nxt.position-gate.position
        x[mask,10] = v[mask] @ gate.normal / np.maximum(speed[mask], .5)
        x[mask,11:14] = [*gate.size, gate.up[2]]
        x[mask,14] = np.arccos(np.clip(gate.normal @ nxt.normal, -1, 1)) if i+1<len(track.gates) else 0
        x[mask,15:18] = [edge[2], np.linalg.norm(edge), mask.sum()/hz]
    summary = {name: dict(mean=float(np.mean(x[:,i])), p90=float(np.quantile(x[:,i],.9)))
               for i,name in enumerate(sum((list(v) for v in GROUPS.values()),[]))}
    summary.update(duration_s=len(s)/hz, braking_fraction=float(np.mean(tangent < -2)),
                   braking_distance_m=float(np.sum(speed[tangent < -2])/hz),
                   total_turn_rad=float(np.sum(np.abs(heading))/hz))
    anchors = np.concatenate([np.rint(np.linspace(np.flatnonzero(phase==i)[0],
                   np.flatnonzero(phase==i)[-1],bins)).astype(int) for i in range(len(track.gates))])
    windows = []
    for h in horizons:
        offsets = np.rint(np.array([0,.5,1])*h*hz).astype(int)
        starts = np.minimum(anchors, len(s)-1-offsets[-1])
        windows.append((x/SCALES)[starts[:,None]+offsets])
    return Demand(np.stack(windows,axis=1), phase[anchors].astype(int), summary)


def weighted_windows(demand, weights):
    if set(weights)-set(GROUPS) or not weights or any(not np.isfinite(w) or w<0 for w in weights.values()) or sum(weights.values())<=0:
        raise ValueError('nonnegative known group weights with positive sum required')
    blocks=[]; start=0
    for name, columns in GROUPS.items():
        block=demand.samples[...,start:start+len(columns)];start+=len(columns)
        if weights.get(name,0)>0:
            blocks.append(block.reshape(len(block),-1)*np.sqrt(weights[name]/sum(weights.values())/np.prod(block.shape[1:])))
    return ControlWindows(np.concatenate(blocks,axis=1), demand.gates)


def behavior_distance(a, b, weights):
    return ordered_distance(weighted_windows(a,weights), weighted_windows(b,weights))


def target_coverage(target, support, candidate, weights):
    """Per-target-phase nearest ordered-window support, no invented skill gain."""
    t=weighted_windows(target,weights).values
    old=np.concatenate([weighted_windows(s,weights).values for s in support])
    before=cdist(t,old).min(1)
    after=np.minimum(before,cdist(t,weighted_windows(candidate,weights).values).min(1))
    return dict(before=float(before.mean()), after=float(after.mean()),
                gain=float((before.mean()-after.mean())/max(before.mean(),1e-8)),
                per_gate_after=[float(after[target.gates==g].mean()) for g in np.unique(target.gates)])


def mixed_cost(geometry, behavior, geometry_weight):
    if not 0 <= geometry_weight <= 1 or not np.isfinite([geometry,behavior]).all():
        raise ValueError('finite costs and convex weight required')
    return float(geometry_weight*geometry+(1-geometry_weight)*behavior)


def wilson_lower(successes, episodes, z=1.96):
    if episodes<=0 or not 0<=successes<=episodes or z<=0:
        raise ValueError('invalid binomial evidence')
    p=successes/episodes
    return float((p+z*z/(2*episodes)-z*np.sqrt(p*(1-p)/episodes+z*z/(4*episodes**2)))/(1+z*z/episodes))


def rank_candidates(rows, geometry_weight, *, minimum_success_lower=.02, minimum_gate_fraction=.5):
    """Measured feasibility/readiness filter then configurable direction score.

    Does NOT promote candidates into mastered support. Only a subsequent policy
    evaluation can do that. Unknown policy or expert evidence is ineligible.
    """
    eligible=[]
    for r in rows:
        if not r.get('qualified') or r.get('episodes',0)<=0:continue
        lower=wilson_lower(r['successes'],r['episodes'])
        if lower < minimum_success_lower or r['gate_fraction'] < minimum_gate_fraction:continue
        eligible.append(dict(r, success_lower=lower, score=mixed_cost(r['geometry_cost'],r['behavior_cost'],geometry_weight)))
    return sorted(eligible,key=lambda r:(r['score'],r['name']))
