"""Measured corpus coverage, separate from grammar names or policy selection.

These coordinates diagnose support; they are not a learned transfer predictor.
The turn at gate i uses incoming and outgoing *center chords*. It is not the
executed MPCC turn, particularly for wrong-side entry and stacked gates.
"""
from collections import Counter
from dataclasses import replace
import numpy as np
from ..tracks import forward_up_quaternion
from .transition_corpus import shape_distance, transition_metrics

TURN_EDGES = [0, 30, 60, 90, 120, 150, 180]


def geometry_record(track):
    p=np.asarray([g.position for g in track.gates],float)
    a=p-np.roll(p,1,axis=0);b=np.roll(p,-1,axis=0)-p
    signed=np.degrees(np.arctan2(a[:,0]*b[:,1]-a[:,1]*b[:,0],(a[:,:2]*b[:,:2]).sum(1)))
    transitions=transition_metrics(track)
    symbols=[]
    for i,r in enumerate(transitions):
        r.update(signed_horizontal_turn_deg=float(signed[i]),
            width_m=float(track.gates[i].size[0]),height_m=float(track.gates[i].size[1]),
            gate_center_height_m=float(p[i,2]),
            gate_normal_change_deg=float(np.degrees(np.arccos(np.clip(
                track.gates[i].normal@track.gates[(i+1)%len(p)].normal,-1,1)))))
        # Quantized ORDERED transition witnesses, not unordered family labels.
        turn=int(np.floor((np.clip(signed[i],-179.99,179.99)+180)/30))
        rise=-1 if a[i,2]<-.75 else (1 if a[i,2]>.75 else 0)
        symbols.append(f'{turn}:{rise}:{int(r["preceding_gate_on_exit_side"])}')
    n=len(p)
    alternating=(signed*np.roll(signed,-1)<0)&(abs(signed)>20)&(abs(np.roll(signed,-1))>20)
    return dict(name=track.name,gate_count=n,transitions=transitions,
        left_turns=int((signed>20).sum()),right_turns=int((signed< -20).sum()),
        alternating_pairs=int(alternating.sum()),
        # Last->first is described geometrically but requires its own multi-lap
        # dynamic qualification before claiming the boundary is raceable.
        bigrams=[','.join(symbols[(i+j)%n] for j in range(2)) for i in range(n)],
        trigrams=[','.join(symbols[(i+j)%n] for j in range(3)) for i in range(n)])


def aggregate_geometry(records):
    ts=[t for r in records for t in r['transitions']]
    if not ts:return dict(courses=0,turn_bin_counts=[0]*6,left_turns=0,right_turns=0,
        alternating_pairs=0,bigrams={},trigrams={})
    return dict(courses=len(records),gate_counts=dict(Counter(r['gate_count'] for r in records)),
        turn_bin_edges_deg=TURN_EDGES,
        turn_bin_counts=np.histogram([t['turn_deg'] for t in ts],TURN_EDGES)[0].tolist(),
        left_turns=sum(r['left_turns'] for r in records),right_turns=sum(r['right_turns'] for r in records),
        alternating_pairs=sum(r['alternating_pairs'] for r in records),
        wrong_side_entries=sum(t['preceding_gate_on_exit_side'] for t in ts),
        normal_reversals=sum(t['gate_normal_change_deg']>150 for t in ts),
        narrow_gates=sum(t['width_m']<=1.65 for t in ts),
        rises_over_2m=sum(t['height_change_m']>2 for t in ts),
        drops_over_2m=sum(t['height_change_m']< -2 for t in ts),
        ranges={k:[min(t[k] for t in ts),max(t[k] for t in ts)] for k in
            ('incoming_m','height_change_m','width_m','gate_center_height_m','incoming_alignment')},
        bigrams=dict(Counter(s for r in records for s in r['bigrams'])),
        trigrams=dict(Counter(s for r in records for s in r['trigrams'])))


def clone_distance(a,b):
    """Leakage distance also removes reflection, unlike behavior distance.

    A mirrored course is not an independent held-out geometry by itself.
    Only centers are used by shape_distance; frames are nevertheless proper.
    """
    direct=shape_distance(a,b)
    if direct is None:return None
    mirrored=replace(a,gates=tuple(replace(g,position=g.position*[1,-1,1],
        quaternion_wxyz=forward_up_quaternion(g.physical_normal*[1,-1,1])) for g in a.gates))
    return min(direct,shape_distance(mirrored,b))


def coverage_gaps(summary):
    if not summary['courses']:return ['empty_split']
    missing=[f'turn_bin_{TURN_EDGES[i]}_{TURN_EDGES[i+1]}' for i,n in enumerate(summary['turn_bin_counts']) if n==0]
    total=summary['left_turns']+summary['right_turns']
    if total==0 or not .25<=summary['right_turns']/total<=.75:missing.append('turn_handedness_imbalance')
    for key in ('alternating_pairs','wrong_side_entries','normal_reversals','narrow_gates','rises_over_2m','drops_over_2m'):
        if summary.get(key,0)==0:missing.append(key)
    return missing
