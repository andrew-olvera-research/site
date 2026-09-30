#!/usr/bin/env python3
"""Export a captured simulator episode in the site's metre/second/radian schema."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from starscream.env.tracks import load_track


def demo_payload(track, states, times, passed, *, name=None, control_hz=130):
    states=np.asarray(states,dtype=np.float64); times=np.asarray(times,dtype=np.float64); passed=np.asarray(passed,dtype=np.int64)
    if states.ndim!=2 or states.shape[1]<7 or len(states)<2 or times.shape!=(len(states),) or passed.shape!=(len(states),):
        raise ValueError('states [T,>=7], times [T], and passed [T] must align')
    if not np.isfinite(states[:,:7]).all() or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('episode contains nonfinite state or nonincreasing timestamps')
    dt=1/float(control_hz)
    if not math.isfinite(dt) or dt<=0: raise ValueError('control Hz must be positive')
    intervals=np.diff(times)
    if not np.allclose(intervals,dt,rtol=2e-3,atol=2e-6):
        raise ValueError(f'episode clock is not uniform at {control_hz} Hz: median dt={np.median(intervals)}')
    if np.any(np.diff(passed)<0): raise ValueError('passed-gate count must be monotonic')
    gates=[]
    for gate in track.gates:
        normal=np.asarray(gate.normal,float)
        gates.append(dict(pos=[round(float(x),4) for x in gate.position],
            yaw=round(math.atan2(float(normal[1]),float(normal[0])),6),
            size=round(float(min(gate.size)),4)))
    quaternions=[]
    for state in states:
        quat=np.asarray(state[3:7],float); length=np.linalg.norm(quat)
        if length<1e-8: raise ValueError('zero-norm trajectory quaternion')
        quat/=length
        if quaternions and np.dot(quat,quaternions[-1])<0: quat=-quat
        quaternions.append([round(float(v),7) for v in quat])
    return dict(name=name or track.name,gates=gates,trajectory=dict(dt=dt,
        pos=[[round(float(v),4) for v in state[:3]] for state in states],
        quat=quaternions,passed=passed.astype(int).tolist()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track',required=True);p.add_argument('--states',type=Path,required=True)
    p.add_argument('--times',type=Path,required=True);p.add_argument('--passed',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--name')
    p.add_argument('--control-hz',type=float,default=130)
    a=p.parse_args();track=load_track(a.track)
    payload=demo_payload(track,np.load(a.states),np.load(a.times),np.load(a.passed),name=a.name,control_hz=a.control_hz)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream: json.dump(payload,stream,separators=(',',':'))
    print(json.dumps(dict(path=str(a.output),frames=len(payload['trajectory']['pos']),
        dt=payload['trajectory']['dt'],hz=1/payload['trajectory']['dt'],
        duration=(len(payload['trajectory']['pos'])-1)*payload['trajectory']['dt'])))


if __name__=='__main__': main()
