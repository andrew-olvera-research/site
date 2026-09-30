"""Physical behavior coverage for bounded course pools; not a transfer predictor."""
import numpy as np
from scipy.ndimage import uniform_filter1d


def behavior_summary(trace):
    """Measures executed motion, never gate-relative position derivatives.

    Banking is body tilt during horizontal turning, not Euler roll (which is
    ambiguous near inverted flight). Fractions use physical-time samples.
    """
    s=np.asarray(trace['states'],float);ns=np.asarray(trace['next_states'],float)
    hz=float(trace['control_hz'])
    if s.ndim!=2 or s.shape[1]<13 or ns.shape!=s.shape or len(s)<2 or hz<=0:
        raise ValueError('invalid trajectory')
    if not np.isfinite(s).all() or not np.isfinite(ns).all():raise ValueError('nonfinite state')
    v=s[:,7:10];speed=np.linalg.norm(v,axis=1)
    acc=uniform_filter1d((ns[:,7:10]-v)*hz,max(1,round(.05*hz)),axis=0,mode='nearest')
    tang=np.sum(acc*v,axis=1)/np.maximum(speed,.5)
    heading=(v[:,0]*acc[:,1]-v[:,1]*acc[:,0])/np.maximum((v[:,:2]**2).sum(1),1.)
    q=s[:,3:7];norm=np.linalg.norm(q,axis=1)
    if (norm<1e-8).any():raise ValueError('invalid quaternion')
    q=q/norm[:,None];up_z=1-2*(q[:,1]**2+q[:,2]**2)
    tilt=np.degrees(np.arccos(np.clip(up_z,-1,1)))
    moving=speed>3
    events={'braking':moving&(tang < -4),
        'banking':moving&(tilt>25)&(tilt<85)&(np.abs(heading)>.3),
        'hard_turn':moving&(np.abs(heading)>1.),
        'inverted':up_z<0,'vertical':np.abs(v[:,2])>3}
    result={f'{k}_fraction':float(mask.mean()) for k,mask in events.items()}
    result.update(duration_s=len(s)/hz,speed_mean_mps=float(speed.mean()),
        speed_p90_mps=float(np.quantile(speed,.9)),braking_p90_mps2=float(np.quantile(np.maximum(-tang,0),.9)),
        heading_rate_p90_rads=float(np.quantile(np.abs(heading),.9)),tilt_p90_deg=float(np.quantile(tilt,.9)),
        body_rate_p90_rads=float(np.quantile(np.linalg.norm(s[:,10:13],axis=1),.9)),
        thrust_p90_mps2=float(np.quantile(np.asarray(trace['actions'])[:,0],.9)))
    # Ordered gate segments retain information that course means erase.
    phase=np.asarray(trace['phase'])
    changes=np.diff(np.r_[False,events['inverted'],False].astype(int))
    runs=np.flatnonzero(changes==-1)-np.flatnonzero(changes==1)
    result['inverted_longest_s']=float(max(runs,default=0)/hz)
    result['inverted_after_first_fraction']=float(events['inverted'][phase>0].mean()) if (phase>0).any() else 0.
    result['per_gate']=[dict(gate=int(g),samples=int((phase==g).sum()),
        **{f'{k}_fraction':float(mask[phase==g].mean()) for k,mask in events.items()}) for g in np.unique(phase)]
    return result


def coverage_vector(summary):
    keys=['speed_mean_mps','braking_p90_mps2','heading_rate_p90_rads',
          'banking_fraction','inverted_fraction','vertical_fraction']
    return np.array([summary[k] for k in keys])/np.array([20,15,3,1,1,1])


def diverse_subset(rows,limit):
    """Train-only farthest-point coverage; fixed physical scales, no eval fit."""
    if not rows:return []
    x=np.stack([coverage_vector(r['behavior']) for r in rows])
    selected=[int(np.argmin(np.linalg.norm(x-np.median(x,axis=0),axis=1)))]
    while len(selected)<min(limit,len(rows)):
        d=np.min(np.linalg.norm(x[:,None]-x[selected][None],axis=-1),axis=1)
        d[selected]=-1;selected.append(int(np.argmax(d)))
    return [rows[i] for i in selected]
