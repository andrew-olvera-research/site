"""Summarize probe_gate_crossing_geometry output: miss geometry and pre-crossing deviation growth."""
import json, sys, numpy as np
from pathlib import Path
for name in sys.argv[1:]:
    d=Path(name); c=json.load(open(d/'crossings.json')); tr=np.load(d/'traces.npz')
    X=[x for x in c['crossings'] if x['gate']>0]
    first=[x for x in X if x['prior_misses']==0]
    # first attempt at each gate: the first crossing record per (slot,gate)
    seen=set(); fa=[]
    for x in X:
        k=(x['slot'],x['gate'])
        if k in seen: continue
        seen.add(k); fa.append(x)
    norm=lambda x: max(abs(x['hit_y'])/x['half_width'],abs(x['hit_z'])/x['half_height'])
    rnorm=lambda x: max(abs(x['ref_y'])/x['half_width'],abs(x['ref_z'])/x['half_height'])
    dev=lambda x: np.hypot(x['hit_y']-x['ref_y'],x['hit_z']-x['ref_z'])
    P=[x for x in fa if x['kind']=='pass']; M=[x for x in fa if x['kind']=='miss']
    print(f"== {d.name}: first attempts {len(fa)}  pass {len(P)}  miss {len(M)}  planned-outside {len(fa)-len(P)-len(M)}  hazard {len(M)/max(1,len(P)+len(M)):.3f}")
    for lab,S in [('pass',P),('miss',M)]:
        if not S: continue
        n=np.array([norm(x) for x in S]); r=np.array([rnorm(x) for x in S]); dv=np.array([dev(x) for x in S]); sp=np.array([x['speed'] for x in S])
        print(f"  {lab}: hit/half-size q10/50/90 {np.round(np.quantile(n,[.1,.5,.9]),2)}  ref line hit/half-size median {np.median(r):.2f}  |hit-ref| m q50/90 {np.round(np.quantile(dv,[.5,.9]),2)}  speed median {np.median(sp):.1f}")
    if M:
        n=np.array([norm(x) for x in M])
        print(f"  misses by outside margin: <1.25x {np.mean(n<1.25):.2f}  1.25-2x {np.mean((n>=1.25)&(n<2)):.2f}  >=2x {np.mean(n>=2):.2f}")
    # pre-crossing contour distance to the reference line (policy's own trace)
    rows={'pass':[], 'miss':[]}
    for x in fa:
        if x['kind'] not in rows: continue
        t=tr[x['slot'].replace('/','_')]; i=int(x['step'])-1
        vals=[]
        for lag in (130,65,32,13,0):
            j=i-lag
            vals.append(t[j,7] if j>=0 and t[j,10]==x['gate'] else np.nan)
        rows[x['kind']].append(vals)
    for k,v in rows.items():
        v=np.array(v)
        if len(v): print(f"  {k}: median line distance (m) at t-1.0/-0.5/-0.25/-0.1/0 s: {np.round(np.nanmedian(v,0),2)}  q90 {np.round(np.nanquantile(v,.9,0),2)}")
    # pass margin usage: fraction of passes that used >70% of half-size
    if P: print(f"  passes using >70% of half-size: {np.mean(np.array([norm(x) for x in P])>.7):.2f}")
    ep=c['episodes']; print(f"  episodes success {np.mean([e['success'] for e in ep]):.2f} clean {np.mean([e['success'] and e['misses']==0 for e in ep]):.2f}")
