"""Explicit physical-distance rays; no hidden geometry repairs."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from starscream.env.tracks import load_track
from starscream.env.procedural_tracks import save_track_yaml,geometry_fingerprint
from starscream.course_model.schema import static_reasons,pack
from starscream.course_model.directed import curve_distance
from starscream.course_model.training import atomic_json

ROOT=Path('outputs/course-model/v4/rays')

def main():
    if ROOT.exists():raise FileExistsError(ROOT)
    source=load_track('swift_champion_2022_exact');target=load_track('multigp_cdra_2026_reconstructed')
    tg,_,_=pack(target); rows=[]
    for method,name in [('vae','anchored_r0.5'),('classical','classical_r0.5')]:
        base=load_track(f'outputs/course-model/v3/matched-panel/tracks/{name}.yaml')
        delta=np.array([b.position-a.position for a,b in zip(source.gates,base.gates)])
        for scale in (1.,4.,12.,24.,40.):
            title=f'v4_{method}_x{int(scale)}'
            gates=tuple(replace(g,position=g.position+scale*d) for g,d in zip(source.gates,delta))
            track=replace(source,name=title,gates=gates)
            reasons=static_reasons(track)
            if any(np.any(g.position<source.bounds[:,0]) or np.any(g.position>source.bounds[:,1]) for g in gates):
                reasons.append('center_bounds')
            path=save_track_yaml(track,ROOT/'tracks'/f'{title}.yaml').resolve()
            pg,_,_=pack(track)
            rows.append(dict(name=title,path=str(path),method=method,scale=scale,
                static_valid=not reasons,reasons=reasons,
                maximum_displacement_m=float(np.linalg.norm(scale*delta,axis=1).max()),
                total_displacement_m=float(np.linalg.norm(scale*delta)),
                target_distance_m=float(curve_distance(torch.tensor(pg[None,:,:3]),torch.tensor(tg[None,:,:3]))),
                geometry_fingerprint=geometry_fingerprint(track),valid=not reasons,split='train',family=method))
    atomic_json(ROOT/'all.json',dict(records=rows))
    atomic_json(ROOT/'screen.json',dict(records=[r for r in rows if r['static_valid']]))
    print([(r['name'],r['static_valid'],round(r['maximum_displacement_m'],2)) for r in rows])

if __name__=='__main__':main()
