"""Publish measured cell witnesses and a course atlas from the frozen manifest."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.audits.prepare_v6222 import OUT,read
from starscream.course_model.training import atomic_json
from starscream.env.tracks import load_track


def main():
    records=read(OUT/'manifest.json')['records']
    cells=[]
    for r in records:
        if not r['name'].startswith('v6222_'):continue
        ts=r['geometry']['transitions'];g=r['grammar']
        if g.startswith('utt_'):
            t=ts[1]
            assert 25<=t['incoming_m']<=42.01 and t['gate_center_height_m']<=.91
            assert 1.49<=t['width_m']<=1.61
            if g=='utt_straight':assert t['turn_deg']<25
            elif g=='utt_brake90':assert 75<t['turn_deg']<105
            else:assert t['turn_deg']>=135
        elif g=='a2rl_chain':
            assert ts[1]['height_change_m']< -1.4 and ts[2]['height_change_m']< -1.4
            assert ts[3]['height_change_m']>2.4
        elif g=='a2rl_drop':assert ts[1]['height_change_m']< -1.4 and ts[1]['turn_deg']>90
        elif g=='a2rl_exit':assert ts[2]['height_change_m']< -1.4 and ts[1]['signed_horizontal_turn_deg']*ts[2]['signed_horizontal_turn_deg']<0
        cells.append(dict(name=r['name'],split=r['split'],family=r['family'],grammar=g,
            gate_count=r['gate_count'],command=r['qualified_speed_mps'],
            motif=[{k:t[k] for k in ('gate','incoming_m','outgoing_m','turn_deg','signed_horizontal_turn_deg',
                'height_change_m','gate_center_height_m','width_m','height_m','incoming_alignment')} for t in ts[:5]],
            qualification=r['qualification']['evidence_path']))
    train=[r for r in records if r['split']=='train']
    atomic_json(OUT/'cell-plan.json',dict(status='MEASURED_AND_QUALIFIED',
        slots=dict(retained_foundation=30,utt_long_low=6,a2rl_ordered=6,composition=3),
        axes=dict(approach_edges_m=[5,12,20,30,40],height_edges_m=[.8,1.2,2.5],
                  aperture_edges_m=[1.2,1.65,2.2],signed_vertical_magnitude_edges_m=[.75,2,3],chain_horizons=[2,3,4]),
        instruction='Use continuous motif witnesses below, not Cartesian coverage counts, to interpret target support.',
        witnesses=cells,train_gate_counts=dict(Counter(r['gate_count'] for r in train))))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(9,5,figsize=(17,25),layout='constrained')
    for ax,r in zip(axes.flat,train):
        t=load_track(r['path']);p=np.asarray([g.position for g in t.gates])
        ax.plot(p[:,0],p[:,1],color='#607487',lw=.8)
        ax.plot(p[[-1,0],0],p[[-1,0],1],color='#607487',lw=.8,ls='--')
        ax.scatter(p[:,0],p[:,1],c=p[:,2],cmap='viridis',s=15,vmin=0,vmax=7)
        for i,point in enumerate(p):ax.annotate(str(i),point[:2],fontsize=5)
        ax.set_title(r['family']+f" · {len(p)} gates",fontsize=8)
        ax.set_aspect('equal');ax.tick_params(labelsize=5);ax.grid(alpha=.15)
    fig.suptitle('v6.22.2 · pretrain45 · 30 retained + 15 qualified support courses\nColour denotes height; dashed closure is not a multi-lap qualification claim',fontsize=14)
    fig.savefig(OUT/'pretrain45-atlas.png',dpi=120)
    fig.savefig(OUT/'pretrain45-atlas.pdf')
    plt.close(fig)
    print('PASS: measured target cells; atlas and cell-plan.json written')


if __name__=='__main__':main()
