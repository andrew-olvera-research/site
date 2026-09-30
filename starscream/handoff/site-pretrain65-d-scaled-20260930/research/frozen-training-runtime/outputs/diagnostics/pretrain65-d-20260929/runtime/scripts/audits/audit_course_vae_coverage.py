"""Independent human-course support with explicit shape/length/count distances."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from starscream.env.tracks import load_track
from starscream.env.procedural_tracks import save_track_yaml
from starscream.course_model.schema import pack,unpack
from scripts.experiments.build_course_vae_dataset import REFS,describe,signature


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);args=p.parse_args()
    report={'selection':'xyz RMSE metres + 0.15*absolute polyline length error metres + 0.5*gate count error',
        'interpretation':'geometry support only; neither latent distance nor a transfer predictor',
        'tracks':{}}
    with h5py.File(args.dataset,'r') as f:
        desc=f['descriptors'][:];sig=f['shape_signature'][:];orig=f['reference'][:]
        # Additional shipped analytic courses are not reference parents. Their
        # metadata is not treated as evidence of real competition provenance.
        for name in (*REFS,'kidney','figure8','big_s'):
            track=load_track(Path('starscream/assets/tracks')/(name+'.yaml'))
            g,c,_=pack(track);d=describe(g,track.loop);s=signature(g,track.loop)
            xyz=np.sqrt(((sig-s)**2).mean(1))
            length=abs(desc[:,1]-d[1]);count=abs(desc[:,0]-len(g))
            score=xyz+.15*length+.5*count
            report['tracks'][name]={}
            for label,select in [('independent',orig<0),('reference_derived',orig>=0)]:
                ids=np.flatnonzero(select);best=int(ids[np.argmin(score[ids])])
                a,b=f['offsets'][best:best+2];ng=f['gates'][a:b];nc=f['context'][best]
                # Normalize by aperture for the footprint-relative closeness
                # criterion; this does not imply equal flight behavior.
                close=(xyz<np.exp(g[:,9:11]).mean())&(length/d[1]<.15)&(count==0)
                report['tracks'][name][label]={'index':best,'gate_count':len(ng),
                    'xyz_rmse_m':float(xyz[best]),'length_error_fraction':float(length[best]/d[1]),
                    'within_aperture_rmse_and_15pct_length_same_count':int((close&select).sum())}
                save_track_yaml(unpack(ng,nc,f'{name}_{label}'),args.dataset.parent/'nearest_joint'/f'{name}_{label}.yaml')
    (args.dataset.parent/'coverage_joint.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
