"""Dataset/contract preflight and six real two-epoch smoke runs."""
import json
from pathlib import Path
import time
import h5py
import numpy as np
import yaml
from starscream.course_model.training import train,atomic_json
from starscream.course_model.model import CourseVAE,VAEConfig
from starscream.course_model.schema import unpack,static_reasons


def main():
    path='outputs/course-model/v5/independent20k/courses.h5'
    with h5py.File(path,'r') as f:
        assert f.attrs['complete'] and len(f['context'])==20000
        p,s=f['parent_id'][:],f['split'][:]
        assert not(set(p[s==0])&set(p[s==1]))
        assert len(set(f['fingerprint'][:]))==20000
        assert (f['reference'][:]==-1).all()
        assert (f['feasibility'][:]==-1).all()
        assert len(np.unique(f['context'][:,13:],axis=0))==1
        off=f['offsets'][:];g=f['gates'][:];c=f['context'][:]
        bad=[]
        for i in range(len(c)):
            reasons=static_reasons(unpack(g[off[i]:off[i+1]],c[i]))
            if reasons:bad.append((i,reasons))
        assert not bad,bad[:5]
    report=dict(dataset=path,static_revalidated=20000,parent_split_disjoint=True,
                unique_fingerprints=20000,known_feasible=0,smoke_runs=[])
    params=[]
    for arm in ('old_control','new_control','bottleneck','geometry','metric','generated'):
        cfg=yaml.safe_load(Path(f'configs/exp/course_model/manippo_vae_v5_{arm}.yaml').read_text())
        assert 'initialize_from' not in cfg and cfg['training']['epochs']==200
        params.append(sum(x.numel() for x in CourseVAE(VAEConfig(**cfg['model'])).parameters()))
        cfg['training']['epochs']=2;cfg['evaluation']['every_epochs']=1;cfg['wandb']['enabled']=False
        output=Path('outputs/course-model/v5')/('smoke_'+arm)
        started=time.perf_counter();train(cfg,output)
        records=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
        report['smoke_runs'].append(dict(arm=arm,elapsed_seconds=time.perf_counter()-started,
            second_epoch_seconds=records[-1]['train_seconds'],
            second_epoch_steps=records[-1]['step']-records[-2]['step'],
            validation_seconds=records[-1]['validation']['seconds']))
    assert len(set(params))==1,params
    report['parameters']=params[0]
    atomic_json(Path('outputs/audits/manippo-vae-v5-preflight.json'),report)
    print('PREFLIGHT',json.dumps(report),flush=True)


if __name__=='__main__':main()
