"""Prepare six from-scratch, fixed-exposure component ablations."""
from pathlib import Path
from copy import deepcopy
import yaml
from starscream.course_model.training import atomic_json


def main():
    base=yaml.safe_load(Path('configs/exp/course_model/vae_pretrain_v1_200total.yaml').read_text())
    base.pop('initialize_from',None)
    base['seed']=2026090802
    base['training'].update(epochs=200,lr=3e-4,min_lr=1e-5,warmup_steps=100)
    base['objective']['kl_warmup_steps']=500
    base['evaluation'].update(every_epochs=20,geometry_samples=256)
    rows=[]
    for name in ('old_control','old_bottleneck','new_control','bottleneck','geometry','metric','generated'):
        cfg=deepcopy(base);cfg['name']='starscream-manippo-vae-v5-'+name
        cfg['output']='outputs/course-model/v5/'+name
        if name not in ('old_control','old_bottleneck'):cfg['dataset']='/workspace/outputs/course-model/v5/independent20k/courses.h5'
        cfg['model']['decoder_context']=name in ('old_control','new_control')
        if name in ('geometry','metric','generated'):
            cfg['objective']['geometry_weights']=dict(transverse=.05,worst_gate=.01,cyclic=.25,floor=.05)
        if name=='metric':cfg['objective']['manippo']=dict(latent_metric_weight=.1)
        if name=='generated':cfg['objective']['manippo']=dict(generated_constraint_weight=.1,generated_radius=.25)
        path=Path(f'configs/exp/course_model/manippo_vae_v5_{name}.yaml')
        if path.exists():
            if yaml.safe_load(path.read_text())!=cfg:raise ValueError(f'existing config differs: {path}')
        else:path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        rows.append(dict(name=name,config=str(path),output=cfg['output']))
    atomic_json(Path('outputs/course-model/v5/experiments.json'),dict(arms=rows,
        scope='Known count, no decoder course-context in bottleneck arms. Not full cross-count ManiPPO.',
        comparison='same initial seed, architecture/parameter count, 200 epochs and 20k courses; differing count histograms can change optimizer steps'))
    print(rows)


if __name__=='__main__':main()
