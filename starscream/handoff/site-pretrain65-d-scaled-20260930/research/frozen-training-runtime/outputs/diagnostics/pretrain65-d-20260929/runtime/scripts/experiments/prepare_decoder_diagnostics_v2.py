"""Frozen-data factorial decoder screen. Every scientific change is an arm."""
from copy import deepcopy
from pathlib import Path
import h5py
import numpy as np
import yaml
from starscream.course_model.training import atomic_json


def small_corpus(source,dest):
    if Path(dest).exists():
        with h5py.File(dest) as existing:
            if existing.attrs.get('complete',False):return
    rng=np.random.default_rng(9823)
    temp=Path(dest).with_suffix('.tmp.h5')
    with h5py.File(source) as f,h5py.File(temp,'w') as out:
        offsets=f['offsets'][:];counts=np.diff(offsets);split=f['split'][:];parents=f['parent_id'][:]
        ids=[]
        for s in (0,1):
            for n in (7,12):
                candidates=rng.permutation(np.flatnonzero((split==s)&(counts==n)));seen=set();chosen=[]
                for i in candidates:
                    if parents[i] not in seen:chosen.append(i);seen.add(parents[i])
                    if len(chosen)==64:break
                if s==1:chosen=candidates[:64].tolist()
                if not chosen or (s==0 and len(chosen)!=64):raise ValueError('insufficient memorization panel')
                ids.extend(chosen)
        ids=np.array(ids);gates=np.concatenate([f['gates'][offsets[i]:offsets[i+1]] for i in ids])
        for k,v in f.attrs.items():out.attrs[k]=v
        out.attrs['complete']=False
        out.create_dataset('gates',data=gates,compression='lzf')
        out.create_dataset('offsets',data=np.r_[0,np.cumsum(counts[ids])])
        for k,v in f.items():
            if k in ('gates','offsets'):continue
            if isinstance(v,h5py.Group):f.copy(k,out)
            else:out.create_dataset(k,data=v[:][ids],compression='lzf')
        out.attrs['source']=str(source);out.attrs['complete']=True
    temp.replace(dest)


def main():
    root=Path('outputs/course-model/decoder-v2');root.mkdir(parents=True,exist_ok=True)
    base=yaml.safe_load(Path('configs/exp/course_model/transfer_v7_direct_weakkl.yaml').read_text())
    base['training'].update(checkpoint_every_epochs=20)
    base['objective']['kl_reference_dim']=64
    base['model'].update(latent_tokens=1,sample_posterior=True)
    base['seed']=2026090802
    small=root/'memorization.h5';small_corpus(base['dataset'],small)
    arms=[]
    for name in ['memorize_ae64','vae64','ae64','ae512','ae64_large','ae64_large_scaled','ae_slots512','vae_slots512','vae_slots_relations','flow_slots_relations']:
        c=deepcopy(base);c.update(name='starscream-decoder-v2-'+name,output=str(root/name))
        if name.startswith(('ae','memorize')):
            c['model']['sample_posterior']=False;c['objective']['beta']=0
        if name=='memorize_ae64':
            c['dataset']='/workspace/'+str(small);c['training'].update(epochs=2000,checkpoint_every_epochs=200)
            c['evaluation'].update(every_epochs=200,geometry_samples=64)
            c['augmentation'].update(probability=0)
        if name=='ae512' or 'slots' in name:c['model']['latent']=512
        if name.startswith('ae64_large'):c['model'].update(width=640,heads=10,ff=2560)
        if name=='ae64_large_scaled':
            c['dataset']='/workspace/outputs/course-model/decoder-v2/scaled-courses.h5'
        if 'slots' in name:c['model']['latent_tokens']=8
        if 'relations' in name:c['objective']['relational_weights']=dict(edge=.25,gate_frame=.25,signed_turn=.1)
        if name.startswith('flow'):c['model']['decoder_type']='flow'
        path=Path(f'configs/exp/course_model/decoder_v2_{name}.yaml')
        if path.exists() and yaml.safe_load(path.read_text())!=c:raise ValueError(f'config changed {path}')
        path.write_text(yaml.safe_dump(c,sort_keys=False))
        arms.append(dict(name=name,config=str(path),output=c['output']))
    atomic_json(root/'matrix.json',dict(arms=arms,comparison='scratch, matched full20k/200epochs except small-panel memorization and declared parameter/data-scaled arm',
        fixed='gate17/context19, ordered inputs, parent splits, AdamW/BF16/Flash',
        interventions='posterior stochasticity+KL, latent capacity, backbone capacity, independent latent slots, relational objective, flow decoder',
        no_RL=True))
    print(arms)


if __name__=='__main__':main()
