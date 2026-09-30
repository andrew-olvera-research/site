"""Materialize small training-only edits, qualify and freeze the v6.19 pool."""
from __future__ import annotations
import argparse,json,sys
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.audits.prepare_v619_pool import SPEC,digest,qualify_one,strata,screen_geometry
from starscream.course_model.training import atomic_json
from starscream.env.tracks import load_track
from starscream.env.procedural_tracks import save_track_yaml,geometry_fingerprint
from starscream.course_navigation.course_edits import EditBudget,deform,codec_delta

def physical_normalizer(values):
    from starscream.privileged_racing import FeatureNormalizer
    fitted=FeatureNormalizer.fit(values)
    floor=np.full(103,1e-4,np.float32)
    # Geometry channels constant in a yaw-only pool must not get a 1e-4
    # denominator. Fixed physical floors are independent of validation data.
    for j in range(6):
        start=19+13*j
        floor[start:start+3]=1.0  # metres
        floor[start+3:start+9]=1.0  # rotation matrix components
        floor[start+9:start+11]=.25  # aperture metres
        floor[start+11:start+13]=1.0  # binary flags
    floor[97:101]=.25  # normalized previous CTBR
    floor[101]=.02;floor[102]=1.0  # age seconds / validity
    return FeatureNormalizer(fitted.mean,np.maximum(fitted.std,floor))

def augment(cfg):
    from scipy.interpolate import CubicSpline
    from starscream.course_model.flow import build_course_model
    from starscream.course_model.model import VAEConfig
    from starscream.course_model.schema import pack
    root=Path(cfg['output']);dest=root/'augmentation_candidates.json'
    if dest.exists():return
    ledger=json.loads((root/'parent_qualification.json').read_text())
    assert not ledger['missing'],ledger['missing']
    parents=ledger['selected'];torch.set_num_threads(2)
    saved=torch.load(cfg['codec'],map_location='cpu',weights_only=False)
    model=build_course_model(VAEConfig(**saved['config'])).eval()
    model.load_state_dict(saved['model']);del saved
    model.cuda();rng=np.random.default_rng(cfg['seed']+2200000)
    budget=EditBudget(cfg['augmentation_position_m'],cfg['augmentation_rotation_deg'],1.,0)
    rows=[];rejected=[]
    for family,*_ in strata():
        candidates=sorted([r for r in parents if r['split']=='train' and r['family']==family],key=lambda r:r['parent_id'])
        for kind in ('spline','vae'):
            slot=f'{family}_{kind}'
            for attempt in range(8):
                row=candidates[attempt%len(candidates)];parent=load_track(row['path']);n=len(parent.gates)
                name=f'v619_{slot}_{attempt}';seed=int(rng.integers(1,2**30))
                random=np.random.default_rng(seed)
                if kind=='spline':
                    # Smooth periodic displacement, locked at gate zero. This
                    # is a local spline edit, not a second independent parent.
                    knots=np.linspace(0,1,5);offset=random.normal(size=(5,3));offset[-1]=offset[0]
                    delta=CubicSpline(knots,offset,bc_type='periodic')(np.arange(n)/n)
                    delta-=delta[0];delta*=.9*budget.position_m/max(np.linalg.norm(delta,axis=1).max(),1e-8)
                    edit=deform(parent,delta,name=name,budget=budget)
                else:
                    g,c,transform=pack(parent)
                    with torch.no_grad():
                        gates=torch.tensor(g[None],device='cuda');context=torch.tensor(c[None],device='cuda')
                        z,_=model.encode(gates,torch.ones((1,n),device='cuda',dtype=torch.bool),context,_uniform=True)
                        dz=torch.tensor(random.normal(size=z.shape),device='cuda',dtype=z.dtype)
                        dz=dz/dz.norm().clamp_min(1e-8)
                        raw,_=model.decode(torch.cat((z,z+dz),0),torch.full((2,),n,device='cuda',dtype=torch.long),context.expand(2,-1),_length=n,steps=64)
                    a,b=raw.cpu().numpy();frame=np.array(transform['frame'])
                    large=codec_delta(parent,a,b,frame,name=name,budget=budget)
                    m=large.metrics
                    strength=.9*min(1.,budget.position_m/max(m['preserved_position_max_m'],1e-8),budget.rotation_deg/max(m['preserved_rotation_max_deg'],1e-8),budget.curve_rms_m/max(m['curve_rms_m'],1e-8))
                    edit=codec_delta(parent,a,b,frame,strength=strength,name=name,budget=budget)
                reasons=edit.reasons+screen_geometry(edit.track)
                if edit.metrics['preserved_position_max_m']<.005:reasons+=['negligible_edit']
                if reasons:rejected.append(dict(name=name,reasons=reasons));continue
                path=save_track_yaml(edit.track,root/'candidates'/f'{name}.yaml').resolve()
                rows.append(dict(name=name,path=str(path),family=family,split='train',parent_id=row['parent_id'],
                    augmentation_slot=slot,source_parent=row['path'],seed=seed,origin=f'bounded_{kind}_edit',
                    fingerprint=geometry_fingerprint(edit.track),edit_metrics=edit.metrics))
            print('prepared',slot,flush=True)
    atomic_json(dest,dict(records=rows,rejections=rejected,codec_sha256=digest(cfg['codec'])))
    del model;torch.cuda.empty_cache()

def qualify_aug(cfg):
    root=Path(cfg['output']);rows=json.loads((root/'augmentation_candidates.json').read_text())['records']
    slots={r['augmentation_slot'] for r in rows};assert len(slots)==16
    chosen={};results=[]
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as pool:
        for attempt in range(8):
            todo=[]
            for slot in sorted(slots-chosen.keys()):
                rr=[r for r in rows if r['augmentation_slot']==slot]
                if attempt<len(rr):todo.append(rr[attempt])
            for row,result in zip(todo,pool.map(qualify_one,[(cfg,r) for r in todo])):
                results.append(result)
                if result['qualified']:chosen[row['augmentation_slot']]=dict(row,qualification=result)
                atomic_json(root/'augmentation_qualification.json',dict(records=results,selected=list(chosen.values()),missing=sorted(slots-chosen.keys())))
                print('augmentation',row['name'],result['qualified'],len(chosen),'/16',flush=True)
            if len(chosen)==16:break
    assert len(chosen)==16,sorted(slots-chosen.keys())

def freeze(cfg):
    root=Path(cfg['output']);rows=[]
    for key in ['parent','augmentation']:
        a=json.loads((root/f'{key}_qualification.json').read_text());assert not a['missing'];rows+=a['selected']
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
    assert (len(train),len(val))==(32,8)
    assert not {r['parent_id'] for r in train}&{r['parent_id'] for r in val}
    assert len({r['fingerprint'] for r in rows})==40
    for r in rows:
        r.update(qualified=True,qualified_speed_mps=cfg['speed_command'],geometry_fingerprint=r['fingerprint'])
    proxies=[]
    for path in cfg['proxy_tracks']:
        t=load_track(path)
        proxies.append(dict(name=t.name,path=path,family='unverified_local_proxy',split='report',
            qualified=False,qualified_speed_mps=cfg['speed_command'],origin='local_generated_proxy_not_official',fingerprint=geometry_fingerprint(t),geometry_fingerprint=geometry_fingerprint(t)))
    assert not {r['fingerprint'] for r in rows}&{r['fingerprint'] for r in proxies}
    atomic_json(root/'manifest.json',dict(schema='starscream-procedural-track-manifest-v1',
        scope='Green-inspired; local report proxies are not verified paper benchmark geometry',records=rows+proxies))
    # Balanced training-only feature/auxiliary statistics, not pretrained weights.
    from starscream.privileged_racing import FeatureNormalizer,TASK_DIM
    features=[];deltas=[]
    for r in train:
        for domain in ['nominal','randomized','dart']:
            with np.load(root/'qualification'/r['name']/domain/'episode-000.npz') as t:
                x=t['features'];phase=t['phase'];valid=(phase[1:]==phase[:-1])&(t['solver_status'][:-1]==0)
                assert x.shape[1]==103 and np.isfinite(x).all()
                features.append(x[np.linspace(0,len(x)-1,256).astype(int)])
                d=np.diff(x[:,:TASK_DIM],axis=0)[valid]*(130./90.)
                assert len(d)>0
                deltas.append(d[np.linspace(0,len(d)-1,256).astype(int)])
    x=np.concatenate(features);d=np.concatenate(deltas)
    stats=dict(normalizer=physical_normalizer(x).state_dict(),dynamics_target_mean=d.mean(0).astype(np.float32),
        dynamics_target_std=np.maximum(d.std(0),1e-4).astype(np.float32),observation_contract='starscream_route_v1',route_gate_count=6,
        provenance='equal train-course/cohort traces; physical geometry std floors; same-frame task deltas at 90Hz reference; no validation/report data')
    torch.save(stats,root/'normalization.pt')
    atomic_json(root/'manifest.lock.json',{str(p):digest(p) for p in [root/'manifest.json',root/'normalization.pt',SPEC]+[Path(r['path']) for r in rows+proxies]})
    print('frozen 32 train / 8 validation / 4 unverified report proxies',flush=True)

def report(cfg):
    root=Path(cfg['output'])
    rows=[dict(name=load_track(p).name,path=p) for p in cfg['proxy_tracks']]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as pool:
        results=list(pool.map(qualify_one,[(cfg,r) for r in rows]))
    atomic_json(root/'proxy_expert_report.json',dict(scope='all four local proxies; no selection or official benchmark claim',records=results))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['augment','qualify','freeze','report']);a=ap.parse_args()
    {'augment':augment,'qualify':qualify_aug,'freeze':freeze,'report':report}[a.mode](yaml.safe_load(SPEC.read_text()))
