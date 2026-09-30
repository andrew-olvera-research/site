"""Bounded, reproducible local-chart module training. Planner stays classical.

Data remain parent-disjoint. Codec is frozen before downstream tool fitting.
No policy updates, target geometry supervision, or implicit MPCC admission.
"""
import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import BayesianRidge
from scipy.stats import spearmanr
from starscream.course_model.data import CourseDataset
from starscream.course_model.flow import build_course_model
from starscream.course_model.model import VAEConfig,rotation6
from starscream.course_model.schema import unpack,static_reasons
from starscream.course_model.training import atomic_json
from .models import RBFPrecision,LocalTraversal,gaussian_precision_nll,pullback_from_jacobians
from .compass import local_inverse_step


def physical(raw):
    """1m positional units, proper frame columns, log aperture; 11 features."""
    r=rotation6(raw[...,3:9].float())
    return torch.cat([raw[...,:3].float(),r[...,0],r[...,1],raw[...,9:11].float()],-1)


def chart_input(z,counts,context,zmean,zscale):
    cs=torch.ones(19,device=z.device);cs[:3]=20;cs[9:12]=10;cs[13:]=30
    return torch.cat([(z-zmean)/zscale,F.one_hot(counts.long()-1,32).float()*4,context/cs],-1)


def load_codec(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    m=build_course_model(VAEConfig(**saved['config'])).cuda().eval()
    m.load_state_dict(saved['model']);m.requires_grad_(False)
    return m,saved


@torch.no_grad()
def collect_chart(codec,dataset,output,parents=4096,validation_parents=512,seed=8801,radii=(.05,.2,.5)):
    """Record direct decoded responses, including invalid endpoints as negatives.

    Perturbations measured in whitened encoder coordinates. The whitened radius
    is not physical track distance and is not a policy-readiness assertion.
    All codec targets use FP32/fixed noise; no BF16 finite-difference artifacts.
    """
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    ds=CourseDataset(dataset,cache=True)
    with h5py.File(dataset) as f:splits=f['split'][:]
    selected=np.concatenate([rng.choice(np.flatnonzero(splits==s),min(n,(splits==s).sum()),replace=False)
                             for s,n in [(0,parents),(1,validation_parents)]])
    examples=[ds[int(i)] for i in selected]; rows=[]
    # Encoding grouped by count keeps the same contract as SSL/Flash training.
    for n in sorted(set(len(e['gates']) for e in examples)):
        bucket=[e for e in examples if len(e['gates'])==n]
        for j in range(0,len(bucket),128):
            batch=bucket[j:j+128];g=torch.stack([e['gates'] for e in batch]).cuda().float()
            c=torch.stack([e['context'] for e in batch]).cuda().float()
            z,_=codec.encode(g,torch.ones(g.shape[:2],device='cuda',dtype=torch.bool),c,_uniform=True)
            counts=torch.full((len(g),),n,device='cuda',dtype=torch.long)
            raw,_=codec.decode(z,counts,c,_length=n)
            for k,e in enumerate(batch):rows.append(dict(z=z[k].cpu(),g=g[k].cpu(),c=c[k].cpu(),raw=raw[k].cpu(),
                parent=e['parent_id'],split=int(splits[e['index']]),index=e['index'],n=n))
    trainz=torch.stack([r['z'] for r in rows if r['split']==0])
    zm=trainz.mean(0).cuda();zs=trainz.std(0,unbiased=False).clamp_min(.05).cuda()
    all_rows={k:[] for k in ('x','y','mask','radius','parent','split','valid','count')}
    bases={k:[] for k in ('x','residual','mask','parent','split','z','count','context')}
    for n in sorted(set(r['n'] for r in rows)):
        bucket=[r for r in rows if r['n']==n]
        for j in range(0,len(bucket),128):
            b=bucket[j:j+128];z=torch.stack([r['z'] for r in b]).cuda();c=torch.stack([r['c'] for r in b]).cuda()
            raw=torch.stack([r['raw'] for r in b]).cuda();g=torch.stack([r['g'] for r in b]).cuda()
            counts=torch.full((len(b),),n,device='cuda',dtype=torch.long)
            x=chart_input(z,counts,c,zm,zs);pad=lambda a:F.pad(a,(0,0,0,32-n)).flatten(1)
            mask=pad(torch.ones(len(b),n,11,device='cuda'))
            bases['x'].append(x.cpu().numpy());bases['residual'].append(pad(physical(raw)-physical(g)).cpu().numpy())
            bases['mask'].append(mask.cpu().numpy());bases['z'].append(z.cpu().numpy());bases['context'].append(c.cpu().numpy())
            for key in ('parent','split'):bases[key].extend(r[key] for r in b)
            bases['count'].extend([n]*len(b))
            for radius in radii:
                direction=F.normalize(torch.randn_like(z),dim=-1);dz=direction*radius
                moved,_=codec.decode(z+dz*zs,counts,c,_length=n)
                y=pad(physical(moved)-physical(raw))
                # x includes start code, count/context, signed whitened delta.
                all_rows['x'].append(torch.cat([x,dz],-1).cpu().numpy())
                all_rows['y'].append(y.cpu().numpy());all_rows['mask'].append(mask.cpu().numpy())
                for key in ('parent','split'):all_rows[key].extend(r[key] for r in b)
                all_rows['radius'].extend([radius]*len(b));all_rows['count'].extend([n]*len(b))
                arr=moved.cpu().numpy();arr[...,15:17]=moved[...,15:17].sigmoid().cpu().numpy()
                for k,a in enumerate(arr):
                    try:
                        track=unpack(a,c[k].cpu().numpy());ok=not static_reasons(track)
                        ok=ok and bool(((a[:,:3]>=track.bounds[:,0])&(a[:,:3]<=track.bounds[:,1])).all())
                    except (ValueError,OverflowError):ok=False
                    all_rows['valid'].append(int(ok))
    with h5py.File(path,'w') as f:
        f.attrs.update(schema='course-navigation-v1',complete=False,seed=seed,codec_frozen=True)
        f.create_dataset('zmean',data=zm.cpu().numpy());f.create_dataset('zscale',data=zs.cpu().numpy())
        for name,data in [('base',bases),('edges',all_rows)]:
            group=f.create_group(name)
            for k,v in data.items():
                a=np.concatenate(v) if isinstance(v[0],np.ndarray) else np.asarray(v)
                group.create_dataset(k,data=a,compression='lzf',shuffle=True)
        f.attrs['complete']=True
    return dict(courses=len(rows),unique_parents=len({r['parent'] for r in rows}),train=sum(r['split']==0 for r in rows),validation=sum(r['split']==1 for r in rows),
                edges=len(all_rows['parent']),bytes=path.stat().st_size)


def read_chart(path,group):
    with h5py.File(path) as f:
        if not f.attrs.get('complete'):raise ValueError('incomplete chart')
        d={k:torch.tensor(v[:],device='cuda') for k,v in f[group].items()}
    for k in ('x','y','mask','residual','radius'):
        if k in d:d[k]=d[k].float()
    tr=d['split']==0;va=~tr
    if set(d['parent'][tr].tolist()) & set(d['parent'][va].tolist()):raise ValueError('parent leakage')
    return d,tr,va


def fit_uncertainty(path,output,epochs=100,seed=8801):
    torch.manual_seed(seed);d,tr,va=read_chart(path,'base');x=d['x'];y=d['residual'];mask=d['mask']
    cpu=x[tr].cpu().numpy();k=min(64,len(cpu))
    km=MiniBatchKMeans(n_clusters=k,random_state=seed,n_init=3,batch_size=1024).fit(cpu)
    centers=torch.tensor(km.cluster_centers_,device='cuda')
    assigned=km.predict(cpu);dist=np.linalg.norm(cpu-km.cluster_centers_[assigned],axis=-1)
    bandwidth=np.array([max(.1,dist[assigned==i].mean() if (assigned==i).any() else 1.) for i in range(k)])
    m=RBFPrecision(centers,torch.tensor(bandwidth,device='cuda',dtype=torch.float32),y.shape[1]).cuda()
    opt=torch.optim.AdamW(m.parameters(),lr=.03,weight_decay=0)
    ids=tr.nonzero().flatten()
    for epoch in range(epochs):
        for batch in ids[torch.randperm(len(ids),device='cuda')].split(512):
            loss=gaussian_precision_nll(m(x[batch]),y[batch],mask[batch]);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        p=m(x[va]);var=(y[tr].square()*mask[tr]).sum(0)/mask[tr].sum(0).clamp_min(1);var=var.clamp_min(1e-4)
        scaled=y[va].abs()*p.sqrt();valid=mask[va].bool()
        report=dict(nll=float(gaussian_precision_nll(p,y[va],mask[va])),
            constant_train_variance_nll=float(gaussian_precision_nll(var.reciprocal().expand_as(p),y[va],mask[va])),
            coverage_1sigma=float((scaled[valid]<=1).float().mean()),coverage_2sigma=float((scaled[valid]<=2).float().mean()),
            in_support_sigma=float(p.rsqrt()[valid].median()),far_sigma=float(m.sigma(x[va]+100).median()),
            meaning='held-out residual calibration; NOT epistemic probability or MPCC feasibility')
    torch.save(dict(state=m.state_dict(),centers=centers.cpu(),bandwidth=m.bandwidth.cpu(),output_dim=y.shape[1],floor=m.floor),Path(output)/'uncertainty.pt')
    atomic_json(Path(output)/'uncertainty_eval.json',report)
    return m,report


def fit_local(path,output,epochs=80,width=256,seed=8801):
    torch.manual_seed(seed);d,tr,va=read_chart(path,'edges');x=d['x'];y=d['y'];mask=d['mask'];r=d['radius']
    # Residual scale estimated only on TRAIN edges, pads excluded.
    scale=((y[tr].square()*mask[tr]).sum(0)/mask[tr].sum(0).clamp_min(1)).sqrt().clamp_min(.005)
    ys=y/scale;members=[];history=[];ids=tr.nonzero().flatten()
    for member in range(3):
        torch.manual_seed(seed+member);m=LocalTraversal(x.shape[1],y.shape[1],width).cuda()
        opt=torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.01,fused=True)
        for epoch in range(epochs):
            for batch in ids[torch.randperm(len(ids),device='cuda')].split(512):
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=m(x[batch],r[batch])
                err=F.smooth_l1_loss(pred.float(),ys[batch],reduction='none',beta=.25)
                loss=(err*mask[batch]).sum()/mask[batch].sum();opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
            if (epoch+1)%20==0:print(json.dumps(dict(module='local',member=member,epoch=epoch+1,loss=float(loss))),flush=True)
        members.append(m.eval())
    with torch.no_grad():
        preds=torch.stack([m(x[va],r[va]).float()*scale for m in members]);pred=preds.mean(0);truth=y[va];mk=mask[va]
        mse=lambda e:float((e.square()*mk).sum()/mk.sum())
        # Linear/ridge reference fitted on train only; penalize intercept too.
        xx=torch.cat([x,torch.ones(len(x),1,device='cuda')],-1)*r[:,None]
        coef=torch.linalg.solve(xx[tr].T@xx[tr]+torch.eye(xx.shape[1],device='cuda'),xx[tr].T@ys[tr])
        linear=xx[va]@coef*scale
        uncertainty=((preds.var(0,unbiased=False)*mk).sum(-1)/mk.sum(-1)).sqrt()
        error=(((pred-truth).square()*mk).sum(-1)/mk.sum(-1)).sqrt()
        rho=spearmanr(uncertainty.cpu(),error.cpu()).statistic
        by_radius={}
        for v in r.unique():
            subset=r[va]==v
            rr=spearmanr(uncertainty[subset].cpu(),error[subset].cpu()).statistic
            by_radius[str(float(v))]=dict(rows=int(subset.sum()),mse=float(error[subset].square().mean()),
                uncertainty_error_spearman=float(rr) if np.isfinite(rr) else None)
        report=dict(mse=mse(pred-truth),zero_move_baseline_mse=mse(truth),ridge_baseline_mse=mse(linear-truth),
            uncertainty_error_spearman=float(rho) if np.isfinite(rho) else None,
            per_radius=by_radius,
            directional_cosine=float(F.cosine_similarity(pred*mk,truth*mk,dim=-1).mean()),
            static_valid_endpoint_fraction=float(d['valid'][va].float().mean()),train_rows=int(tr.sum()),validation_rows=int(va.sum()),
            meaning='learned frozen-decoder response, not true track edit recovery or learning transfer')
        torch.save(dict(states=[m.state_dict() for m in members],input_dim=x.shape[1],output_dim=y.shape[1],width=width,scale=scale.cpu()),Path(output)/'local.pt')
    atomic_json(Path(output)/'local_eval.json',report)
    return report


def metric_probe(codec,uncertainty,chart,output,filename='metric_probe.json'):
    """FP32 central differences, step-halving check, fixed count/context.

    Metric is on whitened coordinates. Position, proper frame, log-size units
    are explicit; not a learned behavioral distance or an SPD-by-assumption.
    """
    with h5py.File(chart) as f:
        b=f['base'];valid=np.flatnonzero(b['split'][:]==1);count_values=b['count'][:]
        rng=np.random.default_rng(8803)
        ids=np.concatenate([rng.choice(valid[count_values[valid]==n],min(2,(count_values[valid]==n).sum()),replace=False)
                            for n in np.unique(count_values[valid])])
        data={k:b[k][:][ids] for k in ('z','count','context')};zm=torch.tensor(f['zmean'][:],device='cuda');zs=torch.tensor(f['zscale'][:],device='cuda')
    records=[]
    with torch.no_grad():
        for i in range(len(ids)):
            z=torch.tensor(data['z'][i],device='cuda');n=int(data['count'][i]);ctx=torch.tensor(data['context'][i],device='cuda')
            def jacs(eps):
                delta=torch.eye(len(z),device='cuda')*eps
                zz=torch.cat([z+delta*zs,z-delta*zs]);cc=ctx[None].expand(len(zz),-1);counts=torch.full((len(zz),),n,device='cuda',dtype=torch.long)
                raw,_=codec.decode(zz,counts,cc,_length=n);feat=physical(raw).flatten(1)
                sigma=uncertainty.sigma(chart_input(zz,counts,cc,zm,zs))[:,:n*11]
                a,b=feat.chunk(2);u,v=sigma.chunk(2)
                return ((a-b)/(2*eps)).T,((u-v)/(2*eps)).T
            jm,js=jacs(.01);half,_=jacs(.005)
            g=pullback_from_jacobians(jm,js);e=torch.linalg.eigvalsh(g)
            # Classical inverse-direction probe; actual decoded response checked
            # after the step, not merely the linearized objective improving.
            wanted=torch.zeros(n*11,device='cuda');wanted[11+2]=.25
            move=local_inverse_step(jm,wanted,metric=g,radius=.25,damping=.01)
            zz=torch.stack([z,z+move*zs]);cc=ctx[None].expand(2,-1)
            raw,_=codec.decode(zz,torch.full((2,),n,device='cuda',dtype=torch.long),cc,_length=n)
            realized=(physical(raw)[1]-physical(raw)[0]).flatten()
            records.append(dict(count=n,mean_jacobian_rank=int(torch.linalg.matrix_rank(jm)),latent_dim=len(z),
                step_halving_relative_error=float((half-jm).norm()/jm.norm().clamp_min(1e-8)),
                minimum_eigenvalue=float(e[0]),maximum_eigenvalue=float(e[-1]),
                mean_trace=float((jm.T@jm).trace()),variance_trace=float((js.T@js).trace()),
                inverse_target_error=float((wanted-realized).norm()),zero_move_error=float(wanted.norm()),
                linearization_error=float((realized-jm@move).norm()),realized_gate2_dz_m=float(realized[13])))
    atomic_json(Path(output)/filename,dict(records=records,ridge=1e-6,units='metres, rotation6, log aperture; whitened z',
        warning='variance derivative alone need not be large on a flat high-uncertainty plateau; planner also needs a support constraint'))


def fit_response(root,output):
    """Train a deliberately small intervention response diagnostic, deduplicated.

    No fake independent samples from episodes or duplicate PPO branches. Frozen
    report seed panel estimates noisy observational deltas, not causal effects
    relative to a randomized no-training counterfactual.
    """
    root=Path(root);rows=[];seen=set()
    def capability(path):
        metrics=json.loads(Path(path).read_text())['metrics'];v=[]
        for track in ('swift_champion_2022_exact','multigp_cdra_2026_reconstructed'):
            for k in ('full_course_success','mean_gates','p2','p3'):v.append(metrics[f'track/{track}/{k}'])
        return np.array(v)
    initial=capability(root/'reference/source_policy.json')
    for result in sorted(root.glob('*/result.json')):
        arm=result.parent.name;before=initial;previous=None
        for h in json.loads(result.read_text())['history']:
            if not h.get('steps'):continue
            folder=result.parent/f"segment-{h['segment']:02d}"
            selection=json.loads((folder/'selection.json').read_text());chosen=selection.get('selected')
            candidate=next((c for c in selection['candidates'] if c['name']==chosen),previous)
            if candidate is None:continue
            after=capability(folder/'reporting.json')
            feature=[candidate['target_distance'],candidate['displacement_rms'],candidate.get('policy_success',0),h['steps']/1e6]
            key=tuple(np.round(np.r_[before,feature,after],7))
            previous=candidate
            if key not in seen:
                rows.append(dict(group=arm,segment=h['segment'],x=np.r_[before,feature].tolist(),y=(after-before).tolist(),
                    course=candidate['path'],report=str(folder/'reporting.json')));seen.add(key)
            before=after
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True);atomic_json(dest/'interventions.json',rows)
    if len(rows)<3:
        report=dict(state='insufficient_data',independent_rows=len(rows),deployable=False)
    else:
        x=np.array([r['x'] for r in rows]);y=np.array([r['y'] for r in rows]);groups=np.array([r['group'] for r in rows]);pred=np.zeros_like(y)
        fits=[]
        for group in np.unique(groups):
            tr=groups!=group;va=~tr
            if tr.sum()<2:continue
            mean=x[tr].mean(0);scale=np.maximum(x[tr].std(0),.05)
            for j in range(y.shape[1]):
                model=BayesianRidge().fit((x[tr]-mean)/scale,y[tr,j]);pred[va,j]=model.predict((x[va]-mean)/scale)
        mean=x.mean(0);scale=np.maximum(x.std(0),.05)
        for j in range(y.shape[1]):
            m=BayesianRidge().fit((x-mean)/scale,y[:,j]);fits.append(dict(coef=m.coef_.tolist(),intercept=float(m.intercept_),sigma=m.sigma_.tolist(),alpha=float(m.alpha_)))
        atomic_json(dest/'response_model.json',dict(mean=mean.tolist(),scale=scale.tolist(),outputs=fits,
            input='8 current capability measurements + geometry distance, displacement, candidate SR, step budget',deployable=False))
        report=dict(state='diagnostic_fit_only',independent_rows=len(rows),lineages=len(np.unique(groups)),
            grouped_mse=float(np.square(pred-y).mean()),zero_change_mse=float(np.square(y).mean()),deployable=False,
            grouped_mse_by_output=np.square(pred-y).mean(0).tolist(),zero_change_mse_by_output=np.square(y).mean(0).tolist(),
            warning='too few interventions/targets; duplicate random/behavior branches removed; shared parent and eval seeds; not causal evidence')
    atomic_json(dest/'response_eval.json',report);return report


def run_modules(config,codec_path,output,smoke=False):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Refusing to overwrite module evidence: {output}')
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    torch.manual_seed(config['seed']);started=time.time();codec,saved=load_codec(codec_path)
    atomic_json(output/'status.json',dict(state='collecting',codec=codec_path))
    chart=output/'chart.h5'
    data=collect_chart(codec,saved['training_config']['dataset'],chart,
        parents=32 if smoke else config['parents'],validation_parents=16 if smoke else config['validation_parents'],seed=config['seed'],radii=config['radii'])
    digest=hashlib.sha256(Path(codec_path).read_bytes()).hexdigest()
    atomic_json(output/'data.json',dict(**data,codec=codec_path,codec_sha256=digest,normalization='train parents only'))
    atomic_json(output/'status.json',dict(state='uncertainty'))
    u,ur=fit_uncertainty(chart,output,epochs=2 if smoke else config['uncertainty_epochs'],seed=config['seed'])
    metric_probe(codec,u,chart,output);del codec,u;torch.cuda.empty_cache()
    atomic_json(output/'status.json',dict(state='local_traversal'))
    lr=fit_local(chart,output,epochs=2 if smoke else config['local_epochs'],width=config['local_width'],seed=config['seed'])
    rr=fit_response(config['response_root'],output)
    atomic_json(output/'bundle.json',dict(schema='course-navigation-v1',codec=codec_path,codec_sha256=digest,
        chart=str(chart),local=str(output/'local.pt'),uncertainty=str(output/'uncertainty.pt'),
        response_deployable=False,planner='classical local trust-region; no RL admission authorized',
        note='All downstream modules are specific to this frozen codec and its train-fitted z normalization.'))
    atomic_json(output/'status.json',dict(state='complete',seconds=time.time()-started,uncertainty=ur,local=lr,response=rr))
