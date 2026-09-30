"""Codec-bound local tools for grammar-v3; no learned policy response/planner.

Parent-disjoint paired decoding, explicit count edits, fixed decoder noise.
Every saved predictor names the frozen codec hash and label precision.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import h5py
import torch
from torch.nn import functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from .experiment import load_codec, physical, chart_input, fit_uncertainty
from .models import LocalTraversal, RBFPrecision
from starscream.course_model.data import CourseDataset
from starscream.course_model.model import rotation6
from starscream.course_model.directed import ordered_curve
from starscream.course_model.schema import unpack, static_reasons
from starscream.course_model.training import atomic_json


def descriptor(raw):
    """Ordered 48-phase curve + directed normals, in the online compass units."""
    p=raw[...,:3].float();length=(p.roll(-1,1)-p).norm(dim=-1).clamp_min(1e-6)
    cumulative=F.pad(length.cumsum(1),(1,0));phase=torch.arange(48,device=p.device)[None]/48*cumulative[:,-1:]
    ids=(torch.searchsorted(cumulative.contiguous(),phase.contiguous(),right=True)-1).clamp(0,p.shape[1]-1)
    normal=rotation6(raw[...,3:9].float())[...,0]*(1-2*(raw[...,15:16]>0).float())
    return torch.cat([ordered_curve(p,48).flatten(1)/np.sqrt(48),normal.gather(1,ids[...,None].expand(-1,-1,3)).flatten(1)*2/np.sqrt(48)],-1)


def physical_pad(raw):return F.pad(physical(raw),(0,0,0,32-raw.shape[1])).flatten(1)


def response_target(base,moved):
    same=base.shape[1]==moved.shape[1]
    a=descriptor(moved)-descriptor(base)
    b=physical_pad(moved)-physical_pad(base) if same else torch.zeros(len(base),352,device=base.device)
    mask=F.pad(torch.ones(len(base),base.shape[1],11,device=base.device),(0,0,0,32-base.shape[1])).flatten(1) if same else torch.zeros_like(b)
    return torch.cat([a,b],-1),torch.cat([torch.ones_like(a),mask],-1)


def balanced_ids(family,count,parent,split,budget,seed):
    """One example per parent, round-robin family/count cells, fixed seed."""
    rng=np.random.default_rng(seed);cells={};seen=set()
    for i in rng.permutation(np.flatnonzero(split)):
        if int(parent[i]) in seen:continue
        seen.add(int(parent[i]));cells.setdefault((int(family[i]),int(count[i])),[]).append(int(i))
    result=[]
    while any(cells.values()) and len(result)<budget:
        for key in rng.permutation(len(cells)):
            bucket=list(cells.values())[key]
            if bucket and len(result)<budget:result.append(bucket.pop())
    return np.array(result,np.int64)


def valid_raw(raw,context):
    g=raw.copy();g[:,15:17]=1/(1+np.exp(-np.clip(g[:,15:17],-50,50)))
    try:
        t=unpack(g,context)
        return int(not static_reasons(t) and all(np.all(v.position>=t.bounds[:,0]) and np.all(v.position<=t.bounds[:,1]) for v in t.gates))
    except (ValueError,OverflowError):return 0


def signature(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


@torch.no_grad()
def collect(spec,root,smoke=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'chart.h5'
    if path.exists():raise FileExistsError(path)
    started=time.monotonic();torch.manual_seed(spec['seed']);torch.set_num_threads(2)
    codec,saved=load_codec(spec['codec']);ds=CourseDataset(spec['dataset'],cache=True)
    with h5py.File(spec['dataset']) as f:
        parent=f['parent_id'][:];split=f['split'][:];family=f['family'][:];count=np.diff(f['offsets'][:])
    rng=np.random.default_rng(spec['seed']);vp=rng.permutation(np.unique(parent[split==1]));cal=set(vp[::2].tolist())
    groups=[split==0,(split==1)&np.isin(parent,list(cal)),(split==1)&~np.isin(parent,list(cal))]
    budgets=[spec['train_courses'],spec['calibration_courses'],spec['test_courses']] if not smoke else [24,8,8]
    ids=[];parts={}
    for part,(group,budget) in enumerate(zip(groups,budgets)):
        chosen=balanced_ids(family,count,parent,group,budget,spec['seed']+part)
        ids.extend(chosen);parts.update({int(i):part for i in chosen})
    bs=spec['decode_batch_size'];rows=[]
    # FP32 responses avoid quantization dominating small delta targets. Fixed
    # Heun64 and fixed noise match the generator algorithm; precision is audited.
    def decode(z,n,c):return codec.decode(z,torch.full((len(z),),n,device='cuda',dtype=torch.long),c,_length=n,steps=spec['decode_steps'])[0]
    for n in sorted(set(count[ids])):
        selected=[int(i) for i in ids if count[i]==n]
        for j in range(0,len(selected),bs):
            block=selected[j:j+bs];examples=[ds[i] for i in block]
            g=torch.stack([e['gates'] for e in examples]).cuda();c=torch.stack([e['context'] for e in examples]).cuda()
            z,_=codec.encode(g,torch.ones(g.shape[:2],device='cuda',dtype=torch.bool),c,_uniform=True)
            raw=decode(z,int(n),c)
            for k,i in enumerate(block):rows.append(dict(index=i,z=z[k].cpu(),g=g[k].cpu(),c=c[k].cpu(),raw=raw[k].cpu(),n=int(n),parent=int(parent[i]),family=int(family[i]),split=parts[i]))
        print(f'encoded count={n} courses={len(rows)}/{len(ids)} seconds={time.monotonic()-started:.1f}',flush=True)
    tz=torch.stack([r['z'] for r in rows if r['split']==0]);zm=tz.mean(0).cuda();zs=tz.std(0,unbiased=False).clamp_min(.05).cuda()
    digest=signature(spec['codec'])
    with h5py.File(path,'w') as f:
        f.attrs.update(schema='course-navigation-v2',complete=False,codec_sha256=digest,decode_steps=spec['decode_steps'],precision='FP32 fixed common noise')
        f.create_dataset('zmean',data=zm.cpu());f.create_dataset('zscale',data=zs.cpu())
        def append(group,data):
            gr=f.require_group(group)
            for key,value in data.items():
                a=value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
                if key not in gr:gr.create_dataset(key,data=a,maxshape=(None,*a.shape[1:]),chunks=True,compression='lzf',shuffle=True)
                else:
                    d=gr[key];old=len(d);d.resize(old+len(a),axis=0);d[old:]=a
        # Donors stay in the same split; never leak held-out target codes.
        for part in range(3):
            donors=torch.stack([r['z'] for r in rows if r['split']==part]).cuda()
            for n in sorted(set(r['n'] for r in rows if r['split']==part)):
                bucket=[r for r in rows if r['n']==n and r['split']==part]
                for j in range(0,len(bucket),bs):
                    b=bucket[j:j+bs];z=torch.stack([r['z'] for r in b]).cuda();c=torch.stack([r['c'] for r in b]).cuda()
                    raw=torch.stack([r['raw'] for r in b]).cuda();g=torch.stack([r['g'] for r in b]).cuda()
                    counts=torch.full((len(b),),n,device='cuda',dtype=torch.long);x=chart_input(z,counts,c,zm,zs)
                    geometry=torch.cat([physical_pad(raw),descriptor(raw)],-1)
                    metadata={k:[r[k] for r in b] for k in ['parent','split','family','index']}
                    mask=F.pad(torch.ones(len(b),n,11,device='cuda'),(0,0,0,32-n)).flatten(1)
                    append('base',dict(x=x,z=z,context=c,count=counts,residual=physical_pad(raw)-physical_pad(g),mask=mask,**metadata))
                    for k,radius in enumerate(spec['radii']):
                        direction=torch.randn_like(z)
                        if k%2:
                            target=donors[torch.randint(len(donors),(len(z),),device='cuda')]
                            direction=(target-z)/zs
                        direction=F.normalize(direction,dim=-1);dz=direction*radius
                        next_n=n
                        # Last move crosses one count chart. No gate-aligned
                        # loss on count changes: only ordered phase descriptors.
                        if k==len(spec['radii'])-1:next_n=n-1 if n>4 else n+1
                        moved=decode(z+dz*zs,next_n,c);y,mk=response_target(raw,moved)
                        nx=F.one_hot(torch.full_like(counts,next_n)-1,32).float()*4
                        effective=torch.full((len(z),),(radius**2+(next_n-n)**2)**.5,device='cuda')
                        padded=F.pad(moved,(0,0,0,32-next_n))
                        valid=[valid_raw(q,cc) for q,cc in zip(moved.cpu().numpy(),c.cpu().numpy())]
                        append('edges',dict(x=torch.cat([x,dz,nx],-1),geometry=geometry,y=y,mask=mk,radius=effective,
                            latent_radius=np.full(len(b),radius),count=counts,next_count=np.full(len(b),next_n),valid=valid,
                            z=z,dz=dz,context=c,raw=padded,**metadata))
                    if (j//bs)%8==0:
                        f.flush();print(f'edits split={part} count={n} rows={len(f["edges/parent"])} seconds={time.monotonic()-started:.1f}',flush=True)
        f.attrs['complete']=True
        report=dict(courses=len(rows),edges=len(f['edges/parent']),seconds=time.monotonic()-started,codec_sha256=digest,
            dataset_sha256=signature(spec['dataset']),splits={str(i):sum(r['split']==i for r in rows) for i in range(3)},
            precision='FP32',decoder_steps=spec['decode_steps'],smoke=smoke)
    del codec,saved,ds;torch.cuda.empty_cache();atomic_json(root/'data.json',report);return report


def load_edges(path,conditioned=False):
    with h5py.File(path) as f:
        assert f.attrs['complete'];digest=f.attrs['codec_sha256']
        d={k:torch.from_numpy(f['edges'][k][:]).cuda() for k in ['x','geometry','y','mask','radius','split','parent','count','next_count','family','latent_radius','valid']}
    for a in [0,1,2]:
        for b in range(a+1,3):assert not (set(d['parent'][d['split']==a].tolist())&set(d['parent'][d['split']==b].tolist()))
    if conditioned:
        g=d['geometry'].float();scale=torch.ones(g.shape[-1],device='cuda');scale[:352]=torch.tensor([20,20,20,1,1,1,1,1,1,1,1]*32,device='cuda');scale[352:496]=20
        d['x']=torch.cat([d['x'],g/scale],-1)
    d['x']=d['x'].float();d['y']=d['y'].float();d['mask']=d['mask'].float();d['radius']=d['radius'].float()
    return d,digest


def fit_local_v2(spec,root,arm,nav=None,smoke=False):
    root=Path(root);out=root/arm;out.mkdir(exist_ok=True);d,digest=load_edges(root/'chart.h5',arm=='conditioned')
    x,y,mk,r=d['x'],d['y'],d['mask'],d['radius'];tr=d['split']==0;ids=tr.nonzero().flatten()
    scale=((y[tr].square()*mk[tr]).sum(0)/mk[tr].sum(0).clamp_min(1)).sqrt().clamp_min(.005)
    members=[];epochs=2 if smoke else spec['local_epochs'];width=spec['local_width'];started=time.monotonic()
    for member in range(1 if smoke else spec['ensemble']):
        torch.manual_seed(spec['seed']+member);model=LocalTraversal(x.shape[1],y.shape[1],width).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=spec['lr'],betas=(.9,.95),weight_decay=.01,fused=True)
        for epoch in range(epochs):
            model.train();losses=[]
            for batch in ids[torch.randperm(len(ids),device='cuda')].split(spec['batch_size']):
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x[batch],r[batch])
                err=F.smooth_l1_loss(pred.float(),y[batch]/scale,reduction='none',beta=.25)
                loss=(err*mk[batch]).sum()/mk[batch].sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();losses.append(loss.detach())
            if (epoch+1)%10==0 or smoke:
                row=dict(epoch=epoch+1,member=member,loss=float(torch.stack(losses).mean()));print(arm,row,flush=True)
                if nav:nav.log({f'{arm}/{k}':v for k,v in row.items()})
        members.append(model.eval())
    reports={}
    with torch.no_grad():
        # Low-cost ridge comparator; randomized feature projection fixed across
        # arms avoids a large cubic solve and uses only training targets.
        torch.manual_seed(spec['seed']);projection=torch.randn(x.shape[1],128,device='cuda')/x.shape[1]**.5
        xx=torch.cat([x@projection,torch.ones(len(x),1,device='cuda')],-1)*r[:,None]
        coef=torch.linalg.solve(xx[tr].T@xx[tr]+torch.eye(129,device='cuda'),xx[tr].T@(y[tr]/scale))
        for part in [1,2]:
            ids=(d['split']==part).nonzero().flatten();preds=[]
            for model in members:
                p=[]
                for batch in ids.split(512):
                    with torch.autocast('cuda',dtype=torch.bfloat16):p.append(model(x[batch],r[batch]).float()*scale)
                preds.append(torch.cat(p))
            p=torch.stack(preds);mean=p.mean(0);mask=mk[ids];truth=y[ids]
            error=((mean-truth).square()*mask).sum(-1)/mask.sum(-1);unc=(p.var(0,unbiased=False)*mask).sum(-1)/mask.sum(-1)
            rho=spearmanr(error.cpu(),unc.cpu()).statistic if len(members)>1 else np.nan
            report=dict(mse=float(error.mean()),zero_mse=float(((truth.square()*mask).sum(-1)/mask.sum(-1)).mean()),
                ridge_mse=float((((xx[ids]@coef*scale-truth).square()*mask).sum(-1)/mask.sum(-1)).mean()),
                direction_cosine=float(F.cosine_similarity(mean*mask,truth*mask,dim=-1).mean()),
                uncertainty_error_spearman=float(rho) if np.isfinite(rho) else None,
                count_change_mse=float(error[d['count'][ids]!=d['next_count'][ids]].mean()),
                per_radius={str(float(v)):float(error[d['latent_radius'][ids]==v].mean()) for v in d['latent_radius'].unique()},
                per_family={str(int(v)):float(error[d['family'][ids]==v].mean()) for v in d['family'][ids].unique()})
            reports['calibration' if part==1 else 'test']=report
        torch.save(dict(states=[m.state_dict() for m in members],input_dim=x.shape[1],output_dim=y.shape[1],width=width,scale=scale.cpu(),conditioned=arm=='conditioned',codec_sha256=digest,decode_steps=spec['decode_steps'],schema='course-navigation-v2',normalization='chart zmean/zscale; geometry position/20; radius includes count edit'),out/'local.pt')
    reports.update(parameters_per_member=sum(p.numel() for p in members[0].parameters()),seconds=time.monotonic()-started)
    atomic_json(out/'eval.json',reports);return reports


def fit_uncertainty_v2(spec,root,smoke=False):
    root=Path(root);out=root/'uncertainty';out.mkdir(exist_ok=True)
    model,report=fit_uncertainty(root/'chart.h5',out,epochs=2 if smoke else spec['uncertainty_epochs'],seed=spec['seed'])
    with h5py.File(root/'chart.h5') as f:
        b=f['base'];cal=b['split'][:]==1;test=b['split'][:]==2
        x=torch.tensor(b['x'][:],device='cuda');y=torch.tensor(b['residual'][:],device='cuda');mask=torch.tensor(b['mask'][:],device='cuda')
        digest=f.attrs['codec_sha256']
    with torch.no_grad():
        precision=model(x);temperature=float((y[cal].square()*precision[cal]*mask[cal]).sum()/mask[cal].sum())
        temperature=max(temperature,1e-4);scaled=y[test].abs()*(precision[test]/temperature).sqrt();valid=mask[test].bool()
        report.update(test_coverage_1sigma=float((scaled[valid]<=1).float().mean()),test_coverage_2sigma=float((scaled[valid]<=2).float().mean()),calibration_temperature=temperature,codec_sha256=digest)
    payload=torch.load(out/'uncertainty.pt',map_location='cpu',weights_only=False);payload.update(codec_sha256=digest,calibration_temperature=temperature);torch.save(payload,out/'uncertainty.pt')
    atomic_json(out/'eval.json',report);return report


def fit_static(spec,root):
    """Geometry-validity screening only: never relabel it physical flyability."""
    root=Path(root)
    with h5py.File(root/'chart.h5') as f:
        e=f['edges'];raw=e['raw'][:];n=e['next_count'][:];ctx=e['context'][:];y=e['valid'][:];split=e['split'][:]
    x=screen_features(raw,n,ctx);tr=split==0;te=split==2
    if len(np.unique(y[tr]))<2:
        report=dict(state='one_class_training_data',deployable=False,positive_fraction=float(y.mean()))
    else:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=300)).fit(x[tr],y[tr]);p=model.predict_proba(x[te])[:,1]
        import joblib
        joblib.dump(model,root/'static_model.joblib')
        report=dict(state='complete',test_brier=float(brier_score_loss(y[te],p)),constant_brier=float(brier_score_loss(y[te],np.full(te.sum(),y[tr].mean()))),test_auc=float(roc_auc_score(y[te],p)) if len(np.unique(y[te]))>1 else None,
            meaning='static geometry validity only, NOT MPCC or physical feasibility')
    atomic_json(root/'static_eval.json',report);return report


def screen_features(raw,count,context):
    # Masked ordered slots retain frames/kinds rather than relying solely on
    # hand-designed mean geometry statistics.
    scale=np.array([20,20,20,1,1,1,1,1,1,1,1,1,1,1,1,4,4],np.float32)
    return np.c_[(raw/scale).reshape(len(raw),-1),np.eye(32)[np.asarray(count)-1],context/20]
