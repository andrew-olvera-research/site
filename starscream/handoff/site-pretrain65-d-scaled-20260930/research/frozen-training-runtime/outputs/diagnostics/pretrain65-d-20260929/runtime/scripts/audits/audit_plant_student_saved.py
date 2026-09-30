"""Read-only saved-result/replay audit; no rollouts or checkpoint updates."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.train_vision_student import load_setup, SCALE

ROOT = Path('outputs/vision-distillation/v62111-plant-ekf-readout-async-48r')
OUT = Path('outputs/diagnostics/plant-student-saved-audit')


def outcomes():
    paired = json.loads(Path('outputs/evals/v62111-plant-student-final/best-real100-hard-v2-e8-paired.json').read_text())
    selection = json.loads(Path('configs/eval/v6211_vision_selection25.json').read_text())
    selected = {r['slot'] for r in selection['records']}
    rows = []
    for t, s in zip(paired['teacher']['tracks'], paired['student']['tracks']):
        assert t['slot'] == s['slot']
        for i, (te, se) in enumerate(zip(t['episodes'], s['episodes'])):
            rows.append(dict(slot=t['slot'], family=t['family'], selection=t['slot'] in selected,
                             episode=i, teacher=te, student=se))
    def summarize(group):
        t = np.array([r['teacher']['success'] for r in group])
        s = np.array([r['student']['success'] for r in group])
        both = [r for r in group if r['teacher']['success'] and r['student']['success']]
        result = dict(n=len(group),teacher_success=float(t.mean()),student_success=float(s.mean()),
                      retention=float(s.sum()/t.sum()) if t.sum() else None,
                      both_success=len(both),lost=int(((t==1)&(s==0)).sum()),gained=int(((t==0)&(s==1)).sum()))
        if both:
            ratios = [r['student']['lap_seconds']/r['teacher']['lap_seconds'] for r in both]
            result['matched_lap_ratio_median'] = float(np.median(ratios))
            result['matched_lap_difference_mean'] = float(np.mean([r['student']['lap_seconds']-r['teacher']['lap_seconds'] for r in both]))
        for arm in ('teacher','student'):
            times = [r[arm]['lap_seconds'] for r in group if r[arm]['success']]
            result[arm+'_successful_lap_quantiles'] = np.quantile(times,[0,.5,.9,1]).tolist() if times else []
            result[arm+'_step_cap_failures'] = sum(not r[arm]['success'] and r[arm]['steps']==10000 for r in group)
        return result
    result = {name:summarize(group) for name,group in [('full100',rows),('selection25',[r for r in rows if r['selection']]),('complement75',[r for r in rows if not r['selection']])]}
    result['completion_deadlines'] = {}
    for cap in (6000,10000):
        counts={arm:sum(r[arm]['success'] and r[arm]['steps']<=cap for r in rows) for arm in ('teacher','student')}
        result['completion_deadlines'][str(cap)]=dict(**counts,retention=counts['student']/counts['teacher'])
    result['families'] = {f:summarize([r for r in rows if r['family']==f]) for f in sorted({r['family'] for r in rows})}
    # Cluster bootstrap courses, retaining all paired episodes within each course.
    rng = np.random.default_rng(725)
    for name, flag in [('full100',None),('complement75',False)]:
        subset = [r for r in rows if flag is None or r['selection']==flag]
        slots = sorted({r['slot'] for r in subset})
        counts = np.array([[sum(r[a]['success'] for r in subset if r['slot']==slot) for a in ('teacher','student')] for slot in slots])
        draws = counts[rng.integers(len(slots),size=(10000,len(slots)))].sum(1)
        result[name]['retention_course_bootstrap_95'] = np.quantile(draws[:,1]/draws[:,0],[.025,.975]).tolist()
        result[name]['difference_course_bootstrap_95'] = np.quantile((draws[:,1]-draws[:,0])/(8*len(slots)),[.025,.975]).tolist()
    result['course_tails'] = []
    for t,s in zip(paired['teacher']['tracks'], paired['student']['tracks']):
        record = dict(slot=t['slot'],family=t['family'])
        for name,arm in [('teacher',t),('student',s)]:
            times=[e['lap_seconds'] for e in arm['episodes'] if e['success']]
            record[name+'_slow_fast_ratio'] = max(times)/min(times) if len(times)>1 else None
        result['course_tails'].append(record)
    return result


def neural():
    torch.set_num_threads(1)
    torch.manual_seed(725)
    saved = torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    config=saved['config']
    _,settings,source,_,teacher,normalizer,student=load_setup(config)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    teacher.to(device).eval(); student.to(device).eval();student.load_state_dict(saved['model'])
    predictor=torch.nn.Linear(student.width,teacher.hidden_dim).to(device)
    predictor.load_state_dict(saved['predictor']);predictor.eval()
    mean=torch.as_tensor(normalizer.mean,device=device)
    std=torch.as_tensor(normalizer.std,device=device)
    rng=np.random.default_rng(725)
    pieces=[]
    for r in (0,3,35,39,47):
        path=ROOT/'replay'/f'round-{r:04d}'
        meta=json.loads((path/'metadata.json').read_text())
        arrays={k:np.load(path/f'{k}.npy',mmap_mode='r') for k in meta['fields']}
        ids=rng.choice(meta['rows'],size=128,replace=False)
        pieces.append({k:np.array(a[ids]) for k,a in arrays.items()})
    raw={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    result=dict(best_round=saved['round']+1,device=device,rows=len(raw['action']),parameter_counts=student.parameter_counts(),predictor_parameters=sum(p.numel() for p in predictor.parameters()))
    accum=defaultdict(list)
    hidden=[];targets=[];predictions=[]
    for start in range(0,len(raw['action']),32):
        b={k:torch.as_tensor(v[start:start+32],device=device).float() for k,v in raw.items() if k!='masks'}
        masks=torch.as_tensor(np.unpackbits(raw['masks'][start:start+32],axis=-1,count=160),device=device).float()
        x=(b['teacher_features']-mean)/std
        with torch.no_grad():
            base=teacher.encode(x);target=teacher._conditioned_encoding(base,b['speed'])
            if teacher.topology_head is not None:target=target+teacher.topology_adapter(teacher.topology_head(target))
            action=teacher(x,b['speed'])
            out=student(b['features'],masks,b['speed'],b['executed'],return_hidden=True)
            prediction=predictor(out['action_readout'])
            accum['target_recompute_max_abs'].append(float((target-b['readout']).abs().max()))
            accum['action_recompute_max_abs'].append(float((action-b['action']).abs().max()))
            for k,v in [('readout_loss',1-F.cosine_similarity(prediction,b['readout'])),('action_mae',(out['action']-b['action']).abs().mean(-1))]:accum[k].extend(v.cpu().tolist())
            hidden.append(out['action_readout'].cpu());targets.append(b['readout'].cpu());predictions.append(prediction.cpu())
            # Same recorded states, varying only hidden plant settings (not a rollout).
            shuffled=b['teacher_features'].clone();shuffled[:,:,103:]=shuffled.flip(0)[:,:,103:]
            altered=teacher((shuffled-mean)/std,b['speed'])
            accum['plant_shuffle_action_mae'].extend((altered-action).abs().mean(-1).cpu().tolist())
            constants=b['teacher_features'].clone();constants[:,:,103:152]=constants.flip(0)[:,:,103:152]
            accum['plant_constants_shuffle_action_mae'].extend((teacher((constants-mean)/std,b['speed'])-action).abs().mean(-1).cpu().tolist())
            for label,ff,mm in [('mask_zero',b['features'],torch.zeros_like(masks)),('history_repeat',b['features'][:,-1:].expand_as(b['features']),masks[:,-1:].expand_as(masks))]:
                changed=student(ff,mm,b['speed'])['action']
                accum[label+'_action_shift'].extend((changed-out['action']).abs().mean(-1).cpu().tolist())
        if start==0:
            out=student(b['features'],masks,b['speed'],b['executed'],return_hidden=True)
            pred=predictor(out['action_readout'])
            losses={'action':F.smooth_l1_loss(out['action'],b['action'],beta=.05),
                    'readout':config['readout_weight']*(1-F.cosine_similarity(pred,b['readout'])).mean()}
            estimated=b['teacher_features'].clone();estimated[:,-1,:19]=out['estimate']*torch.as_tensor(SCALE,device=device)
            losses['interface']=config['interface_weight']*F.smooth_l1_loss(teacher((estimated-mean)/std,b['speed']),b['action'],beta=.05)
            params=[p for n,p in student.named_parameters() if n.startswith(('blocks.','state.','vision_encoder.'))]
            grads={k:torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True))]) for k,loss in losses.items()}
            result['weighted_gradient_norms']={k:float(g.norm()) for k,g in grads.items()}
            result['gradient_cosines']={k:float(F.cosine_similarity(grads['action'][None],g[None])) for k,g in grads.items() if k!='action'}
            result['teacher_frozen']=not any(p.requires_grad for p in teacher.parameters())
    h=torch.cat(hidden).double();y=torch.cat(targets).double();p=torch.cat(predictions).double()
    yc=y-y.mean(0)
    eigen=torch.linalg.svdvals(yc).square()
    result['target_centered_energy_top256']=float(eigen[:256].sum()/eigen.sum())
    result['target_effective_rank']=float(torch.exp(-((eigen/eigen.sum()).clamp_min(1e-30)*(eigen/eigen.sum()).clamp_min(1e-30).log()).sum()))
    result['target_mean_cosine_baseline']=float(F.cosine_similarity(y.mean(0).expand_as(y),y).mean())
    result['predicted_cosine']=float(F.cosine_similarity(p,y).mean())
    result['centered_predicted_cosine']=float(F.cosine_similarity(p-p.mean(0),yc).mean())
    # Fit disposable probes on frozen hidden features; use a disjoint round-47 test split.
    train=torch.arange(512);test=torch.arange(512,640)
    hm=h[train].mean(0);ym=y[train].mean(0)
    hx=h[train]-hm;yx=y[train]-ym
    fit=torch.linalg.solve(hx.T@hx+torch.eye(h.shape[1],dtype=h.dtype)*10,hx.T@yx)
    yp=(h[test]-hm)@fit+ym
    result['ridge_probe_test_cosine']=float(F.cosine_similarity(yp,y[test]).mean())
    result['existing_predictor_test_cosine']=float(F.cosine_similarity(p[test],y[test]).mean())
    result['metrics']={k:dict(mean=float(np.mean(v)),max=float(np.max(v))) for k,v in accum.items()}
    result['strata']={str(s):dict(count=int((raw['stratum']==s).sum()),action_mae=float(np.mean(np.array(accum['action_mae'])[raw['stratum']==s])),readout_loss=float(np.mean(np.array(accum['readout_loss'])[raw['stratum']==s]))) for s in range(3)}
    return result


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(outcomes=outcomes(),neural=neural())
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result['outcomes'].items() if k!='course_tails'},indent=2))
    print(json.dumps(result['neural'],indent=2))
