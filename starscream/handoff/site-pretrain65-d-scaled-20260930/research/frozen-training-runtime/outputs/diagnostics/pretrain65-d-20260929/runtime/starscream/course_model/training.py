"""Configurable course-VAE SSL. Never trains or admits a flight policy."""
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend

from .augmentation import corrupt_encoder
from .data import make_loader
from .model import CourseVAE,VAEConfig,vae_loss,rotation6
from .schema import VERSION,unpack,static_reasons


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,path)


def attention_context(config):
    backend=config['training'].get('attention','flash')
    if backend not in ('flash','auto'):raise ValueError('attention must be flash or auto')
    return sdpa_kernel(SDPBackend.FLASH_ATTENTION) if backend=='flash' else nullcontext()


def setup(config):
    if config['schema']!=VERSION:raise ValueError('schema mismatch')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This benchmark requires CUDA BF16')
    torch.set_num_threads(config['training'].get('cpu_threads',2))
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    from .flow import build_course_model
    model=build_course_model(VAEConfig(**config['model'])).cuda()
    data=config['data'];loaders=[make_loader(config['dataset'],split=i,seed=seed,**data) for i in (0,1)]
    with h5py.File(config['dataset'],'r') as f:
        p=f['parent_id'][:];s=f['split'][:]
        if set(p[s==0])&set(p[s==1]):raise ValueError('parent split leakage')
        if len(f['context'])!=len(loaders[0].dataset)+len(loaders[1].dataset):raise ValueError('invalid split')
    return model,loaders


def to_device(batch):
    return batch['gates'].cuda(non_blocking=True),batch['context'].cuda(non_blocking=True)


def objective(out,target,config,beta=None,context=None):
    o=config['objective'];mask=torch.ones(target.shape[:2],dtype=torch.bool,device=target.device)
    actual_beta=o['beta'] if beta is None else beta
    if o.get('kl_reference_dim'):
        actual_beta*=float(o['kl_reference_dim'])/out['mu'].shape[-1]
    parts=vae_loss(out,target,mask,beta=actual_beta,weights=o.get('weights'))
    if o.get('geometry_weights'):
        if context is None:raise ValueError('geometry objective requires course context')
        from .geometry_objective import geometry_terms
        with torch.autocast('cuda',enabled=False):extra=geometry_terms(out['raw'],target,mask,context)
        for name,weight in o['geometry_weights'].items():
            if name not in extra or weight<0:raise ValueError('invalid geometry weight')
            parts['loss']=parts['loss']+weight*extra[name]
        parts.update({'geometry_'+k:v for k,v in extra.items()})
    if 'flow_matching' in out:
        # Keep KL independent of reconstruction weighting; CFM is not an ELBO.
        reconstruction=parts['loss']-actual_beta*parts['kl']
        parts['loss']=o.get('flow_reconstruction_weight',.1)*reconstruction+actual_beta*parts['kl']+out['flow_matching']
        parts['flow_matching']=out['flow_matching']
        if 'prior_flow_matching' in out:
            parts['prior_flow_matching']=out['prior_flow_matching']
            parts['loss']=parts['loss']+out['prior_flow_matching']
    if o.get('relational_weights'):
        if context is None:raise ValueError('relations require context')
        from .relational import relational_terms
        with torch.autocast('cuda',enabled=False):extra=relational_terms(out['raw'],target,mask,context)
        for name,weight in o['relational_weights'].items():
            if name not in extra or weight<0:raise ValueError('invalid relational weight')
            parts['loss']=parts['loss']+weight*extra[name]
        parts.update({'relation_'+k:v for k,v in extra.items()})
    return parts


def lr_at(step,total,config):
    t=config['training'];warm=t['warmup_steps']
    if step<warm:return t['lr']*(step+1)/max(warm,1)
    fraction=min(1.,(step-warm)/max(1,total-warm))
    return t['min_lr']+.5*(t['lr']-t['min_lr'])*(1+math.cos(math.pi*fraction))


@torch.no_grad()
def evaluate(model,loader,config,geometry=False):
    """All validation courses, clean inputs, fixed beta/RNG across epochs.

    Sampled weighted ELBO proxy selects best; mean-decoded physical errors and
    separate families prevent stochastic latent error being called geometry.
    """
    model.eval();total=0;sums={};groups={};mus=[];geometry_counts=[0,0,0]
    started=time.perf_counter()
    selected=set(np.random.default_rng(config['seed']+121).choice(loader.dataset.indices,
        size=min(len(loader.dataset),config['evaluation'].get('geometry_samples',256)),replace=False).tolist())
    prior_rng=torch.Generator(device='cuda').manual_seed(config['seed']+122)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(config['seed']+909)
        with attention_context(config),torch.autocast('cuda',dtype=torch.bfloat16):
            for batch in loader:
                g,c=to_device(batch);n=len(g)
                sampled=model.forward_uniform(g,c,sample=True)
                loss=objective(sampled,g,config,context=c)
                mean=model.forward_uniform(g,c,sample=False);raw=mean['raw'].float()
                for k,v in loss.items():sums[k]=sums.get(k,0.)+float(v)*n
                mus.append(mean['mu'].float().cpu())
                # Per-course Euclidean position RMS and proper-frame angle.
                position=(raw[...,:3]-g[...,:3]).square().sum(-1).mean(-1).sqrt()
                with torch.autocast('cuda',enabled=False):
                    relative=rotation6(raw[...,3:9]).transpose(-1,-2)@rotation6(g[...,3:9])
                cosine=((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
                angle=cosine.acos().mean(-1)*180/math.pi
                aperture=(raw[...,9:11].clamp(-5,5).exp()-g[...,9:11].exp()).abs().mean((1,2))
                # Count/context retained: zero-latent ablation measures reliance.
                counts=torch.full((n,),g.shape[1],device=g.device,dtype=torch.long)
                zero,_=model.decode(torch.zeros_like(mean['z']),counts,c,_length=g.shape[1])
                zero_error=(zero[...,:3].float()-g[...,:3]).square().sum(-1).mean(-1).sqrt()
                metrics=torch.stack([position,angle,aperture,zero_error],-1).cpu().numpy()
                for i,row in enumerate(metrics):
                    for key in ('all',f"family_{int(batch['family'][i])}",
                                'independent' if batch['reference'][i]<0 else 'reference_derived'):
                        entry=groups.setdefault(key,{'count':0,'sum':np.zeros(4)})
                        entry['count']+=1;entry['sum']+=row
                if geometry:
                    # Bounded prior audit, same observed counts/context. Not
                    # unconditional sampling and not an MPCC feasibility test.
                    chosen=[i for i,index in enumerate(batch['indices'].tolist()) if index in selected]
                    take=len(chosen)
                    if take:
                        prior_z=torch.randn((take,mean['z'].shape[1]),device='cuda',generator=prior_rng)
                        if hasattr(model,'sample_prior'):prior_z=model.sample_prior(prior_z,counts[chosen],c[chosen])
                        prior,_=model.decode(prior_z,counts[chosen],c[chosen],_length=g.shape[1])
                        for which,values in enumerate((raw[chosen],prior.float())):
                            arr=values.float().cpu().numpy().copy();arr[...,15:17]=values[...,15:17].sigmoid().float().cpu().numpy()
                            for i in range(take):
                                try:
                                    valid=(abs(arr[i,:,9:11])<5).all() and not static_reasons(unpack(arr[i],c[chosen[i]].float().cpu().numpy()))
                                except ValueError:valid=False
                                geometry_counts[which]+=int(valid)
                        geometry_counts[2]+=take
                total+=n
    report={k:v/total for k,v in sums.items()}
    names=['position_rms_m','rotation_deg','aperture_mae_m','zero_latent_position_rms_m']
    report['groups']={key:dict(count=v['count'],**dict(zip(names,(v['sum']/v['count']).tolist()))) for key,v in groups.items()}
    variance=torch.cat(mus).var(0,unbiased=False)
    report.update(courses=total,active_latents=int((variance>.01).sum()),mu_variance_mean=float(variance.mean()),
                  seconds=time.perf_counter()-started)
    if geometry:report['static_geometry']={'samples':geometry_counts[2],
            'mean_reconstruction_valid':geometry_counts[0],'conditional_prior_valid':geometry_counts[1],
            'flight_feasibility':'unknown; static screens only'}
    return report


def save_checkpoint(path,model,optimizer,config,epoch,step,best,wandb_id,exposure_offset=0):
    payload=model.checkpoint();payload.update(training_config=config,optimizer=optimizer.state_dict(),
        epoch=epoch,step=step,best=best,wandb_id=wandb_id,total_exposure_epochs=epoch+exposure_offset,
        rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),
             'numpy':np.random.get_state(),'python':random.getstate()})
    temp=path.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,path)


def initialize_continuation(model,optimizer,config):
    """New experiment/schedule retaining source parameters and AdamW moments.

    Distinct from exact --resume. No silent schema/data/objective changes.
    """
    path=config.get('initialize_from')
    if not path:return None
    source=torch.load(path,map_location='cpu',weights_only=False)
    previous=source['training_config']
    if source['schema']!=VERSION:raise ValueError('initialization schema differs')
    for key in ('model','dataset','data','augmentation'):
        if previous[key]!=config[key]:raise ValueError(f'continuation {key} differs')
    for key in ('beta','weights'):
        if previous['objective'][key]!=config['objective'][key]:raise ValueError(f'continuation objective {key} differs')
    geometry_changed=previous['objective'].get('geometry_weights',{})!=config['objective'].get('geometry_weights',{})
    if geometry_changed and not config.get('allow_geometry_objective_change',False):
        raise ValueError('geometry objective change needs explicit ablation opt-in')
    navigation_changed=previous['objective'].get('navigation',{})!=config['objective'].get('navigation',{})
    if navigation_changed and not config.get('allow_navigation_objective_change',False):
        raise ValueError('navigation objective change needs explicit ablation opt-in')
    for key in ('betas','weight_decay'):
        if previous['training'][key]!=config['training'][key]:raise ValueError(f'continuation optimizer {key} differs')
    model.load_state_dict(source['model']);optimizer.load_state_dict(source['optimizer'])
    total_epoch=source.get('total_exposure_epochs',source['epoch'])
    previous_lineage=Path(path).parent/'lineage.json'
    if 'total_exposure_epochs' not in source and previous_lineage.exists():
        ancestor=json.loads(previous_lineage.read_text())
        total_epoch+=ancestor.get('source_total_epochs',ancestor['source_epoch'])
    return dict(checkpoint=str(path),source_epoch=source['epoch'],source_total_epochs=total_epoch,source_step=source['step'],
                geometry_objective_changed=geometry_changed,
                restored='model and AdamW moments',schedule='new declared experiment-local schedule')


def train(config,output,resume=None):
    output=Path(output)
    if not resume and output.exists():raise FileExistsError(f'Refusing existing run {output}')
    output.mkdir(parents=True,exist_ok=True);atomic_json(output/'config.json',config)
    model,(train_loader,val_loader)=setup(config);t=config['training']
    optimizer=torch.optim.AdamW(model.parameters(),lr=t['lr'],betas=tuple(t['betas']),
                               weight_decay=t['weight_decay'],fused=True)
    epoch0=0;step=0;best=float('inf');wandb_id=None
    lineage=None
    if not resume:lineage=initialize_continuation(model,optimizer,config)
    elif (output/'lineage.json').exists():lineage=json.loads((output/'lineage.json').read_text())
    if lineage:atomic_json(output/'lineage.json',lineage)
    exposure_offset=lineage.get('source_total_epochs',lineage['source_epoch']) if lineage else 0
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved['schema']!=VERSION or saved['training_config']!=config:raise ValueError('resume config/contract differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        epoch0=saved['epoch'];step=saved['step'];best=saved['best'];wandb_id=saved['wandb_id']
        torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python'])
    run=None
    if config.get('wandb',{}).get('enabled',False):
        import wandb
        run=wandb.init(project=config['wandb']['project'],name=config['name'],config=config,
                       dir=str(output),id=wandb_id,resume='must' if wandb_id else None,
                       mode=config['wandb'].get('mode','online'))
        wandb_id=run.id
    started=time.perf_counter();total=t['epochs']*len(train_loader)
    if lineage and not resume:
        initial=evaluate(model,val_loader,config,geometry=False)
        best=initial['loss'];atomic_json(output/'initial_eval.json',initial)
        save_checkpoint(output/'best.pt',model,optimizer,config,0,0,best,wandb_id,exposure_offset)
        save_checkpoint(output/'latest.pt',model,optimizer,config,0,0,best,wandb_id,exposure_offset)
    atomic_json(output/'status.json',dict(state='training',pid=os.getpid(),epoch=epoch0,step=step,wandb_url=run.url if run else None))
    print(json.dumps(dict(event='started',parameters=sum(p.numel() for p in model.parameters()),
        train_courses=len(train_loader.dataset),validation_courses=len(val_loader.dataset),
        batches_per_epoch=len(train_loader),epochs=t['epochs'],attention=t['attention'],amp='bf16',
        wandb_url=run.url if run else None)),flush=True)
    try:
        for epoch in range(epoch0,t['epochs']):
            model.train();train_loader.batch_sampler.set_epoch(epoch+exposure_offset)
            sums={};seen=0;load_seconds=0.;tick=time.perf_counter();epoch_start=tick
            with attention_context(config):
                for batch in train_loader:
                    load_seconds+=time.perf_counter()-tick
                    g,c=to_device(batch);n=len(g);optimizer.zero_grad(set_to_none=True)
                    lr=lr_at(step,total,config)
                    for group in optimizer.param_groups:group['lr']=lr
                    beta=config['objective']['beta']*min(1.,(step+1)/max(1,config['objective']['kl_warmup_steps']))
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        inp=corrupt_encoder(g,**config['augmentation'])
                        out=model.training_forward(inp,g,c) if hasattr(model,'training_forward') else model.forward_uniform(inp,c,sample=True)
                        loss=objective(out,g,config,beta,context=c)
                        if config['objective'].get('manippo'):
                            from .manippo_objectives import auxiliary_losses
                            extra=auxiliary_losses(model,out,g,c,config['objective']['manippo'])
                            loss['loss']=loss['loss']+sum(extra.values())
                            loss.update(extra)
                        if config['objective'].get('navigation'):
                            from starscream.course_navigation.objectives import navigation_losses
                            extra=navigation_losses(model,g,c,config['objective']['navigation'])
                            loss['loss']=loss['loss']+sum(extra.values())
                            loss.update(extra)
                    loss['loss'].backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),t['grad_clip'],error_if_nonfinite=True)
                    optimizer.step()
                    for k,v in loss.items():sums[k]=sums.get(k,torch.zeros((),device=g.device))+v.detach()*n
                    seen+=n;step+=1;tick=time.perf_counter()
            torch.cuda.synchronize();elapsed=time.perf_counter()-epoch_start
            record=dict(epoch=epoch+1,step=step,lr=lr,beta=beta,train={k:float(v)/seen for k,v in sums.items()},
                train_seconds=elapsed,loader_wait_seconds=load_seconds,courses_per_second=seen/elapsed,
                grad_norm=float(norm),peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2)
            record['total_exposure_epochs']=epoch+1+exposure_offset
            do_eval=(epoch+1)%config['evaluation']['every_epochs']==0 or epoch==0 or epoch+1==t['epochs']
            if do_eval:
                result=evaluate(model,val_loader,config,geometry=epoch+1==t['epochs'])
                record['validation']=result
                if result['loss']<best:
                    best=result['loss'];save_checkpoint(output/'best.pt',model,optimizer,config,epoch+1,step,best,wandb_id,exposure_offset)
            if (epoch+1)%t.get('checkpoint_every_epochs',1)==0 or epoch+1==t['epochs']:
                save_checkpoint(output/'latest.pt',model,optimizer,config,epoch+1,step,best,wandb_id,exposure_offset)
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            atomic_json(output/'status.json',dict(state='training',pid=os.getpid(),epoch=epoch+1,step=step,
                                                 best_validation_loss=best,elapsed_seconds=time.perf_counter()-started))
            print(json.dumps(record,allow_nan=False),flush=True)
            if run:
                concise={'epoch':epoch+1,'train/loss':record['train']['loss'],'train/kl':record['train']['kl'],
                    'throughput/courses_per_second':record['courses_per_second'],'train/beta':beta}
                if do_eval:
                    concise.update({'val/loss':result['loss'],'val/position_rms_m':result['groups']['all']['position_rms_m'],
                        'val/rotation_deg':result['groups']['all']['rotation_deg'],'val/active_latents':result['active_latents'],
                        'val/independent_position_rms_m':result['groups']['independent']['position_rms_m'],
                        **({'val/reference_position_rms_m':result['groups']['reference_derived']['position_rms_m']}
                           if 'reference_derived' in result['groups'] else {})})
                run.log(concise,step=step)
        best_saved=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(best_saved['model'])
        result=evaluate(model,val_loader,config,geometry=True)
        result.update(checkpoint='best.pt',epoch=best_saved['epoch'],interpretation='parent-held-out geometry reconstruction; not flight/generalization qualification')
        atomic_json(output/'best_eval.json',result)
        atomic_json(output/'status.json',dict(state='complete',pid=os.getpid(),epoch=t['epochs'],step=step,
            elapsed_seconds=time.perf_counter()-started,best_epoch=best_saved['epoch'],best_validation_loss=best))
    except BaseException as error:
        atomic_json(output/'status.json',dict(state='failed',pid=os.getpid(),step=step,error=repr(error)))
        raise
    finally:
        if run:run.finish()
