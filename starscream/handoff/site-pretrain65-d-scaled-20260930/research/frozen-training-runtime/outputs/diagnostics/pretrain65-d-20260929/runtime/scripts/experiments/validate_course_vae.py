"""Full corpus read audit plus bounded BF16 optimization/loader diagnostics."""
import argparse
import json
import time
import resource
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from starscream.course_model.data import CourseDataset,collate_courses
from starscream.course_model.model import CourseVAE,vae_loss,rotation6
from starscream.course_model.schema import unpack,static_reasons


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(817)
    report={}
    with h5py.File(args.dataset,'r') as f:
        offsets=f['offsets'][:];count=len(f['context'])
        assert offsets[0]==0 and offsets[-1]==len(f['gates']) and (np.diff(offsets)>=4).all()
        assert len(set(f['fingerprint'][:]))==count
        assert (f['feasibility'][:]==-1).all()
        nbytes=sum(x.size*x.dtype.itemsize for x in f.values() if isinstance(x,h5py.Dataset))
        failures=[]
        for i in range(count):
            g=f['gates'][offsets[i]:offsets[i+1]];c=f['context'][i]
            t=unpack(g,c)
            reasons=static_reasons(t)
            if reasons:failures.append((i,reasons))
        assert not failures,failures[:10]
        report.update(courses=count,decoded_static_failures=len(failures),raw_array_bytes=nbytes,
            disk_bytes=args.dataset.stat().st_size,compression_ratio=nbytes/args.dataset.stat().st_size)
    ds=CourseDataset(args.dataset,split=0)
    # Open in parent before forking too: Dataset must reopen process-local HDF5.
    ds[0]
    loader=DataLoader(ds,batch_size=64,num_workers=2,shuffle=True,collate_fn=collate_courses)
    start=time.perf_counter();seen=0
    for i,b in enumerate(loader):
        seen+=len(b['indices'])
        if i==39:break
    report['loader_courses_per_second']=seen/(time.perf_counter()-start)
    model=CourseVAE().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    # Fixed mini-corpus overfit proves gradient/magnitude behavior, NOT a trained
    # generator or coverage evidence. No long experiment is launched here.
    batch=collate_courses([ds[i*17] for i in range(16)])
    g=batch['gates'].cuda();m=batch['mask'].cuda();c=batch['context'].cuda()
    losses=[];start=time.perf_counter()
    for step in range(160):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(g,m,c,sample=True);loss=vae_loss(out,g,m,beta=.0001)
        assert torch.isfinite(loss['loss'])
        loss['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        losses.append(float(loss['loss'].detach()))
    model.eval()
    with torch.no_grad():
        mu,_=model.encode(g,m,c);base,_=model.decode(mu,m.sum(1),c);changed,_=model.decode(mu+1,m.sum(1),c)
        dependence=float((base[m]-changed[m]).abs().mean())
    assert np.mean(losses[-10:])<.6*np.mean(losses[:10]),losses[-10:]
    assert dependence>.001
    report.update(parameters=sum(p.numel() for p in model.parameters()),bf16_steps=160,
        initial_loss_mean=float(np.mean(losses[:10])),final_loss_mean=float(np.mean(losses[-10:])),
        latent_perturbation_mean_change=dependence,optimization_seconds=time.perf_counter()-start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        interpretation='short numerical overfit only; no pretrained model or feasibility/compass claim')
    (args.dataset.parent/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
