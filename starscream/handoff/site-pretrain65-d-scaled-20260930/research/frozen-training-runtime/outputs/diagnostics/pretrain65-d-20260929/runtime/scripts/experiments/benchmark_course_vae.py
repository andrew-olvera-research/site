"""Bounded real-loader BF16/Flash/fused-AdamW throughput comparison."""
import argparse
import copy
import gc
import time
from pathlib import Path
import torch
import yaml
from starscream.course_model.training import setup,to_device,attention_context,objective,atomic_json
from starscream.course_model.augmentation import corrupt_encoder


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    with open(args.config) as f:base=yaml.safe_load(f)
    reports=[]
    for size in (128,256,512):
        cfg=copy.deepcopy(base);cfg['data']['batch_size']=size
        net,(loader,_)=setup(cfg);opt=torch.optim.AdamW(net.parameters(),lr=3e-4,fused=True)
        iterator=iter(loader);seen=0;load_time=0.;torch.cuda.reset_peak_memory_stats()
        with attention_context(cfg):
            for step in range(40):
                if step==10:torch.cuda.synchronize();start=time.perf_counter()
                before=time.perf_counter()
                try:b=next(iterator)
                except StopIteration:iterator=iter(loader);b=next(iterator)
                if step>=10:load_time+=time.perf_counter()-before;seen+=len(b['gates'])
                g,c=to_device(b);opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    out=net.forward_uniform(corrupt_encoder(g,**cfg['augmentation']),c)
                    loss=objective(out,g,cfg)['loss']
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True);opt.step()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        reports.append(dict(batch_size=size,courses_per_second=seen/elapsed,seconds=elapsed,
            loader_wait_fraction=load_time/elapsed,peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,
            attention='forced FlashAttention; fallback disabled',amp='bf16'))
        del net,opt,loader,iterator,g,c,out,loss;gc.collect();torch.cuda.empty_cache()
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,reports);print(reports)


if __name__=='__main__':main()
