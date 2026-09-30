"""Measure SDPA dispatch and unchanged-policy batch-shape drift on real states."""
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from scripts import train_privileged_racing as t


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--rollout',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    policy,_,payload,_=t.load_policy_checkpoint(args.checkpoint,'cuda')
    policy.set_exact_likelihood_mode(True); policy.eval()
    data=torch.load(args.rollout,map_location='cpu',weights_only=False)['rollout']
    selection=torch.linspace(0,len(data['history'])-1,8192,dtype=torch.long)
    histories=data['history'][selection].cuda(); speeds=data['speed_command'][selection].cuda()
    correlation=payload['training_config']['ppo']['ppo_exploration_correlation']
    std=policy.log_std_parameter.exp()*(1-correlation**2)**.5
    result={}
    original=policy.step_embedding.forward
    for mode in ['fp32','bf16_backbone','bf16_all','fp16_backbone','tf32']:
        policy.step_embedding.forward=original
        torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'
        torch.set_float32_matmul_precision('high' if mode == 'tf32' else 'highest')
        if mode in {'bf16_backbone','fp16_backbone'}:
            def backbone(*a,**kw):
                with torch.autocast('cuda',dtype=torch.bfloat16 if mode=='bf16_backbone' else torch.float16):
                    return original(*a,**kw).float()
            policy.step_embedding.forward=backbone
        context=lambda: torch.autocast('cuda',dtype=torch.bfloat16) if mode=='bf16_all' else nullcontext()
        with torch.no_grad(),context():
            small=torch.cat([policy.distribution(h,s).location.float()
                for h,s in zip(histories.split(280),speeds.split(280))])
            large=policy.distribution(histories,speeds).location.float()
            torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(10): policy.distribution(histories,speeds)
            torch.cuda.synchronize()
            elapsed=(time.perf_counter()-start)/10
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                policy.distribution(histories,speeds)
            error=(large-small)
            row=dict(forward_seconds_8192=elapsed,max_location_difference=float(error.abs().max()),
                mean_behavior_kl=float((.5*(error/std).square()).sum(-1).mean()),
                max_behavior_kl=float((.5*(error/std).square()).sum(-1).max()),
                attention_ops=[e.key for e in prof.key_averages() if 'attention' in e.key])
            if mode=='fp32': reference=small.clone()
            else:
                row['kl_vs_fp32']=float((.5*((small-reference)/std).square()).sum(-1).mean())
            result[mode]=row
            print(mode,row,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
