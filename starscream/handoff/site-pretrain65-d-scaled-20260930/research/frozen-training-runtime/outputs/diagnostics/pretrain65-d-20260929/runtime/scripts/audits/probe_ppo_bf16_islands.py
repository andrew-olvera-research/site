"""Locate BF16 batch drift and test FP32 islands on frozen real PPO states."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from scripts import train_privileged_racing as t
from starscream.sequence_architecture import FixedRandomFourierFeatures, SwiGLUProjection


def precision_wrapper(original, dtype):
    def wrapped(*args, **kwargs):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dtype=='bf16'):
            return original(*args,**kwargs).float()
    return wrapped


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--rollout',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',default='')
    p.add_argument('--evaluate',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(args.rollout,map_location='cpu',weights_only=False)['rollout']
    idx=torch.linspace(0,len(data['history'])-1,7680,dtype=torch.long)
    h=data['history'][idx].cuda(); s=data['speed_command'][idx].cuda()
    modes=['fp32','bf16','bf16_no_reduced','fp32_geometry_fourier',
           'fp32_projections','fp32_attention','bf16_ffn_only']
    if args.modes: modes=args.modes.split(',')
    results={}
    for mode in modes:
        policy,normalizer,payload,_=t.load_policy_checkpoint(args.checkpoint,'cuda')
        policy.set_exact_likelihood_mode(True); policy.eval()
        backbone=policy.step_embedding
        if mode!='fp32':
            backbone.forward=precision_wrapper(backbone.forward,'bf16' if mode!='bf16_ffn_only' else 'fp32')
        if mode=='stable_bf16':
            from scripts.audits.stable_bf16_probe_kernel import install
            install(backbone)
        if mode in ['fp32_geometry_fourier','fp32_projections','fp32_attention']:
            backbone.geometric_routes=precision_wrapper(backbone.geometric_routes,'fp32')
            for module in backbone.modules():
                if isinstance(module,FixedRandomFourierFeatures) or (
                    mode in ['fp32_projections','fp32_attention'] and isinstance(module,SwiGLUProjection)):
                    module.forward=precision_wrapper(module.forward,'fp32')
        if mode=='fp32_attention':
            for block in backbone.blocks:
                block.attention.forward=precision_wrapper(block.attention.forward,'fp32')
        if mode=='bf16_ffn_only':
            for name,module in backbone.named_modules():
                if isinstance(module,torch.nn.Linear) and 'feedforward_' in name:
                    module.forward=precision_wrapper(module.forward,'bf16')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=mode!='bf16_no_reduced'
        traces={}; hooks=[]
        for name,module in backbone.named_modules():
            if isinstance(module,(torch.nn.Linear,FixedRandomFourierFeatures)):
                def hook(mod,inputs,output,name=name):
                    traces[name]=output[:280].detach().float().clone()
                hooks.append(module.register_forward_hook(hook))
        with torch.no_grad():
            policy.distribution(h[:280],s[:280]); small_trace=traces.copy()
            policy.distribution(h,s); large_trace=traces.copy()
        for hook in hooks: hook.remove()
        layer_errors={k:float((v-large_trace[k]).abs().max()) for k,v in small_trace.items()}
        with torch.no_grad():
            small=torch.cat([policy.distribution(a,b).location.float() for a,b in zip(h.split(280),s.split(280))])
            large=policy.distribution(h,s).location.float()
            torch.cuda.synchronize(); start=time.perf_counter()
            for _ in range(5): policy.distribution(h,s)
            torch.cuda.synchronize(); seconds=(time.perf_counter()-start)/5
        with torch.enable_grad():
            update=policy.distribution(h,s).location.detach().float()
        std=policy.log_std_parameter.detach().exp()*(1-payload['training_config']['ppo']['ppo_exploration_correlation']**2)**.5
        kl=lambda a,b: float((.5*((a-b)/std).square()).sum(-1).mean())
        if mode=='fp32': reference=small
        row=dict(batch_kl=kl(small,large),update_kl=kl(small,update),kl_vs_fp32=kl(small,reference),
                 max_location_error=float((small-large).abs().max()),seconds=seconds,layer_errors=layer_errors)
        if args.evaluate:
            settings=payload['training_config']['ppo']
            curriculum=settings['evaluation_curriculum']
            stage=t.parse_stage(curriculum[0] if isinstance(curriculum,list) else curriculum)
            evaluation=t.evaluate_policy(policy,normalizer,settings,stage,count=100,
                seed_base=int(settings.get('evaluation_seed',20261000)),device='cuda')
            row['evaluation']={k:v for k,v in evaluation.items() if k in
                ['full_course_success','successful_lap_time_seconds','crash_rate','episodes']}
        results[mode]=row
        print(mode,{k:v for k,v in row.items() if k!='layer_errors'},flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        del policy,backbone,small_trace,large_trace,traces
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True


if __name__=='__main__':main()
