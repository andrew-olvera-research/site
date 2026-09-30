"""Matched real replay rows and optimizer states; bounded DAgger update benchmark."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch,h5py
from scripts import train_privileged_racing as t
from starscream.training_acceleration import configure_policy_acceleration
from starscream.dagger_throughput import HierarchicalReplayPlan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['baseline','static','compiled'],required=True)
    p.add_argument('--steps',type=int,default=180)
    args=p.parse_args()
    torch.set_num_threads(1); torch.set_float32_matmul_precision('high')
    import torch._inductor.config as config
    config.compile_threads=1
    root=Path('outputs/checkpoints/starscream-v6.21.1.update-fix-pretrain65-dagger')
    policy,_,payload,_=t.load_policy_checkpoint(root/'latest.pt','cuda')
    settings=copy.deepcopy(payload['training_config']['dagger'])
    with h5py.File(root/'dagger-replay/round-00218.h5') as f:
        data={k:v[:] for k,v in f['online'].items()}
    if args.variant!='baseline':
        settings['dagger_statistics_group_count']=int(data['groups'].max())+1
    if args.variant=='compiled': settings['compile_dagger_loss']=True
    configure_policy_acceleration(policy,settings)
    fn=torch.compile(t.imitation_loss,fullgraph=True,dynamic=False,
        options={'triton.cudagraphs':False}) if args.variant=='compiled' else t.imitation_loss
    optimizer=torch.optim.AdamW(policy.parameters(),lr=float(settings['learning_rate']),fused=True)
    optimizer.load_state_dict(payload['optimizer'])
    plan=HierarchicalReplayPlan(*(data[k] for k in ['families','tracks','gate_indices','events','teacher_modes','occupancy_modes']),1536)
    rng=np.random.default_rng(2036092404); torch.manual_seed(2036092404)
    digest=hashlib.sha256(); policy.train(); elapsed=[]; losses=[]
    torch.cuda.reset_peak_memory_stats()
    for i in range(args.steps):
        torch.cuda.synchronize(); start=time.perf_counter()
        indices=plan.sample(rng); digest.update(indices.tobytes())
        batch={k:torch.from_numpy(v[indices].astype(np.float32) if k=='histories' else v[indices]).cuda() for k,v in data.items()
               if k in ['histories','actions','previous','dynamics','dynamics_valid','speed_commands','groups']}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,pieces=fn(policy,batch['histories'],batch['actions'],batch['previous'],batch['dynamics'],settings,
                batch['dynamics_valid'],batch['speed_commands'],group_ids=batch['groups'])
        loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(),float(settings['gradient_clip']))
        optimizer.step();torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-start);losses.append(float(loss.detach()))
        if i in [0,19,args.steps-1]:print(args.variant,i,elapsed[-1],losses[-1],flush=True)
    output=Path('outputs/dagger-throughput');output.mkdir(parents=True,exist_ok=True)
    result=dict(variant=args.variant,steps=args.steps,cold_seconds=elapsed[0],
        warm_seconds=float(np.mean(elapsed[20:])),losses=losses,index_digest=digest.hexdigest(),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        note='Real final online shard, hierarchical draws; isolates updates, not full online/permanent replay mixture.')
    (output/f'updates-{args.variant}.json').write_text(json.dumps(result,indent=2))
    torch.save(policy.state_dict(),output/f'updates-{args.variant}-parameters.pt')
    print({k:v for k,v in result.items() if k!='losses'},flush=True)


if __name__=='__main__':main()
