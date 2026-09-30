"""Real MPCC/DAgger collection comparison; no training or production writes."""
import argparse,json,sys,time,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts import train_privileged_racing as t


def memory():
    info={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line and line.split()[1].isdigit()}
    vm=dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return dict(available=info['MemAvailable'],swap_used=info['SwapTotal']-info['SwapFree'],
                pswpin=int(vm['pswpin']),pswpout=int(vm['pswpout']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=65)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--variants',default='1:0,4:0.2,8:0.5,1:0')
    p.add_argument('--ready-file',type=Path)
    p.add_argument('--start-file',type=Path)
    p.add_argument('--warmup-episodes',type=int,default=0)
    args=p.parse_args()
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    cfg=t.load_config(Path('configs/exp/v6.21.1/update_fix_dagger.yaml'))
    settings=cfg['dagger'].copy(); settings['rollout_envs']=args.workers
    policy,norm,_,_=t.load_policy_checkpoint(settings['resume_checkpoint'],'cuda')
    policy.eval()
    curriculum=settings['curriculum']
    stage=t.parse_stage(curriculum[-1] if isinstance(curriculum,list) else curriculum)
    collector=t.ProcessDaggerCollector(policy,norm,settings,stage,'cuda')
    results=[]
    try:
        if args.warmup_episodes:
            warmup=collector.collect(episodes=args.warmup_episodes,beta=.35,seed_base=2036092402)
            del warmup
            gc.collect()
            torch.cuda.synchronize()
        if args.ready_file:
            if args.start_file is None:
                raise ValueError('--ready-file requires --start-file')
            args.ready_file.touch()
            deadline=time.monotonic()+300
            while not args.start_file.exists():
                if time.monotonic()>deadline:
                    raise TimeoutError('Benchmark start gate timed out')
                time.sleep(.01)
        for spec in args.variants.split(','):
            parts=spec.split(':')
            minimum,wait=parts[:2]
            collector.settings.update(dagger_async_min_inference_batch=int(minimum),dagger_async_batch_wait_ms=float(wait))
            collector.settings['dagger_dispatch_teacher_first'] = len(parts)>2 and parts[2]=='early'
            collector.settings['dagger_chunked_labels'] = len(parts)>2 and 'chunk' in parts[2]
            collector.settings['dagger_event_inference'] = len(parts)>2 and 'event' in parts[2]
            before=memory(); start=time.perf_counter()
            batch=collector.collect(episodes=args.episodes,beta=.35,seed_base=2036092403)
            elapsed=time.perf_counter()-start
            result=dict(variant=spec,seconds=elapsed,steps=batch.environment_steps,
                steps_s=batch.environment_steps/elapsed,labels=len(batch.histories),
                accepted=batch.accepted_episodes,rejected=batch.rejected_episodes,
                solver_failure_fraction=batch.solver_failures/max(batch.total_queries,1),
                teacher_wall_seconds=batch.teacher_wall_seconds,
                teacher_solve_seconds=batch.teacher_solve_seconds,feature_seconds=batch.feature_seconds,
                controller_wall_breakdown=batch.controller_wall_breakdown.tolist(),
                backend_wall_breakdown=batch.backend_wall_breakdown.tolist(),
                env_seconds=batch.environment_step_seconds,host=collector.last_profile,
                memory_before=before,memory_after=memory(),workers=args.workers)
            print(json.dumps(result),flush=True);results.append(result)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(results,indent=2)+'\n')
            del batch;gc.collect()
    finally: collector.close()


if __name__=='__main__':main()
