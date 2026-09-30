"""Launch one scratch mini-slalom leg with a durable status and console log."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--courses',type=int,choices=(2,4,6))
    parser.add_argument('--augmented',action='store_true')
    parser.add_argument('--kind',choices=('broad','bounded'),required=True)
    args=parser.parse_args()
    if not args.augmented and args.courses is None:
        parser.error('--courses is required unless --augmented is set')
    stem=(f'mini_slalom_aug2_{args.kind}' if args.augmented
          else f'mini_slalom_{args.courses}_{args.kind}')
    config=ROOT/f'configs/exp/v6.21.1.1/{stem}.yaml'
    status=ROOT/f'outputs/diagnostics/mini-slalom-recovery-baselines/{stem}.status.json'
    log=ROOT/f'outputs/logs/{stem}.console.log'
    if not config.exists() or status.exists() or log.exists():
        raise RuntimeError('Missing config or an existing run status/log for '+stem)
    status.parent.mkdir(parents=True,exist_ok=True)
    run=dict(name=stem,config=str(config.relative_to(ROOT)),log=str(log.relative_to(ROOT)),
             state='launching',started_unix=time.time())
    status.write_text(json.dumps(run,indent=2)+'\n')
    command=['/opt/conda/bin/python','-u','scripts/train_privileged_racing.py',
             '--config',str(config.relative_to(ROOT)),'--stage','dagger','--device','cuda']
    with log.open('w') as stream:
        p=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                           start_new_session=True,env={**os.environ,'PYTHONUNBUFFERED':'1'})
        run.update(state='running',pid=p.pid)
        status.write_text(json.dumps(run,indent=2)+'\n')
        code=p.wait()
    run.update(state='complete' if code==0 else 'failed',exit_code=code,finished_unix=time.time())
    status.write_text(json.dumps(run,indent=2)+'\n')
    return code

if __name__=='__main__':sys.exit(main())
