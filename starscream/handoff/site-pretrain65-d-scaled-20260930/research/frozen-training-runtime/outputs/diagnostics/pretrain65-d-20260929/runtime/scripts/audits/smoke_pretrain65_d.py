"""Exercise all training/selection courses, CUDA updates, D collection and resume."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/workspace')
OUT=ROOT/'outputs/diagnostics/pretrain65-d-20260929'

def main():
    c=json.loads((ROOT/'configs/exp/v6.21.1.1/pretrain65_d_scaled.json').read_text())
    c['output_root']=str(OUT/'smoke')
    s=c['dagger']
    s.update(run_name='pretrain65-d-smoke', rounds=2, episodes_per_round=65, updates_per_round=3,
        online_replay_capacity=8000, dagger_online_replay_rows_per_round=4000,
        dagger_permanent_expert_capacity=20000, evaluation_suite_episodes_per_track=1,
        top_k=1, mpcc_build_root='/tmp/pretrain65-d-smoke')
    c['checkpoint'].update(run_name=s['run_name'], top_k=1)
    c['wandb'].update(enabled=False, run_name=s['run_name'], name=s['run_name'],
        local_event_path=str(OUT/'smoke.events.jsonl'), local_full_event_path=str(OUT/'smoke.full.events.jsonl'))
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}
    for phase in ('initial','resume'):
        if phase=='resume':
            s.update(rounds=3, resume_checkpoint=str(OUT/'smoke/checkpoints/pretrain65-d-smoke/latest.pt'))
        path=OUT/f'smoke-{phase}.json'
        path.write_text(json.dumps(c,indent=2)+'\n')
        with (OUT/f'smoke-{phase}.log').open('x') as log:
            subprocess.run([sys.executable,'-u','scripts/train_privileged_racing.py','--config',str(path),
                '--stage','dagger','--device','cuda'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (OUT/'smoke.complete.json').write_text(json.dumps(dict(state='passed',finished=time.time()))+'\n')

if __name__=='__main__':main()
