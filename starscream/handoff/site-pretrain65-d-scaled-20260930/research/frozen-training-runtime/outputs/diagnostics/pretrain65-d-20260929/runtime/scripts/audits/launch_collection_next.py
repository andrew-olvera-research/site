"""Single-use sequential queue; no retries, scheduler or monitoring automation."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/workspace')
OUT = ROOT/'outputs/diagnostics/collection-next-20260929'
RUNTIME = OUT/'runtime'

def save(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)

def verify():
    report=json.loads((OUT/'validation.json').read_text())
    if report['state'] != 'passed': raise RuntimeError('Validation not passed')
    if json.loads((OUT/'eval.status.json').read_text())['state'] != 'complete':
        raise RuntimeError('Fresh evaluations must finish before training')
    for p, digest in report['config_sha256'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest: raise RuntimeError(f'Config changed: {p}')
    for p, digest in json.loads((OUT/'runtime-sha256.json').read_text()).items():
        if hashlib.sha256((RUNTIME/p).read_bytes()).hexdigest()!=digest: raise RuntimeError(f'Runtime changed: {p}')

def main():
    os.chdir(ROOT)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=OUT/'queue.status.json'
        if state_path.exists(): raise FileExistsError('Queue already reserved; inspect status')
        verify()
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit(): continue
            try: args=(proc/'cmdline').read_bytes().split(b'\0')
            except OSError: continue
            if any(a.endswith(b'/train_privileged_racing.py') for a in args):
                raise RuntimeError(f'Trainer already active: {proc.name}')
        jobs=json.loads((OUT/'jobs.json').read_text())
        for job in jobs:
            directory=ROOT/'outputs/checkpoints'/job['run_name']
            if directory.exists() or (OUT/f"{job['arm']}.status.json").exists():
                raise FileExistsError(f"Existing artifacts: {job['arm']}")
        state=dict(state='running', pid=os.getpid(), started=time.time(), order=[j['arm'] for j in jobs], completed=[])
        save(state_path,state)
        for job in jobs: save(OUT/f"{job['arm']}.status.json",dict(**job,state='queued'))
        try:
            for job in jobs:
                verify()
                if shutil.disk_usage(ROOT).free < 10*2**30: raise RuntimeError('Less than 10 GiB free; queue stopped')
                arm=job['arm']; state['active_arm']=arm; save(state_path,state)
                status=dict(**job,state='running',started=time.time())
                log_path=OUT/f'{arm}.console.log'
                with log_path.open('x') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(RUNTIME/'scripts/train_privileged_racing.py'),
                        '--config',job['config'],'--stage','dagger','--device','cuda'],cwd=ROOT,
                        env={**os.environ,'PYTHONPATH':str(RUNTIME),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                             'OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'},stdout=log,stderr=subprocess.STDOUT)
                status.update(pid=child.pid,log=str(log_path)); save(OUT/f'{arm}.status.json',status)
                code=child.wait()
                if code: raise RuntimeError(f'Arm {arm} exited {code}')
                import torch
                directory=ROOT/'outputs/checkpoints'/job['run_name']
                payload=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
                if payload['round']!=24: raise RuntimeError(f'Incomplete checkpoint: {arm}')
                status.update(state='complete',exit_code=0,round=24,steps=payload['environment_steps'],finished=time.time())
                del payload
                if len(list(directory.glob('*.pt')))>2: raise RuntimeError(f'Checkpoint retention exceeded: {arm}')
                save(OUT/f'{arm}.status.json',status)
                subprocess.run([sys.executable,str(RUNTIME/'scripts/audits/cleanup_collection_storage.py'),
                    '--run',job['run_name'],'--apply'],cwd=ROOT,check=True)
                state['completed'].append(arm); save(state_path,state)
            state.update(state='complete',finished=time.time(),active_arm=None); save(state_path,state)
        except BaseException as exc:
            state.update(state='failed',error=repr(exc),finished=time.time()); save(state_path,state)
            if 'status' in locals():
                status.update(state='failed',error=repr(exc),finished=time.time())
                save(OUT/f'{arm}.status.json',status)
            raise

if __name__=='__main__': main()
