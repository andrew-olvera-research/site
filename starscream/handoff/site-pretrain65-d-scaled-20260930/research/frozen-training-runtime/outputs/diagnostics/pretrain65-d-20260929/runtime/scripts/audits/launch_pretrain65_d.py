"""Launch one validated scaled-D experiment; no retries or automatic restart."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path('/workspace')
OUT=ROOT/'outputs/diagnostics/pretrain65-d-20260929'
CONFIG=ROOT/'configs/exp/v6.21.1.1/pretrain65_d_scaled.json'

def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def main():
    os.chdir(ROOT)
    with (OUT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status_path=OUT/'train.status.json'
        if status_path.exists():raise FileExistsError('Run already reserved')
        report=json.loads((OUT/'validation.json').read_text())
        assert report['state']=='passed'
        for path,digest in report['sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Validated input changed: {path}')
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:args=(proc/'cmdline').read_bytes().split(b'\0')
            except OSError:continue
            if any(a.endswith(b'train_privileged_racing.py') for a in args):raise RuntimeError(f'Trainer active: {proc.name}')
        c=json.loads(CONFIG.read_text());name=c['dagger']['run_name']
        if (ROOT/'outputs/checkpoints'/name).exists():raise FileExistsError(name)
        for key in ('local_event_path','local_full_event_path'):
            if Path(c['wandb'][key]).exists():raise FileExistsError(c['wandb'][key])
        if shutil.disk_usage(ROOT).free<80*2**30:raise RuntimeError('Less than 80 GiB free')
        runtime=OUT/'runtime'
        for folder in ('scripts','starscream'):
            shutil.copytree(ROOT/folder,runtime/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        save(OUT/'runtime-sha256.json',{str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime.rglob('*') if p.is_file()})
        frozen=OUT/'production-config.json';shutil.copyfile(CONFIG,frozen)
        state=dict(state='starting',launcher_pid=os.getpid(),run_name=name,started=time.time(),config=str(frozen),runtime=str(runtime))
        save(status_path,state)
        try:
            log_path=OUT/'train.console.log'
            with log_path.open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(runtime/'scripts/train_privileged_racing.py'),
                    '--config',str(frozen),'--stage','dagger','--device','cuda'],cwd=ROOT,
                    env={**os.environ,'PYTHONPATH':str(runtime),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                         'OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'},stdout=log,stderr=subprocess.STDOUT)
            state.update(state='running',pid=child.pid,log=str(log_path));save(status_path,state)
            code=child.wait()
            if code:raise RuntimeError(f'Trainer exited {code}')
            import torch
            checkpoint=torch.load(ROOT/'outputs/checkpoints'/name/'latest.pt',map_location='cpu',weights_only=False)
            if checkpoint['round']!=218:raise RuntimeError('Trainer ended before round 218')
            state.update(state='complete',exit_code=0,round=218,finished=time.time());save(status_path,state)
        except BaseException as exc:
            state.update(state='failed',error=repr(exc),finished=time.time());save(status_path,state);raise

if __name__=='__main__':main()
