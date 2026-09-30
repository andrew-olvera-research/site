"""Fresh-seed final-best evaluation; no checkpoint re-selection on these seeds."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json

ROOT=Path('/workspace')
OUT=ROOT/'outputs/evals/v6201-final-best-r221'
CONFIG=ROOT/'configs/exp/v6.20.1/transition_corpus_dagger_r240.yaml'
CHECKPOINT=ROOT/'outputs/checkpoints/starscream-v6.20.1-transition-corpus-dagger-r240/best-step-044144097-full_course_success-0.70714286.pt'


def job(spec):
    split,mode,count=spec
    name=f'{split}-{mode}'
    output=OUT/f'{name}.json'
    if output.exists():raise FileExistsError(output)
    cmd=[sys.executable,'-u','scripts/eval_privileged_dagger.py','--config',str(CONFIG),
         '--checkpoint',str(CHECKPOINT),'--curriculum',split,'--episodes',str(count),
         '--seed','2044091462','--matched-track-seeds','--workers','6',
         '--device','cuda','--output',str(output),
         '--nominal-environment' if mode=='nominal' else '--randomized-environment']
    with (OUT/f'{name}.log').open('w') as f:
        subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    report=json.loads(output.read_text())
    return dict(name=name,output=str(output),success=report['metrics']['full_course_success'])


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    specs=[('reporting','nominal',64),('validation','randomized',32),
           ('reporting','randomized',64),('validation','nominal',32),('training','randomized',16)]
    done=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,s) for s in specs]):
            r=f.result();done.append(r)
            atomic_json(OUT/'status.json',dict(complete=len(done)==len(specs),records=done))
            print(r,flush=True)
