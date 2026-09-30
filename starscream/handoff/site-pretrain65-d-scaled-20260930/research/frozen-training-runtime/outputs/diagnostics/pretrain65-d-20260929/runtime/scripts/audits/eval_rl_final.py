"""Fresh-seed final evaluation of any actor checkpoint under the v6.20.1 protocol.

Same seeds, cohorts, config section and worker count as
``scripts/audits/eval_v6201_final.py`` so results are directly comparable with
``outputs/evals/v6201-final-best-r221``. No checkpoint re-selection on these seeds.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json

ROOT=Path('/workspace')
CONFIG=ROOT/'configs/exp/v6.20.1/transition_corpus_dagger_r240.yaml'
SEED='2044091462'
SPECS=[('reporting','nominal',64),('validation','randomized',32),
       ('reporting','randomized',64),('validation','nominal',32),('training','randomized',16)]


def job(out,checkpoint,spec,workers):
    split,mode,count=spec
    name=f'{split}-{mode}'
    output=out/f'{name}.json'
    if output.exists():raise FileExistsError(output)
    cmd=[sys.executable,'-u','scripts/eval_privileged_dagger.py','--config',str(CONFIG),
         '--checkpoint',str(checkpoint),'--section','dagger','--curriculum',split,
         '--episodes',str(count),'--seed',SEED,'--matched-track-seeds','--workers',str(workers),
         '--device','cuda','--output',str(output),
         '--nominal-environment' if mode=='nominal' else '--randomized-environment']
    with (out/f'{name}.log').open('w') as f:
        subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    report=json.loads(output.read_text())
    return dict(name=name,output=str(output),success=report['metrics']['full_course_success'])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--parallel',type=int,default=2)
    a=ap.parse_args()
    assert a.checkpoint.is_file(),a.checkpoint
    a.out.mkdir(parents=True,exist_ok=True)
    atomic_json(a.out/'provenance.json',dict(checkpoint=str(a.checkpoint),config=str(CONFIG),
        seed=int(SEED),section='dagger',workers=a.workers,specs=SPECS))
    done=[]
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        for f in as_completed([pool.submit(job,a.out,a.checkpoint,s,a.workers) for s in SPECS]):
            r=f.result();done.append(r)
            atomic_json(a.out/'status.json',dict(complete=len(done)==len(SPECS),records=done))
            print(r,flush=True)
