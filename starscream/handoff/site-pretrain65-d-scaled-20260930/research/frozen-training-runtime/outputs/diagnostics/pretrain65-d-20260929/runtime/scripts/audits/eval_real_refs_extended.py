"""Extended fresh-seed real-reference cohorts (same seed sequence as eval_rl_final; superset of the 64-episode cohorts)."""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from scripts.audits.eval_rl_final import ROOT, CONFIG, SEED

def job(out,checkpoint,mode,count,workers):
    name=f'reporting-{mode}-e{count}'
    output=out/f'{name}.json'
    if output.exists():raise FileExistsError(output)
    cmd=[sys.executable,'-u','scripts/eval_privileged_dagger.py','--config',str(CONFIG),'--checkpoint',str(checkpoint),
         '--section','dagger','--curriculum','reporting','--episodes',str(count),'--seed',SEED,'--matched-track-seeds',
         '--workers',str(workers),'--device','cuda','--output',str(output),
         '--nominal-environment' if mode=='nominal' else '--randomized-environment']
    with (out/f'{name}.log').open('w') as f:
        subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    return dict(name=name,success=json.loads(output.read_text())['metrics']['full_course_success'])

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs',nargs='+',required=True,help='out_dir=checkpoint')
    ap.add_argument('--episodes',type=int,default=256)
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--parallel',type=int,default=2)
    a=ap.parse_args()
    jobs=[]
    for s in a.runs:
        out,ck=s.split('=',1); out=Path(out); out.mkdir(parents=True,exist_ok=True)
        assert Path(ck).is_file(),ck
        for mode in ('randomized','nominal'): jobs.append((out,Path(ck),mode))
    done=[]
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        for f in as_completed([pool.submit(job,o,c,m,a.episodes,a.workers) for o,c,m in jobs]):
            done.append(f.result()); print(done[-1],flush=True)
    print('EXTENDED-DONE',flush=True)
