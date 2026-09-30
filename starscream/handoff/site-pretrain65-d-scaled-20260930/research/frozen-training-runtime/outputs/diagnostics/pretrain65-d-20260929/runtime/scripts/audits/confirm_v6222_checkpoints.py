"""Independent-seed finalist evaluation; uncertainty is not hidden by top-k rank.

Run after training. Equal deterministic episode schedules make comparisons
matched within this panel. Course-cluster bootstrap estimates generalization
uncertainty across these development courses, not independent episode precision.
All pairwise intervals use Bonferroni correction, including the selected winner.
"""
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/exp/v6.22.2/pretrain45_dagger.yaml'
NAME='starscream-v6.22.2-pretrain45'


def paired_intervals(scores,seed=622232,b=20000,alpha=.05):
    names=list(scores)
    arrays=np.asarray([scores[n] for n in names],float)
    assert arrays.ndim==2 and arrays.shape[1]==45 and np.isfinite(arrays).all()
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,arrays.shape[1],size=(b,arrays.shape[1]))
    means=arrays[:,draws].mean(axis=2)
    pairs=list(combinations(range(len(names)),2))
    tail=alpha/(2*max(len(pairs),1))
    return [dict(a=names[i],b=names[j],delta=float((arrays[i]-arrays[j]).mean()),
                 simultaneous_95_interval=np.quantile(means[i]-means[j],[tail,1-tail]).tolist())
            for i,j in pairs]


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes-per-course',type=int,default=32)
    parser.add_argument('--evaluate-checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.episodes_per_course<32:raise ValueError('confirmation requires at least 32 episodes/course')
    if args.evaluate_checkpoint:
        sys.path.insert(0,str(ROOT))
        from scripts.train_privileged_racing import ProcessRaceCollector, parse_stage, multitrack_metrics
        from starscream.privileged_racing import load_policy_checkpoint
        cfg=json.loads(CONFIG.read_text()); settings=cfg['dagger']
        policy,normalizer,payload,resolved=load_policy_checkpoint(args.evaluate_checkpoint,'cuda')
        stage=parse_stage(settings['evaluation_curriculum'])
        collector=ProcessRaceCollector(policy,normalizer,settings,stage,'cuda',workers=32)
        try:
            rows=collector.evaluate_rows(episodes=45*args.episodes_per_course,seed_base=2046091922)
        finally:
            collector.close()
        report=dict(checkpoint=str(resolved),checkpoint_environment_steps=payload['environment_steps'],
                    episodes=len(rows),seed=2046091922,rows=rows,
                    metrics=multitrack_metrics(rows,stage.target_gates*stage.rollout_laps))
        args.output.write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
        return
    checkpoint=ROOT/'outputs/checkpoints'/NAME
    entries=json.loads((checkpoint/'top-k.json').read_text())['checkpoints']
    paths=[checkpoint/Path(e['path']) for e in entries]
    latest=checkpoint/'latest.pt'
    if latest.exists():paths.append(latest)
    paths=list(dict.fromkeys(paths))
    out=ROOT/'outputs/evals'/NAME/'confirmation'
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(CONFIG.read_text())
    courses=[Path(p).stem for p in cfg['dagger']['evaluation_curriculum']['tracks']]
    results=[];scores={}; outcomes={}
    for path in paths:
        if not path.is_absolute():path=ROOT/path
        target=out/(path.stem+f'-e{args.episodes_per_course}.json')
        if not target.exists():
            subprocess.run([sys.executable,str(Path(__file__).resolve()),
                '--evaluate-checkpoint',str(path),'--episodes-per-course',str(args.episodes_per_course),
                '--output',str(target)],check=True,cwd=ROOT)
        d=json.loads(target.read_text());m=d['metrics']
        assert d['seed']==2046091922 and d['episodes']==45*args.episodes_per_course
        assert all(m[f'track/{c}/episodes']==args.episodes_per_course for c in courses)
        values=[m[f'track/{c}/full_course_success'] for c in courses]
        scores[path.name]=values
        outcomes[path.name]={(r['track'],int(r['episode_seed'])):bool(r['gates']>=r['target_gates'] and not r['crashed']) for r in d['rows']}
        assert len(outcomes[path.name])==45*args.episodes_per_course
        results.append(dict(checkpoint=str(path),macro_sr=float(np.mean(values)),
                            step=d['checkpoint_environment_steps'],report=str(target)))
    comparisons=paired_intervals(scores)
    from scipy.stats import binomtest
    for pair in comparisons:
        a,b=outcomes[pair['a']],outcomes[pair['b']]
        assert a.keys()==b.keys()
        win=sum(a[k] and not b[k] for k in a)
        loss=sum(b[k] and not a[k] for k in a)
        pair.update(paired_wins=win,paired_losses=loss,
                    exact_mcnemar_p=float(binomtest(win,win+loss,.5).pvalue) if win+loss else 1.)
    prior=0.
    for i,pair in enumerate(sorted(comparisons,key=lambda x:x['exact_mcnemar_p'])):
        prior=max(prior,min(1.,(len(comparisons)-i)*pair['exact_mcnemar_p']))
        pair['holm_adjusted_p']=prior
    ranked=sorted(results,key=lambda r:(r['macro_sr'],r['step']),reverse=True)
    winner=Path(ranked[0]['checkpoint']).name
    wins=[]
    for pair in comparisons:
        lo,hi=pair['simultaneous_95_interval']
        if pair['a']==winner:wins.append(lo>0 and pair['holm_adjusted_p']<.05)
        elif pair['b']==winner:wins.append(hi<0 and pair['holm_adjusted_p']<.05)
    payload=dict(protocol='45 courses x32 or more disjoint fixed-seed episodes; exact paired McNemar with all-pairs Holm correction, plus course-cluster bootstrap with all-pairs Bonferroni',
        episodes_per_course=args.episodes_per_course,ranking=ranked,comparisons=comparisons,
        proposed_checkpoint=ranked[0]['checkpoint'],
        statistically_separated_from_all_finalists=bool(wins and all(wins)),
        interpretation='Highest confirmation SR is the operational choice; overlapping intervals remain inconclusive, not equivalent. No saturation or champion-speed claim follows.',
        episode_only_worst_case_95_halfwidth=1.96*math.sqrt(.25/(45*args.episodes_per_course)))
    (out/'selection.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
