"""Import public-safe aggregates from a frozen Starscream paper evidence package."""
from pathlib import Path
import argparse,csv,json,statistics,hashlib
a=argparse.ArgumentParser(); a.add_argument('--paper',type=Path,required=True); a.add_argument('--site',type=Path,required=True); args=a.parse_args()
P=args.paper; R=args.site
def csvrows(path): return list(csv.DictReader((P/path).open(encoding='utf-8')))
rows=csvrows('data/final/policy-summary.csv'); episodes=csvrows('data/final/episodes.csv')
metrics={'clean_timely':'clean_timely_success','clean':'clean_success','eventual':'success','timely':'timely_success','rejoin':'rejoin_success'}
reg=json.loads((P/'provenance/final/registry.json').read_text())['policies']
out={'schema':2,'source_package':'Starscream focused review, 2026-10-13','units':'percent of assigned starts; seed spread in percentage points','aggregation':'Unweighted mean and sample standard deviation across three independently trained seeds per setup. Crashes and hard-cap failures remain in denominators. Training-seed indices are not paired treatments.','definitions':{'eventual':'Full ordered completion without crash within 6000 control steps (46.154 s).','clean':'Eventual completion with zero genuine reference-aware missed crossings; not a body-clearance guarantee.','clean_timely':'Clean completion within floor(130 * max(1.5 * reference_seconds, reference_seconds + 3)) control steps.','timely':'Eventual completion within the course deadline.','rejoin':'First accepted gate crossing after a controlled lateral-offset start.'},'datasets':{'pretrain65':'65 qualified training courses; original manifest train split.','eval100':'Public alias for real100-hard-v2 / real100-v2, the 100-course development-exposed panel.','eval60':'Public alias for real60, a separate 60-course suite; not the recovery panel reported here.'},'panels':{},'provenance':[]}
for phase,n in [('nominal',1200),('recovery',60)]:
    panel={'dataset':'eval100','starts_per_seed':n,'courses':100 if phase=='nominal' else 15,'starts_per_course':12 if phase=='nominal' else 4,'models':{}}
    for model in ['mlp','lstm','transformer']:
        seeds=[]
        for row in [r for r in rows if r['phase']==phase and r['architecture']==model]:
            ee=[e for e in episodes if e['phase']==phase and e['policy']==row['policy']]
            assert len(ee)==n==int(row['denominator']) and int(row['missing'])==0
            counts={k:sum(int(e[v]) for e in ee) for k,v in metrics.items()}
            for k,v in metrics.items(): assert counts[k]==int(row['rejoin' if k=='rejoin' else v]),(phase,model,k)
            assert sum(e['crashed']=='True' for e in ee)==int(row['crashed'])
            assert sum(int(e['reference_misses'])>0 for e in ee)==int(row['miss_episodes'])
            seeds.append({'seed_index':int(row['policy'].split('seed')[1]),'training_seed':int(row['training_seed']),'denominator':n,'counts':counts,'crashed':int(row['crashed']),'miss_episodes':int(row['miss_episodes'])})
        assert len(seeds)==3
        stats={k:{'mean':statistics.mean(100*s['counts'][k]/n for s in seeds),'sample_sd':statistics.stdev(100*s['counts'][k]/n for s in seeds)} for k in metrics}
        panel['models'][model]={'seeds':seeds,'statistics':stats}
    out['panels'][phase]=panel
out['panels']['nominal']['conditions']='12 starts per course, approximately 3 m before gate zero, forward speed 2.9â€“3.2 m/s with small perturbations; speed command 16.5 m/s.'
out['panels']['recovery']['conditions']='One course per each of 15 families, four assigned starts; 1.5 m lateral offset and 2 m/s lateral velocity. Full-course evaluation without collection segment resets.'
out['training_exposure']=[{k:r[k] for k in ['architecture','seed_index','training_seed','selected_environment_steps','selected_updates']} for r in reg]
out['training_exposure_definition']='Recorded collected environment steps and optimizer updates at each selected checkpoint. These are not expert-query counts, total solver work, or FLOPs; unequal budgets are part of the full-setup comparison.'
out['limitations']=['Selection25 lies inside eval100; the full panel and complement75 were inspected during development.','Inputs, feature handling, and training budgets differ; these are complete-setup comparisons, not isolated attention effects.','Privileged-state simulation. Plant generalization, deployment, and matched solver/RL comparisons remain open.']
for path in ['data/final/policy-summary.csv','data/final/episodes.csv','provenance/final/registry.json','sections/evaluation.tex','sections/method.tex','sections/mpcc-details.tex']:
    raw=(P/path).read_bytes();out['provenance'].append({'package_relative_path':path,'sha256':hashlib.sha256(raw).hexdigest()})
dest=R/'starscream/assets/data/development-comparison.json'; dest.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(f'Verified {len(episodes)} episode records against {len(rows)} per-policy summaries; exported {dest.stat().st_size} bytes.')
