"""Side-by-side macro comparison of fresh-seed final evaluations (base vs RL checkpoints)."""
import argparse, json
from pathlib import Path


def load(p):
    return json.loads((Path(p)/'summary.json').read_text())['reports']


def pct(x):
    return '-' if x is None else f'{100*x:.2f}%'


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs',nargs='+',required=True,help='label=summary_dir')
    a=ap.parse_args()
    runs={l:load(p) for l,p in (s.split('=',1) for s in a.runs)}
    labels=list(runs)
    cohorts=['training-randomized','validation-randomized','validation-nominal','reporting-randomized','reporting-nominal']
    print('## Macro completion (full course success) / crash rate / mean gate fraction\n')
    print('| Cohort | Episodes | '+' | '.join(labels)+' |'); print('|---|---:|'+'---:|'*len(labels))
    for c in cohorts:
        cells=[]
        for l in labels:
            r=runs[l].get(c)
            cells.append('-' if r is None else f"{pct(r['macro']['full_course_success'])} / {pct(r['macro']['crash_rate'])} / {pct(r['macro']['mean_gate_fraction'])}")
        eps=next((runs[l][c]['macro']['episodes'] for l in labels if c in runs[l]),'-')
        print(f'| {c} | {eps} | '+' | '.join(cells)+' |')
    for grp in ['grammar','gate_count']:
        for c in ['training-randomized','validation-randomized','validation-nominal']:
            print(f'\n## {c}: completion by {grp}\n')
            print('| '+grp+' | '+' | '.join(labels)+' |'); print('|---|'+'---:|'*len(labels))
            keys=[k for l in labels if c in runs[l] for k in runs[l][c]['groups'][grp]]
            seen=[]; [seen.append(k) for k in keys if k not in seen]
            for k in sorted(seen,key=lambda s:(len(s),s)):
                print(f'| {k} | '+' | '.join(pct(runs[l][c]['groups'][grp][k]['full_course_success']) if c in runs[l] and k in runs[l][c]['groups'][grp] else '-' for l in labels)+' |')
    for c in ['reporting-randomized','reporting-nominal']:
        print(f'\n## {c}: real reference tracks\n')
        print('| Track | '+' | '.join(f'{l} success (Wilson95) / crash / gate frac / mean lap / fastest lap / gate speed' for l in labels)+' |'); print('|---|'+'---|'*len(labels))
        names=[row['name'] for row in runs[labels[0]][c]['per_course']]
        for n in names:
            cells=[]
            for l in labels:
                row=next((x for x in runs[l][c]['per_course'] if x['name']==n),None) if c in runs[l] else None
                if row is None: cells.append('-'); continue
                w=row['episode_wilson95']
                lap=row['successful_completion_time_s']; fl=row['fastest_successful_time_s']
                cells.append(f"{pct(row['full_course_success'])} ({100*w[0]:.1f}-{100*w[1]:.1f}) / {pct(row['crash_rate'])} / {pct(row['mean_gate_fraction'])} / {'-' if lap is None else f'{lap:.2f}s'} / {'-' if fl is None else f'{fl:.2f}s'} / {row['mean_gate_speed_mps']:.2f} m/s")
            print(f'| {n} | '+' | '.join(cells)+' |')
    # per-course validation deltas
    if len(labels)>=2:
        c='validation-randomized'; base,other=labels[0],labels[1]
        if c in runs[base] and c in runs[other]:
            print(f'\n## {c}: per-course completion, {base} -> {other} (sorted by delta)\n')
            print('| Course | gates | '+base+' | '+other+' | delta |'); print('|---|---:|---:|---:|---:|')
            rows=[]
            for row in runs[base][c]['per_course']:
                o=next(x for x in runs[other][c]['per_course'] if x['name']==row['name'])
                rows.append((o['full_course_success']-row['full_course_success'],row['name'],row['gate_count'],row['full_course_success'],o['full_course_success']))
            for d,n,g,a_,b_ in sorted(rows):
                print(f'| {n} | {g} | {pct(a_)} | {pct(b_)} | {100*d:+.1f} |')
            print('\ncourses improved / same / worse:',sum(1 for r in rows if r[0]>0),sum(1 for r in rows if r[0]==0),sum(1 for r in rows if r[0]<0))
    # speed / lap time on validation
    for c in ['validation-randomized','validation-nominal','training-randomized']:
        print(f'\n## {c}: success-conditioned lap time and speed (course means over courses with any success)\n')
        print('| run | courses w/ success | mean successful lap (s) | mean fastest lap (s) | mean gate speed (m/s) | median policy/teacher time ratio |'); print('|---|---:|---:|---:|---:|---:|')
        import statistics as st
        for l in labels:
            if c not in runs[l]: continue
            pc=[r for r in runs[l][c]['per_course'] if r['successful_completion_time_s'] is not None]
            ratios=[r['conditional_policy_teacher_time_ratio'] for r in pc if r.get('conditional_policy_teacher_time_ratio')]
            gs=[r['mean_gate_speed_mps'] for r in runs[l][c]['per_course'] if r.get('mean_gate_speed_mps') is not None]
            print(f"| {l} | {len(pc)} | {st.mean(r['successful_completion_time_s'] for r in pc):.3f} | {st.mean(r['fastest_successful_time_s'] for r in pc):.3f} | {st.mean(gs):.2f} | {st.median(ratios):.3f} |" if ratios else f"| {l} | {len(pc)} | {st.mean(r['successful_completion_time_s'] for r in pc):.3f} | {st.mean(r['fastest_successful_time_s'] for r in pc):.3f} | {st.mean(gs):.2f} | - |")
