"""Aggregate the completed 12-arm collection screen from immutable local events."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path('/workspace')
OUT = ROOT / 'outputs/diagnostics/collection-next-20260929'
EVAL = ('full_course_success', 'timely_success', 'clean_timely_success',
        'recovered_success', 'crash_rate', 'reference_misses',
        'selection_suite_success', 'selection_suite_timely_success',
        'selection_suite_clean_timely_success')
TRAIN = ('new_labels', 'new_online_replay_labels', 'fresh/action_loss_unweighted',
         'fresh/action_loss_weighted', 'fresh/thrust_physical_mae',
         'fresh/roll_physical_mae', 'fresh/pitch_physical_mae',
         'fresh/yaw_physical_mae')

def main() -> None:
    jobs = json.loads((OUT / 'jobs.json').read_text())
    status = json.loads((OUT / 'queue.status.json').read_text())
    if status['state'] != 'complete' or len(status['completed']) != len(jobs):
        raise RuntimeError('Collection queue is not fully complete')
    summary = {}
    for job in jobs:
        arm, name = job['arm'], job['run_name']
        state = json.loads((OUT / f'{arm}.status.json').read_text())
        if state['state'] != 'complete' or state['round'] != 24 or state['exit_code'] != 0:
            raise RuntimeError(f'{arm} lacks a successful 24-round checkpoint')
        events = [json.loads(line)['metrics'] for line in
                  (ROOT / 'outputs/logs' / f'{name}.full.events.jsonl').read_text().splitlines()]
        ev = [r for r in events if 'eval/selection_suite_success' in r]
        tr = [r for r in events if 'train/round' in r]
        if len(ev) != 25 or len(tr) != 24 or [int(x['train/round']) for x in tr] != list(range(1, 25)):
            raise RuntimeError(f'{arm}: evaluation or training records incomplete')
        curve = [dict(round=i, steps=int(e['eval/step']),
                      **{k:float(e[f'eval/{k}']) for k in EVAL})
                 for i,e in enumerate(ev)]
        final = curve[-1]
        late = {k:mean(row[k] for row in curve[-6:]) for k in EVAL}
        best_manifest = json.loads((ROOT / 'outputs/checkpoints' / name / 'top-k.json').read_text())
        if len(best_manifest['checkpoints']) != 1:
            raise RuntimeError(f'{arm}: best checkpoint retention differs')
        best = best_manifest['checkpoints'][0]
        matching = [r for r in curve if r['steps'] == best['step']]
        if len(matching) != 1:
            raise RuntimeError(f'{arm}: best checkpoint not on curve')
        best_row = matching[0]
        ckeys = sorted({k for row in tr for k in row if k.startswith('train/collection/')})
        counts = {k.removeprefix('train/collection/'):sum(float(row.get(k,0)) for row in tr[1:]) for k in ckeys}
        valid_labels = sum(float(row['train/new_labels']) for row in tr)
        retained = sum(float(row['train/new_online_replay_labels']) for row in tr)
        underfilled = [int(row['train/round']) for row in tr[1:]
                       if row['train/new_online_replay_labels'] < 25002]
        family = {k.removeprefix('eval/'):float(v) for k,v in ev[-1].items()
                  if k.startswith('eval/selection_family/') or
                     (k.startswith('eval/track/') and k.endswith(('/full_course_success','/clean_timely_success')))}
        summary[arm] = dict(description=job['description'], run_name=name,
            steps=int(state['steps']), valid_labels=int(valid_labels), retained_online=int(retained),
            underfilled_rounds=underfilled, final=final, late_six=late,
            best_timely=dict(path=best['path'], score=best['score'], round=best_row['round'],
                             metrics={k:best_row[k] for k in EVAL}),
            fresh_final={k:float(tr[-1][f'train/{k}']) for k in TRAIN if f'train/{k}' in tr[-1]},
            fresh_late={k:mean(float(r[f'train/{k}']) for r in tr[-6:]) for k in TRAIN if f'train/{k}' in tr[-1]},
            collection=counts, final_course=family, curve=curve)
    (OUT/'aggregate.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fields = ['arm','steps','valid_labels','retained_online','underfilled_rounds',
              'final_eventual','final_timely','final_clean_timely','final_recovered','final_crash',
              'late_eventual','late_timely','late_clean_timely','late_recovered','late_crash',
              'best_round','best_timely','best_eventual','best_clean_timely',
              'fresh_loss','collection_active','collection_teacher','recovery_completed',
              'recovery_failed','segment_completed','segment_time_limit']
    with (OUT/'aggregate.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for arm,s in summary.items():
            writer.writerow(dict(arm=arm,steps=s['steps'],valid_labels=s['valid_labels'],
                retained_online=s['retained_online'],underfilled_rounds=len(s['underfilled_rounds']),
                final_eventual=s['final']['selection_suite_success'],
                final_timely=s['final']['selection_suite_timely_success'],
                final_clean_timely=s['final']['selection_suite_clean_timely_success'],
                final_recovered=s['final']['recovered_success'],final_crash=s['final']['crash_rate'],
                late_eventual=s['late_six']['selection_suite_success'],
                late_timely=s['late_six']['selection_suite_timely_success'],
                late_clean_timely=s['late_six']['selection_suite_clean_timely_success'],
                late_recovered=s['late_six']['recovered_success'],late_crash=s['late_six']['crash_rate'],
                best_round=s['best_timely']['round'],best_timely=s['best_timely']['score'],
                best_eventual=s['best_timely']['metrics']['selection_suite_success'],
                best_clean_timely=s['best_timely']['metrics']['selection_suite_clean_timely_success'],
                fresh_loss=s['fresh_final']['fresh/action_loss_unweighted'],
                collection_active=s['collection'].get('active_steps',0),
                collection_teacher=s['collection'].get('active_teacher_steps',0),
                recovery_completed=s['collection'].get('recovery_completed',0),
                recovery_failed=s['collection'].get('recovery_failed',0),
                segment_completed=s['collection'].get('segment_completed',0),
                segment_time_limit=s['collection'].get('segment_time_limit',0)))
    for arm,s in summary.items():
        f=s['final']; l=s['late_six']; b=s['best_timely']
        print(f"{arm:20s} steps={s['steps']:8d} "
              f"final E/T/C={f['selection_suite_success']:.3f}/{f['selection_suite_timely_success']:.3f}/{f['selection_suite_clean_timely_success']:.3f} "
              f"late E/T/C={l['selection_suite_success']:.3f}/{l['selection_suite_timely_success']:.3f}/{l['selection_suite_clean_timely_success']:.3f} "
              f"best T={b['score']:.3f}@{b['round']}")

if __name__ == '__main__': main()
