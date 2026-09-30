"""Summarize the pinned scaled-D real60 and real100-v2 evaluation reports."""
import json
from pathlib import Path

import numpy as np

OUT = Path('/workspace/outputs/evals/pretrain65-d-scaled-final-20260929')
HZ = 130.0


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(count=0, fastest_s=None, mean_s=None, median_s=None)
    return dict(count=int(len(values)), fastest_s=float(np.min(values)),
                mean_s=float(np.mean(values)), median_s=float(np.median(values)),
                p10_s=float(np.quantile(values, 0.1)),
                p90_s=float(np.quantile(values, 0.9)))


real60 = json.loads((OUT / 'real60-e32.json').read_text())
real100 = json.loads((OUT / 'real100-v2-e32.json').read_text())
m = real60['metrics']
names = sorted({k.split('/')[1] for k in m if k.startswith('track/')})
courses60 = {}
for name in names:
    prefix = f'track/{name}/'
    if prefix + 'full_course_success' not in m:
        continue
    courses60[name] = dict(
        episodes=int(m[prefix + 'episodes']),
        eventual=float(m[prefix + 'full_course_success']),
        fastest_s=float(m[prefix + 'successful_minimum_steps']) / HZ
            if np.isfinite(m.get(prefix + 'successful_minimum_steps', float('nan'))) else None,
        mean_s=float(m[prefix + 'successful_mean_steps']) / HZ
            if np.isfinite(m.get(prefix + 'successful_mean_steps', float('nan'))) else None,
        median_s=float(m[prefix + 'successful_median_steps']) / HZ
            if np.isfinite(m.get(prefix + 'successful_median_steps', float('nan'))) else None,
    )
rows = real100['episodes']
courses100 = {}
for slot in sorted({row['slot'] for row in rows}):
    subset = [row for row in rows if row['slot'] == slot]
    times = [row['steps'] / HZ for row in subset if row['success']]
    courses100[slot] = dict(
        episodes=len(subset), eventual=float(np.mean([row['success'] for row in subset])),
        timely=float(np.mean([row['timely_success'] for row in subset])),
        clean_timely=float(np.mean([row['clean_success'] and row['timely_success'] for row in subset])),
        successful_laps=distribution(times),
    )
summary = dict(
    checkpoint=real100['checkpoint'], checkpoint_round=real100['checkpoint_round'],
    real60=dict(episodes=real60['episodes'], eventual=m['full_course_success'],
                timing_note='No complete frozen real60 deadline protocol; do not treat logged timely fields as a suite score.',
                course_lap_statistics_s=dict(
                    fastest=distribution([v['fastest_s'] for v in courses60.values() if v['fastest_s'] is not None]),
                    mean=distribution([v['mean_s'] for v in courses60.values() if v['mean_s'] is not None]),
                    median=distribution([v['median_s'] for v in courses60.values() if v['median_s'] is not None])),
                courses=courses60),
    real100_v2=dict(aggregate=real100['aggregate'],
                    successful_laps=distribution([row['steps'] / HZ for row in rows if row['success']]),
                    courses=courses100),
)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(OUT / 'summary.json')
