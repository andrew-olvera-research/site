"""Course-balanced closed-loop reporting; never ranks by raw gate count.

Inputs are existing evaluator metrics and a frozen manifest. Per-course Wilson
intervals quantify episode uncertainty, not uncertainty over unseen geometries.
Missing courses are errors, not zeros or silently excluded denominators.
"""
from collections import defaultdict
import math
import numpy as np


def wilson(successes, episodes, z=1.959963984540054):
    if episodes <= 0 or not 0 <= successes <= episodes:
        raise ValueError('invalid success counts')
    p = successes / episodes
    scale = 1 + z*z/episodes
    center = (p + z*z/(2*episodes))/scale
    half = z*math.sqrt(p*(1-p)/episodes + z*z/(4*episodes**2))/scale
    return [max(0., center-half), min(1., center+half)]


def course_balanced_report(metrics, records, split='validation', *, control_hz):
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError('positive control_hz required for lap timing')
    rows = [r for r in records if r['split'] == split]
    if not rows or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate evaluation course set')
    courses = []
    for r in rows:
        prefix = f"track/{r['name']}/"
        def required(key):
            value = float(metrics[prefix + key])
            if not np.isfinite(value): raise ValueError(f'nonfinite {prefix}{key}')
            return value
        count = required('episodes'); sr = required('full_course_success')
        n = int(r['gate_count'])
        if count < 1 or count != int(count) or not 0 <= sr <= 1 or n < 1:
            raise ValueError('invalid evaluation counts')
        row = dict(name=r['name'], grammar=r.get('grammar', r['family']),
            gate_count=n, episodes=int(count), full_course_success=sr,
            episode_wilson95=wilson(sr*count, count), crash_rate=required('crash_rate'),
            mean_gate_fraction=float(np.clip(required('mean_gates') / n, 0, 1)))
        for fraction in (.25, .50, .75):
            row[f'p{int(100*fraction)}_percent_course'] = required(f'p{math.ceil(n*fraction)}')
        pace = metrics.get(prefix + 'successful_mean_steps')
        row['successful_completion_time_s'] = (
            float(pace)/control_hz if sr > 0 and pace is not None and np.isfinite(pace) else None)
        courses.append(row)
    fields = ['full_course_success', 'crash_rate', 'mean_gate_fraction',
              'p25_percent_course', 'p50_percent_course', 'p75_percent_course']
    def aggregate(items):
        return dict(courses=len(items), episodes=sum(r['episodes'] for r in items),
            **{key:float(np.mean([r[key] for r in items])) for key in fields})
    groups = {}
    for field in ('grammar', 'gate_count'):
        buckets = defaultdict(list)
        for r in courses: buckets[str(r[field])].append(r)
        groups[field] = {k:aggregate(v) for k,v in sorted(buckets.items())}
    return dict(primary='macro/full_course_success', macro=aggregate(courses),
        minimum_grammar_success=min(v['full_course_success'] for v in groups['grammar'].values()),
        per_course=courses, groups=groups,
        uncertainty_scope='Per-course Wilson intervals only; not a joint interval or unseen-course guarantee.',
        speed_scope='Success-conditioned time is secondary; report beside success, never omit failed courses.')
