"""Summarize frozen v6.22 actor evaluation JSON files."""
from collections import defaultdict
import json
from pathlib import Path


def main():
    root = Path('outputs/evals/v622-frozen')
    manifests = {}
    for split in ('real60', 'real100_hard'):
        manifest = json.loads(Path(f'configs/eval/v6_22_{split}.manifest.json').read_text())
        manifests[split] = {r['name']: (r['suite'], r['family']) for r in manifest['records']}
    result = {}
    for path in sorted(root.glob('*-8ep.json')):
        payload = json.loads(path.read_text())
        if 'metrics' not in payload: continue
        metrics = payload['metrics']
        split = 'real60' if 'real60' in path.name else 'real100_hard'
        suite_name = split.replace('_', '-')
        rows = []
        for name, (suite, family) in manifests[split].items():
            prefix = f'track/{name}/'
            rows.append(dict(name=name, suite=suite, family=family,
                success=metrics.get(prefix+'full_course_success', float('nan')),
                gates=metrics.get(prefix+'mean_gates', float('nan')),
                crash=metrics.get(prefix+'crash_rate', float('nan')),
                gate_speed=metrics.get(prefix+'episode_mean_gate_speed_mps_median', float('nan')),
                max_speed=metrics.get(prefix+'episode_maximum_speed_mps_median', float('nan'))))
        groups = {}
        for label, selected in [('generated', [r for r in rows if r['suite'] == suite_name]),
                                ('public_reference', [r for r in rows if r['suite'] == 'public_reference'])]:
            if not selected: continue
            groups[label] = dict(courses=len(selected),
                completion=sum(r['success'] for r in selected)/len(selected),
                mean_gates=sum(r['gates'] for r in selected)/len(selected),
                crash_rate=sum(r['crash'] for r in selected)/len(selected),
                gate_speed=sum(r['gate_speed'] for r in selected)/len(selected),
                maximum_speed=sum(r['max_speed'] for r in selected)/len(selected))
        families = defaultdict(list)
        for row in rows: families[row['family']].append(row)
        groups['families'] = {family: dict(courses=len(rs),
            completion=sum(r['success'] for r in rs)/len(rs),
            mean_gates=sum(r['gates'] for r in rs)/len(rs),
            crash_rate=sum(r['crash'] for r in rs)/len(rs))
            for family, rs in sorted(families.items())}
        groups['aggregate'] = {key: metrics[key] for key in
            ('full_course_success','mean_gates','crash_rate','mean_speed_mps',
             'mean_gate_speed_mps','maximum_speed_mps','episodes','episodes_per_track')}
        groups['worst_courses'] = sorted(
            [{'name': r['name'], 'family': r['family'], 'success': r['success'],
              'mean_gates': r['gates'], 'crash_rate': r['crash']} for r in rows],
            key=lambda r: (r['success'], -r['crash_rate']))[:10]
        result[path.stem] = groups
    out = root/'summary-8ep.json'; out.write_text(json.dumps(result, indent=2, sort_keys=True));
    print(json.dumps(result, indent=2, sort_keys=True)); print('WROTE', out)


if __name__ == '__main__': main()
