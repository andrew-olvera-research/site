"""Summarize saved evidence without treating unlike evaluation contracts as equal."""
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/diagnostics/pretraining-mode-decision-20260928'

def main():
    paths = {
        'v621': 'outputs/evals/v6222-crosscompare/v621-v6_22_real100_hard_v2-e8.json',
        'update_fix': 'outputs/evals/v6211-update-fix-final/update-fix-real100-hard-v2-e8.json',
        'v6222': 'outputs/evals/v6222-crosscompare/v6222-v6_22_real100_hard_v2-e8.json',
        'plant': 'outputs/evals/v62111-plant-final/best-real100-hard-v2-e8.json',
    }
    historical = {}
    for name, path in paths.items():
        d = json.loads((ROOT/path).read_text())
        m = d['metrics']
        medians = [v/130 for k,v in m.items() if k.endswith('/successful_median_steps') and math.isfinite(v)]
        historical[name] = dict(source=path, checkpoint=d['checkpoint'], round=d.get('checkpoint_round'),
            steps=d.get('checkpoint_environment_steps'), success=m['full_course_success'], crash=m['crash_rate'],
            successful_courses=len(medians), median_of_successful_course_medians_seconds=statistics.median(medians),
            note='Conditional course statistic, not pooled episode median or matched survival comparison.')
    logs = {}
    patterns = ['*plant-selection25-dagger.full.events.jsonl', '*recovery-fix-scale.events.jsonl', '*recovery-support-v3.full.events.jsonl', '*midtrain*full.events.jsonl']
    for pattern in patterns:
        for path in (ROOT/'outputs/logs').glob(pattern):
            panels = []
            last_quality = None
            for line in path.open():
                try:
                    d=json.loads(line)
                except json.JSONDecodeError:
                    continue
                m=d.get('metrics', {})
                if 'train/quality/combined/sampled_recovery' in m:
                    last_quality={k:v for k,v in m.items() if k.startswith('train/quality/') or k=='train/round'}
                if 'eval/full_course_success' in m:
                    panels.append(dict(step=d.get('step'), **{k:v for k,v in m.items() if k in
                        ['eval/full_course_success','eval/selection_suite_success','eval/selection_suite_clean_success',
                         'eval/selection_suite_clean_timely_success','eval/successful_median_steps','eval/mean_gates','eval/crash_rate',
                         'eval/dagger_policy_version']}))
            logs[path.name]=dict(panels=len(panels), last=panels[-1] if panels else None,
                last_quality=last_quality,
                peak=max(panels,key=lambda r:r['eval/full_course_success']) if panels else None,
                matched_40m=min(panels,key=lambda r:abs(r['step']-40_000_000))
                    if panels and min(r['step'] for r in panels)<=40_000_000<=max(r['step'] for r in panels) else None)
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(historical=historical,logs=logs)
    (OUT/'history-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
