"""Summarize matched mini-course training and validation by round and step."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/diagnostics/mini-slalom-recovery-baselines"
NAMES = ("broad", "bounded")
PREFIX = "starscream-v6.21.1.1-mini-slalom-"
FIELDS = ("round", "steps", "new_labels", "loss", "successful_episode_label_fraction",
          "teacher_valid_fraction", "round_seconds", "eval_success", "eval_timely",
          "eval_clean", "eval_clean_timely", "eval_mean_gates",
          "eval_010_success", "eval_019_success")


def load(name):
    path = ROOT / f"outputs/logs/{PREFIX}{name}-r32.full.events.jsonl"
    if not path.exists():
        return []
    events=[]
    for line in path.read_text().splitlines():
        try: events.append(json.loads(line))
        except json.JSONDecodeError: continue  # possible live final line
    evaluated={}
    for e in events:
        m=e['metrics']
        if 'eval/selection_suite_success' in m:
            evaluated[e['step']]=m
    rows=[]
    for e in events:
        m=e['metrics']
        if 'train/round' not in m: continue
        v=evaluated.get(e['step'],{})
        row=dict(round=int(m['train/round']),steps=int(m['train/environment_steps']),
            new_labels=m.get('train/new_labels'),loss=m.get('train/loss'),
            successful_episode_label_fraction=m.get('train/successful_episode_label_fraction'),
            teacher_valid_fraction=m.get('train/teacher_valid_fraction'),
            round_seconds=m.get('train/round_seconds'),
            eval_success=v.get('eval/selection_suite_success'),
            eval_timely=v.get('eval/selection_suite_timely_success'),
            eval_clean=v.get('eval/selection_suite_clean_success'),
            eval_clean_timely=v.get('eval/selection_suite_clean_timely_success'),
            eval_mean_gates=v.get('eval/mean_gates'),
            eval_010_success=v.get('eval/track/v622_hard_slalom_010_024_mirror1/full_course_success'),
            eval_019_success=v.get('eval/track/v622_hard_slalom_019_012/full_course_success'))
        rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary={}
    for name in NAMES:
        rows=load(name)
        with (OUT/f'{name}-rounds.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,FIELDS);writer.writeheader();writer.writerows(rows)
        if rows:
            valid=[r for r in rows if r['eval_success'] is not None]
            summary[name]=dict(completed_rounds=len(rows),last=rows[-1],
                best_eventual=max(valid,key=lambda r:r['eval_success']) if valid else None,
                best_clean_timely=max(valid,key=lambda r:r['eval_clean_timely']) if valid else None,
                total_round_seconds=sum(r['round_seconds'] or 0 for r in rows))
        else: summary[name]=dict(completed_rounds=0)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__': main()
