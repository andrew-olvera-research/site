"""Convert an eval_privileged_dagger JSON into course-balanced corpus metrics."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.corpus_evaluation import course_balanced_report
from starscream.course_model.training import atomic_json

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--split', default='validation')
    p.add_argument('--control-hz',type=float,required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    raw=json.loads(a.evaluation.read_text())
    report=course_balanced_report(raw.get('metrics',raw),
        json.loads(a.manifest.read_text())['records'], a.split, control_hz=a.control_hz)
    atomic_json(a.output,report)
    print(json.dumps(report['macro'],indent=2))
