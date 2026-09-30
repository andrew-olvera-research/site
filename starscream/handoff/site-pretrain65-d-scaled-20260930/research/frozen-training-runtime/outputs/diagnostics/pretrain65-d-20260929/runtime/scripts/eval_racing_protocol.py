"""Unified teacher/student timed evaluation. Default is preflight; --run executes."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import torch

from scripts.train_vision_student import load_setup,evaluate,parse_stage,local_path,sha256
from starscream.racing_evaluation import load_protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/exp/v6.21.1.student-v2/control.json'))
    parser.add_argument('--student-checkpoint',type=Path)
    parser.add_argument('--policy',choices=['teacher','student','paired'],default='teacher')
    parser.add_argument('--suite',choices=['selection25','complement75','full100'],default='selection25')
    parser.add_argument('--episodes',type=int,default=4)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.episodes<1:parser.error('episodes must be positive')
    config=json.loads(args.config.read_text());config['evaluation_episodes']=args.episodes
    saved=None
    if args.policy!='teacher':
        if args.student_checkpoint is None:parser.error('student checkpoint required')
        saved=torch.load(args.student_checkpoint,map_location='cpu',weights_only=False)
        if saved['config']['teacher_sha256']!=config['teacher_sha256']:
            raise ValueError('teacher/student reference checkpoint differs')
        config['model']=deepcopy(saved['config']['model'])
    _,settings,source,selection,teacher,norm,student=load_setup(config)
    protocol=load_protocol(config['evaluation_protocol'],config['evaluation_protocol_sha256'])
    selected={r['slot'] for r in selection['records']}
    records=[r for r in protocol['records'] if args.suite=='full100' or
             ((r['slot'] in selected)==(args.suite=='selection25'))]
    counts=Counter(r['family'] for r in records)
    weights=selection['family_weights'] if args.suite=='selection25' else {f:n/len(records) for f,n in counts.items()}
    suite=dict(records=records,family_weights=weights)
    if not args.run:
        print(json.dumps(dict(status='preflight_only',courses=len(records),episodes=len(records)*args.episodes,
            policy=args.policy,protocol_sha256=config['evaluation_protocol_sha256'],deadline_range=[min(r['deadline_steps'] for r in records),max(r['deadline_steps'] for r in records)])))
        return
    if args.output is None:parser.error('--run requires --output')
    if args.output.exists():raise FileExistsError(args.output)
    if saved:student.load_state_dict(saved['model'])
    student.to(args.device).eval();teacher.to(args.device).eval()
    result=dict(schema='starscream-racing-report-v2',suite=args.suite,teacher_checkpoint=config['teacher_checkpoint'],
        teacher_sha256=config['teacher_sha256'],evaluation_seed=config['evaluation_seed'])
    for arm in ('teacher','student'):
        if args.policy in ('paired',arm):
            result[arm]=evaluate(config,settings,parse_stage(source['evaluation_curriculum']),suite,teacher,norm,student,args.device,arm=='teacher')
    if saved:result.update(student_checkpoint=str(args.student_checkpoint),student_sha256=sha256(args.student_checkpoint))
    if args.policy=='paired':
        denominator=result['teacher']['selection_score']
        result['timely_success_retention']=result['student']['selection_score']/denominator if denominator else None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(args.output)


if __name__=='__main__':main()
