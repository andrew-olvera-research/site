#!/usr/bin/env python3
"""Paired teacher/student report on real100-v2 complement75 or full100."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from scripts.train_vision_student import load_setup, evaluate, parse_stage, local_path, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full100',action='store_true')
    parser.add_argument('--episodes',type=int,default=8)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--protocol',default='configs/eval/real100_v2_timed_protocol_v1.json')
    parser.add_argument('--legacy-eval',action='store_true',help='explicitly reproduce the historical long-deadline evaluator')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.episodes<1:
        raise ValueError('episodes must be positive')
    torch.set_num_threads(1)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=saved['config']; config['evaluation_episodes']=args.episodes
    if args.legacy_eval:
        config.pop('evaluation_protocol',None);config.pop('evaluation_protocol_sha256',None)
    else:
        config['evaluation_protocol']=args.protocol
        config['evaluation_protocol_sha256']=sha256(local_path(args.protocol))
    _,settings,source,selection,teacher,normalizer,student=load_setup(config)
    student.load_state_dict(saved['model']); student.to(args.device).eval(); teacher.to(args.device)
    manifest_path=local_path(selection['source'])
    if sha256(manifest_path)!=selection['source_sha256']:
        raise ValueError('real100 source manifest changed')
    records=json.loads(manifest_path.read_text())['records']
    if not args.full100:
        records=[r for r in records if r['slot'] in selection['report_slots']]
    counts=Counter(r['family'] for r in records)
    suite=dict(records=records,family_weights={f:n/len(records) for f,n in counts.items()})
    stage=parse_stage(source['evaluation_curriculum'])
    student_report=evaluate(config,settings,stage,suite,teacher,normalizer,student,args.device)
    teacher_report=evaluate(config,settings,stage,suite,teacher,normalizer,student,args.device,True)
    reference=teacher_report['selection_score']
    result=dict(suite='full100' if args.full100 else 'complement75',
        checkpoint=str(args.checkpoint),checkpoint_sha256=sha256(args.checkpoint),
        teacher=teacher_report,student=student_report,
        success_retention=student_report['selection_score']/reference if reference else None,
        success_difference=student_report['selection_score']-reference,
        evaluation_protocol=config.get('evaluation_protocol','legacy'),
        note='Complement is excluded from student checkpoint selection; prior teacher exposure is not excluded. Timed-v2 completion is deadline-qualified, not eventual completion.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)


if __name__=='__main__':
    main()
