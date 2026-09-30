"""Bounded evaluator invariance experiment: two courses, two policies, worker regrouping."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.train_vision_student import load_setup,evaluate,parse_stage
from starscream.racing_evaluation import canonical_inference,fixed_shape_call


def main():
    torch.set_num_threads(1)
    config=json.loads(Path('configs/exp/v6.21.1.student-v2/control.json').read_text())
    saved=torch.load('outputs/vision-distillation/v62111-plant-ekf-readout-async-48r/best.pt',map_location='cpu',weights_only=False)
    _,settings,source,selection,teacher,norm,student=load_setup(config)
    teacher.cuda().eval();student.load_state_dict(saved['model']);student.cuda().eval()
    path=Path('outputs/vision-distillation/v62111-plant-ekf-readout-async-48r/replay/round-0039')
    ids=np.arange(16)*100
    raw={k:torch.as_tensor(np.array(np.load(path/f'{k}.npy',mmap_mode='r')[ids]),device='cuda').float() for k in ('teacher_features','features','speed')}
    masks=torch.as_tensor(np.unpackbits(np.array(np.load(path/'masks.npy',mmap_mode='r')[ids]),axis=-1,count=160),device='cuda').float()
    x=(raw['teacher_features']-torch.as_tensor(norm.mean,device='cuda'))/torch.as_tensor(norm.std,device='cuda')
    static={}
    for arm,call,inputs in [('teacher',teacher,[x,raw['speed']]),('student',lambda f,m,s:student(f,m,s)['action'],[raw['features'],masks,raw['speed']])]:
        with canonical_inference():
            together=fixed_shape_call(call,inputs,list(range(16)))
            differences=[]
            for i in (0,1,5,15):
                for lane in (0,5,15):
                    alone=fixed_shape_call(call,[v[i:i+1] for v in inputs],[lane])
                    differences.append(float((alone[0]-together[i]).abs().max()))
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(30):fixed_shape_call(call,inputs,list(range(16)))
            torch.cuda.synchronize()
        static[arm]=dict(max_action_difference=max(differences),milliseconds_per_16=(time.perf_counter()-start)/30*1000)
        if max(differences)!=0:raise AssertionError(f'{arm}: lane/neighbour invariance failed {static[arm]}')
    print(json.dumps(dict(static=static)),flush=True)
    records=json.loads(Path(selection['source']).read_text())['records']
    records=[r for r in records if r['slot'] in ('real100-hard-021','real100-hard-038')]
    counts=Counter(r['family'] for r in records);suite=dict(records=records,family_weights={f:n/len(records) for f,n in counts.items()})
    config['evaluation_episodes']=1
    reports={}
    for arm in ('teacher','student'):
        reports[arm]={}
        for workers in (1,2):
            config['evaluation_workers']=workers;start=time.perf_counter()
            report=evaluate(config,settings,parse_stage(source['evaluation_curriculum']),suite,teacher,norm,student,'cuda',arm=='teacher')
            reports[arm][str(workers)]=report
            print(json.dumps(dict(arm=arm,workers=workers,seconds=time.perf_counter()-start,success=report['selection_score'],
                episodes=[r['episodes'][0]['steps'] for r in report['tracks']])),flush=True)
        if reports[arm]['1']['tracks']!=reports[arm]['2']['tracks']:
            raise AssertionError(f'{arm}: regrouping changes episode traces')
    output=Path('outputs/diagnostics/racing-eval-v2');output.mkdir(exist_ok=True,parents=True)
    (output/'batch-invariance.json').write_text(json.dumps(dict(status='passed',static=static,reports=reports),indent=2)+'\n')


if __name__=='__main__':main()
