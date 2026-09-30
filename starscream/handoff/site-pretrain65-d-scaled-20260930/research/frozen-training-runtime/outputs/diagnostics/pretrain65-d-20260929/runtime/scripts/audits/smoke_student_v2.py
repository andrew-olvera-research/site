"""Disposable 320-step collection and three update checks; never saves a trained model."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from scripts.train_vision_student import load_setup,parse_stage,collect_batch
from scripts.train_vision_student_async import Reservoir,ShardReplay,update
from starscream.student_learning import make_readout_predictor,calibrate_readout


def main():
    torch.set_num_threads(1);torch.manual_seed(991)
    c=json.loads(Path('configs/exp/v6.21.1.student-v2/combined.json').read_text())
    _,settings,source,_,teacher,norm,student=load_setup(c)
    teacher.cuda().eval();student.cuda()
    frozen={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
    stage=replace(parse_stage(source['curriculum']),max_steps=160)
    tracks=source['curriculum']['tracks'][:2]
    bank=Reservoir(tracks,128,991)
    collected=collect_batch([(track,991+i) for i,track in enumerate(tracks)],c,settings,stage,teacher,norm,student,'cuda',1.,bank,True)
    with tempfile.TemporaryDirectory(prefix='starscream-student-v2-smoke-') as folder:
        path=Path(folder)/'round-0000';bank.save(path)
        replay=ShardReplay(tracks,2,c['replay_strata'],991,c['recovery_success_fraction']);replay.add(path,True)
        arrays=replay.shards[0]['arrays'];normalization=calibrate_readout(arrays['readout'],arrays['track_id'])
        predictor=make_readout_predictor(student.width,teacher.hidden_dim,c).cuda()
        optimizer=torch.optim.AdamW(list(student.parameters())+list(predictor.parameters()),lr=1e-4)
        c['batch_size']=8;losses=[]
        for _ in range(3):
            loss,_,_=update(student,predictor,optimizer,replay,c,'cuda',teacher,norm,settings,normalization)
            assert all(np.isfinite(v) for v in loss.values());losses.append(loss)
        gradients={name:float(sum(p.grad.float().square().sum() for p in module.parameters() if p.grad is not None).sqrt())
                   for name,module in [('memory',student.memory_encoder),('predictor',predictor),('vision',student.vision_encoder)]}
        assert all(v>0 and np.isfinite(v) for v in gradients.values()),gradients
        assert all(p.grad is None for p in teacher.parameters())
        assert all(torch.equal(frozen[k],v.detach().cpu()) for k,v in teacher.state_dict().items())
        result=dict(status='passed',simulation_steps=sum(e['steps'] for e in collected),disposable_updates=3,
                    model_checkpoint_saved=False,teacher_unchanged=True,parameters=student.parameter_counts(),
                    gradient_norms=gradients,losses=losses,memory_shape=list(arrays['memory'].shape))
    output=Path('outputs/diagnostics/racing-eval-v2');output.mkdir(exist_ok=True,parents=True)
    (output/'student-smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
