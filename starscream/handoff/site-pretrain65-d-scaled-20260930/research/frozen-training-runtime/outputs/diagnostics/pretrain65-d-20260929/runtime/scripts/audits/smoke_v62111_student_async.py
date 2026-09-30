"""Short end-to-end collector, replay, and readout-gradient qualification."""
import json
from dataclasses import replace
from pathlib import Path
import tempfile

import torch

from scripts.train_vision_student import collect_batch, load_setup, parse_stage
from scripts.train_vision_student_async import Reservoir, ShardReplay, update

config = json.loads(Path('configs/exp/v6.21.1.1/plant_ekf_readout_student_async.json').read_text())
_, settings, source, _, teacher, normalizer, student = load_setup(config)
teacher.cuda().eval(); student.cuda()
stage = replace(parse_stage(source['curriculum']), max_steps=12)
track = source['curriculum']['tracks'][0]
replay = Reservoir([track], 32, 17)
rows = collect_batch([(track, 17)], config, settings, stage, teacher,
                     normalizer, student, 'cuda', beta=1., replay=replay, teacher_only=True)
assert len(rows) == 1 and replay.rows[track]
sample = replay.rows[track][0]
assert sample['teacher_features'].shape == (3, 167)
assert sample['features'].shape == (3, 125)
assert sample['readout'].shape == (teacher.hidden_dim,)
with tempfile.TemporaryDirectory() as temp:
    path = Path(temp)/'round-0000'
    replay.save(path)
    bank = ShardReplay([track], 2, [.40,.35,.25], 19)
    bank.add(path, permanent=True)
    predictor = torch.nn.Linear(student.width,teacher.hidden_dim).cuda()
    optimizer = torch.optim.AdamW(list(student.parameters())+list(predictor.parameters()),lr=1e-4)
    tiny = dict(config,batch_size=2)
    metrics,requested,actual = update(student,predictor,optimizer,bank,tiny,
                                      'cuda',teacher,normalizer)
    assert all(float(value) == float(value) for value in metrics.values())
    assert requested.sum() == actual.sum() == 2
    print(json.dumps(dict(status='pass',rows=len(replay.rows[track]),
                          losses=metrics,requested=requested.tolist(),actual=actual.tolist())))
