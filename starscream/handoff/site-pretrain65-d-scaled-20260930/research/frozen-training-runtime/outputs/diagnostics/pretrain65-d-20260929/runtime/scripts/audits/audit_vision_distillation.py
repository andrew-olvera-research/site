#!/usr/bin/env python3
"""Disposable native/gradient/throughput audit. Never writes run checkpoints."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.train_vision_student import (load_setup, collect_batch, BalancedReplay,
    update, environment, parse_stage, sample_curriculum_spawn)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--collection-steps', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(1)
    reports = []
    for path in sorted(Path('configs/exp/v6.21.1.distill').glob('*.json')):
        config = json.loads(path.read_text())
        setup = load_setup(config)
        reports.append(dict(config=path.name, parameters=setup[-1].parameter_counts()))
        del setup
    config = json.loads(Path('configs/exp/v6.21.1.distill/base_raw_interface_3m.json').read_text())
    _, settings, source, suite, teacher, normalizer, student = load_setup(config)
    teacher.to(args.device); student.to(args.device)
    config['batch_size'] = args.batch_size
    stage = replace(parse_stage(source['curriculum']), max_steps=args.collection_steps)
    tracks = source['curriculum']['tracks'][:config['rollout_envs']]
    replay = BalancedReplay(10000,tracks)
    started = time.perf_counter()
    episodes = collect_batch([(p, 381+i) for i,p in enumerate(tracks)], config, settings,
        stage, teacher, normalizer, student, args.device, beta=.5, replay=replay)
    collection_seconds = time.perf_counter()-started
    assert len(replay)>0 and all(np.isfinite(r['steps']) for r in episodes)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config['learning_rate'])
    rng = np.random.default_rng(1)
    metrics = []
    for i in range(5):
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = update(student,optimizer,replay,config,rng,args.device,teacher,normalizer)
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize()
        metrics.append(dict(seconds=time.perf_counter()-start,**result))
    assert all(np.isfinite(m['loss']) for m in metrics)
    assert all(p.grad is None for p in teacher.parameters())
    assert student.estimate.weight.grad.abs().sum().item()>0
    assert student.vision_film.weight.grad.abs().sum().item()>0
    assert student.route_film.weight.grad.abs().sum().item()>0
    # Prove action loss through frozen teacher alone reaches the state head.
    student.zero_grad(set_to_none=True)
    sample = replay.sample(2,rng)
    f = torch.tensor(np.stack([r['features'] for r in sample]),device=args.device)
    masks = torch.tensor(np.stack([np.unpackbits(r['masks'],axis=-1,count=160) for r in sample]),device=args.device).float()
    speed = torch.full((2,),16.5,device=args.device)
    estimate = student(f,masks,speed)['estimate']
    truth = torch.tensor(np.stack([r['teacher_features'] for r in sample]),device=args.device)
    from scripts.train_vision_student import SCALE
    truth[:,-1,:19] = estimate*torch.tensor(SCALE,device=args.device)
    teacher((truth-torch.tensor(normalizer.mean,device=args.device))/torch.tensor(normalizer.std,device=args.device),speed).square().mean().backward()
    assert student.estimate.weight.grad.abs().sum().item()>0
    # Reference raycaster vs optimized native rasterizer on varied courses/poses.
    raster = []
    for track in tracks[:3]:
        env = environment(track,config,settings,stage)
        try:
            for seed in [2,17,93]:
                spawn = sample_curriculum_spawn(env.track,stage,seed=seed,episode_index=seed)
                env.reset(seed=seed,options=dict(state=spawn.state,gate_index=spawn.gate_index))
                proprio = env._proprio(env._native.get_proprioception())
                native = env._render_gate_mask(proprio)
                reference,_ = env._render_gate_mask_and_depth(proprio)
                np.testing.assert_array_equal(native,reference)
                raster.append(dict(track=Path(track).name,seed=seed,pixels=int((native>0).sum())))
        finally:
            env.close()
    print(json.dumps(dict(status='pass', device=args.device, experiment_runs_launched=0,
        configurations=reports, collection=dict(steps=sum(r['steps'] for r in episodes),
            seconds=collection_seconds, steps_per_second=sum(r['steps'] for r in episodes)/collection_seconds),
        batch_size=config['batch_size'], updates=metrics,
        steady_update_seconds=float(np.median([m['seconds'] for m in metrics[1:]])),
        peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3 if str(args.device).startswith('cuda') else None,
        frozen_teacher_input_gradient=True, raster_comparisons=raster),indent=2))


if __name__=='__main__':
    main()
