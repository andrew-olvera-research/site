"""Three real PPO windows at production scale, traversing all reward stages.

Isolated outputs, W&B disabled, source checkpoint read-only. This checks runtime
contracts and resource use, not the scientific quality of 50M-step learning.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train_privileged_racing import load_policy_checkpoint, run_ppo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/exp/v6.20.rl.1/ppo.yaml')
    parser.add_argument('--test-warmup', action='store_true')
    parser.add_argument('--warmup-cycles', type=int, default=1)
    parser.add_argument('--steady-stage', action='store_true')
    args = parser.parse_args()
    if args.warmup_cycles < 1:
        parser.error('--warmup-cycles must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    s = config['ppo']
    actor, normalizer, checkpoint, path = load_policy_checkpoint(s['initial_checkpoint'], 'cpu')
    assert actor.input_dim == 103 and actor.context_steps == 3
    assert actor.observation_contract == 'starscream_route_v1'
    assert actor.action_head_type == 'mlp'
    assert actor.unified_readout_mode == 'action_token'
    assert sum(p.numel() for p in actor.parameters()) == 10_840_603
    print('Verified v6201 direct-action privileged103/H3/action-query actor', flush=True)
    del actor, normalizer, checkpoint
    args.output.mkdir(parents=True)
    config['output_root'] = str(args.output.resolve())
    config['checkpoint'].update(run_name='v620-rl-production-smoke',top_k=1)
    config['wandb'] = dict(enabled=False,run_name='v620-rl-production-smoke',
                         event_path=str((args.output/'events.jsonl').resolve()))
    cycles = max(3, args.warmup_cycles+1) if args.test_warmup else 3
    s.update(run_name='v620-rl-production-smoke',cycles=cycles,
             target_environment_steps=cycles*s['rollout_envs']*s['ppo_rollout_window_steps'],
             evaluation_interval=1,evaluation_episodes=35,
             reporting_evaluation_interval=1,reporting_evaluation_episodes=6,top_k=1)
    if args.test_warmup:
        s['ppo_critic_warmup_cycles'] = args.warmup_cycles
        import scripts.train_privileged_racing as trainer
        original_update = trainer.update_ppo
        def audited_update(rollout, policy, reference, critic, *pos, **kw):
            before = [p.detach().clone() for p in policy.parameters()]
            noise_before = policy.log_std_parameter.detach().clone()
            critic_before = [p.detach().clone() for p in critic.parameters()]
            result = original_update(rollout, policy, reference, critic, *pos, **kw)
            unchanged = all(torch.equal(a, b) for a,b in zip(before, policy.parameters()))
            if result['actor_updates_completed'] == 0:
                assert unchanged, 'warmup mutated actor'
                assert any(not torch.equal(a,b) for a,b in zip(critic_before,critic.parameters()))
                print('PASS: warmup actor bitwise unchanged; critic updated', flush=True)
            else:
                assert not unchanged, 'post-warmup actor did not update'
                if s.get('train_log_std', False):
                    assert not torch.equal(noise_before,policy.log_std_parameter), 'learned noise did not update'
                    from starscream.ppo_exploration import noise_bounds
                    low, high = noise_bounds(s,policy)
                    values = policy.log_std_parameter.detach().cpu().numpy()
                    assert (values >= low-1e-7).all() and (values <= high+1e-7).all()
                    print('PASS: learned noise updated within configured bounds',flush=True)
            return result
        trainer.update_ppo = audited_update
    if not args.steady_stage:
        for stage in s['curriculum']:
            stage['minimum_environment_steps'] = 0
    (args.output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    torch.set_num_threads(1)
    run_ppo(config, 'cuda')
    events = [json.loads(line) for line in (args.output/'events.jsonl').read_text().splitlines()]
    rows = [e['metrics'] for e in events if 'train/cycle' in e.get('metrics',{})]
    assert len(rows) == cycles
    for i, row in enumerate(rows):
        assert row['train/curriculum_stage'] == (0 if args.steady_stage else min(i,2))
        assert row['train/rollout_transitions'] == 215040
        courses = len(s['curriculum'][0]['tracks'])
        counts = [row[f'train/window_track_{track}_transitions'] for track in range(courses)]
        assert sum(counts) == 215040 and min(counts) >= 768
        if not s.get('ppo_adaptive_level_replay', {}).get('enabled', False):
            assert set(counts) == {215040 // courses}
        if args.test_warmup:
            assert row['train/critic_only_warmup'] == float(i < args.warmup_cycles)
            assert (row['train/actor_updates_completed'] == 0) == (i < args.warmup_cycles)
            assert 'train/window_critic_prefit_td_rms' in row
    print(f'PASS: requested stages/warmup, {courses}-course quotas, production-scale PPO updates and final eval', flush=True)


if __name__ == '__main__':
    main()
