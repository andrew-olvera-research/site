"""Isolated real-rollout PPO benchmark; never writes production checkpoints."""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts import train_privileged_racing as t
from starscream.training_acceleration import configure_policy_acceleration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rollout', type=Path, required=True)
    p.add_argument('--collect', action='store_true')
    p.add_argument('--overrides', default='{}')
    p.add_argument('--variant', choices=['baseline', 'cached_compiled', 'single_pass', 'single_pass_bf16', 'full_compile', 'stable_bf16'], default='baseline')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=2036092401)
    p.add_argument('--actor-lr-scale',type=float,default=1.)
    args = p.parse_args()
    overrides = json.loads(args.overrides if args.overrides.lstrip().startswith('{')
                           else Path(args.overrides).read_text())
    variants = dict(baseline={}, cached_compiled=dict(ppo_cached_minibatches=True, compile_ppo_critic=True),
        single_pass=dict(ppo_epoch_mode='weighted_single_pass', ppo_cached_minibatches=True,
                         compile_ppo_critic=True, minibatch_size=7680, critic_minibatch_size=7680),
        single_pass_bf16=dict(ppo_epoch_mode='weighted_single_pass', ppo_cached_minibatches=True,
                         compile_ppo_critic=True, minibatch_size=7680, critic_minibatch_size=7680,
                         ppo_critic_update_precision='bf16'),
        full_compile=dict(ppo_epoch_mode='weighted_single_pass', ppo_cached_minibatches=True,
                         compile_ppo_critic=True, minibatch_size=7680, critic_minibatch_size=7680,
                         compile_ppo_actor=True, ppo_current_route_projection_training=True,
                         ppo_audit_update_kernel=True))
    variants['stable_bf16']=dict(variants['full_compile'],compile_ppo_actor=False,
        compile_policy_backbone=False,ppo_actor_precision='stable_bf16')
    overrides = {**variants[args.variant], **overrides}
    if args.variant != 'baseline':
        overrides.setdefault('ppo_profile_updates', True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy, normalizer, payload, _ = t.load_policy_checkpoint(args.checkpoint, 'cuda')
    settings = copy.deepcopy(payload['training_config']['ppo'])
    policy.set_exact_likelihood_mode(True)
    precision=overrides.get('ppo_actor_precision','fp32')
    if precision=='stable_bf16':
        from starscream.ppo_bf16 import configure_stable_bf16
        configure_stable_bf16(policy)
    critic = t.build_ppo_critic(policy, settings, 'cuda')
    critic.load_state_dict(payload['critic'])
    if args.collect:
        stage = t.parse_stage(settings['curriculum'][0])
        collector = t.ProcessRaceCollector(policy, normalizer, settings, stage, 'cuda', sampling_prefix='ppo')
        try:
            sampling = payload.get('ppo_adaptive_sampling_state')
            if sampling:
                if list(stage.tracks) != sampling['tracks']:
                    raise ValueError('benchmark course order differs from checkpoint')
                collector.track_weights = np.asarray(sampling['weights'], np.float64)
            started = time.perf_counter()
            rollout, episodes, sps = collector.collect(critic, episodes=settings['rollout_envs'], seed_base=args.seed)
            collection = dict(seconds=time.perf_counter()-started, sps=sps, episodes=len(episodes))
            args.rollout.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(rollout=rollout, checkpoint=args.checkpoint, collection=collection,
                            precision=precision), args.rollout)
            print('collected', collection, flush=True)
        finally:
            collector.close()
    saved = torch.load(args.rollout, map_location='cpu', weights_only=False)
    if saved['checkpoint'] != args.checkpoint:
        raise ValueError('rollout behavior policy must match benchmark checkpoint')
    if saved.get('precision','fp32')!=precision:
        raise ValueError('rollout behavior precision must match benchmark policy')
    rollout = saved['rollout']
    settings.update(overrides)
    actor_state = {k:v.detach().clone() for k,v in policy.state_dict().items()}
    critic_state = {k:v.detach().clone() for k,v in critic.state_dict().items()}
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    configure_policy_acceleration(policy, settings)
    actor_optimizer = t.build_actor_optimizer(policy, settings, 'cuda')
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=settings['critic_learning_rate'],
        weight_decay=settings['critic_weight_decay'], fused=True)
    # Compare real continuation updates, including populated Adam moments.
    actor_optimizer.load_state_dict(payload['optimizer'])
    critic_optimizer.load_state_dict(payload['critic_optimizer'])
    results = []
    for i in range(args.repeats):
        policy.load_state_dict(actor_state); critic.load_state_dict(critic_state)
        actor_optimizer.load_state_dict(copy.deepcopy(payload['optimizer']))
        for group in actor_optimizer.param_groups:
            group['lr']*=args.actor_lr_scale
        critic_optimizer.load_state_dict(copy.deepcopy(payload['critic_optimizer']))
        torch.manual_seed(args.seed+1)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        metrics = t.update_ppo(rollout, policy, reference, critic, actor_optimizer, critic_optimizer,
                               settings, device='cuda', anchor_weight=0.)
        torch.cuda.synchronize()
        result = dict(repeat=i, seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), metrics=metrics)
        results.append(result)
        print(json.dumps(dict(repeat=i, seconds=result['seconds'],
            actor_epochs=metrics['actor_epochs_completed'], actor_updates=metrics['actor_updates_completed'],
            kl=metrics.get('epoch_approximate_kl'), critic_ev=metrics['critic_explained_variance'])), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(checkpoint=args.checkpoint, rollout=str(args.rollout),
            torch_version=torch.__version__, overrides=overrides,
            actor_lr_scale=args.actor_lr_scale,
            collection=saved['collection'], results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
