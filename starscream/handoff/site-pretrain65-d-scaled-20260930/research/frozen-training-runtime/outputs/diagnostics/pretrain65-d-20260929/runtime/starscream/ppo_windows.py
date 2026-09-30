"""Fixed-window on-policy collection with persistent lanes and dense buffers.

Explicit sampling variant: a lane contributes H transitions regardless of episode
length; completions reset immediately. Never manufacture an episode completion at
a window boundary. Policy/critic weights are frozen for the entire window.
"""
import time
import numpy as np
import torch


def apply_group_probability_floors(weights, labels, floors):
    """Project positive sampling weights onto configured group probability floors.

    The residual probability mass follows the unmodified group distribution and
    within-group relative weights are preserved. Unlisted groups therefore remain
    eligible (notably newly generated online courses).
    """
    weights = np.asarray(weights, np.float64)
    labels = np.asarray(labels, dtype=object)
    requested = {str(key): float(value) for key, value in dict(floors).items()}
    if weights.ndim != 1 or labels.shape != weights.shape or not len(weights):
        raise ValueError("sampling weights and group labels must be aligned vectors")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("sampling weights must be finite and positive")
    if any(not np.isfinite(value) or value < 0.0 for value in requested.values()):
        raise ValueError("group probability floors must be finite and nonnegative")
    total_floor = float(sum(requested.values()))
    if total_floor > 1.0 + 1.0e-12:
        raise ValueError("group probability floors must sum to at most one")
    present = set(map(str, labels))
    missing = sorted(key for key, value in requested.items() if value > 0 and key not in present)
    if missing:
        raise ValueError(f"sampling floor groups are absent from the corpus: {missing}")

    raw = weights / weights.sum()
    groups = tuple(dict.fromkeys(map(str, labels)))
    raw_mass = {
        group: float(raw[labels == group].sum())
        for group in groups
    }
    residual = max(1.0 - total_floor, 0.0)
    projected = np.empty_like(raw)
    for group in groups:
        mask = labels == group
        target_mass = requested.get(group, 0.0) + residual * raw_mass[group]
        projected[mask] = raw[mask] * (target_mass / raw_mass[group])
    # Preserve the caller's scale; downstream scheduling only uses ratios.
    return projected * weights.sum()


def actor_early_stop_scope(settings):
    """Return the PPO trust-region checkpoint granularity.

    Epoch scope is the safe default for fixed-window PPO: every rollout sample
    is consumed once before a full-cohort KL probe may stop later epochs.  The
    legacy minibatch mode remains explicit for reproducing completed runs, but
    must never be selected accidentally by omission.
    """
    scope = str(settings.get("actor_early_stop_scope", "epoch"))
    if scope not in {"minibatch", "epoch", "none"}:
        raise ValueError("actor_early_stop_scope must be minibatch, epoch, or none")
    if bool(settings.get("require_complete_actor_epoch", False)) and scope == "minibatch":
        raise ValueError(
            "require_complete_actor_epoch is incompatible with minibatch early stopping"
        )
    if (
        bool(settings.get("require_complete_actor_epoch", False))
        and int(settings.get("actor_epochs", 6)) < 1
    ):
        raise ValueError("require_complete_actor_epoch needs actor_epochs >= 1")
    return scope


def lane_reassignments(current, desired):
    """Meet an integer course quota while interrupting the fewest live lanes."""
    from collections import Counter
    if len(current) != len(desired):
        raise ValueError('course lane schedules must have equal size')
    remaining = Counter(desired)
    surplus = []
    for lane, track in enumerate(current):
        if remaining[track] > 0:
            remaining[track] -= 1
        else:
            surplus.append(lane)
    deficits = [track for track, count in remaining.items() for _ in range(count)]
    return dict(zip(surplus, deficits))


def validate_window_settings(settings, lanes):
    # Validate before worker processes start or the first rollout is collected.
    actor_early_stop_scope(settings)
    horizon = int(settings.get('ppo_rollout_window_steps', 0))
    if not 1 <= horizon <= 16384:
        raise ValueError('window horizon must be in 1..16384')
    # Per-lane task replacement (Green Algorithm 2) and dual constraints need
    # one completed episode per slot per update; a 768-tick window completes
    # fewer than one lap per lane, so those stay episode-mode only. Frontier
    # (PLR-style) course weights are applied as between-window lane quotas.
    for key in ('green2026_adaptive_task_switching', 'ppo_group_constraints'):
        if settings.get(key, {}).get('enabled', False):
            raise ValueError(f'window mode does not support episode-based {key}')
    if settings.get('ppo_task_blocks') or settings.get('ppo_persistent_task_slots'):
        raise ValueError('window lane quotas cannot use episode-quota task blocks/persistent task slots')
    if lanes < 1:
        raise ValueError('window needs positive lane count')
    cohorts = int(settings.get('ppo_inference_cohorts', 1))
    per_worker = int(settings.get('ppo_envs_per_worker', 1))
    if cohorts < 1 or lanes % cohorts or (lanes // cohorts) % per_worker:
        raise ValueError('inference cohorts must divide lanes and align with worker groups')


def check_window_storage(horizon,lanes,history_elements,critic_elements,task_dim,limit):
    # Conservative allowance for all saved fields plus GAE workspaces/outputs.
    estimate = horizon*lanes*(4*(history_elements+critic_elements+task_dim+64)+128)
    if limit <= 0 or estimate > limit:
        raise MemoryError(f'window rollout estimate {estimate:,} bytes exceeds host budget {limit:,}; '
                          'reduce horizon/lanes or explicitly raise ppo_window_max_host_bytes')
    return estimate


def window_gae(reward, value, next_value, terminal, episode_end, gamma, lam):
    """Bootstrap truncations, but never propagate GAE into a reset episode."""
    shape = np.shape(reward)
    if len(shape) != 2 or any(np.shape(x) != shape for x in (value,next_value,terminal,episode_end)):
        raise ValueError('window GAE requires matching[T,N] arrays')
    if np.any(np.asarray(terminal) & ~np.asarray(episode_end)):
        raise ValueError('true terminal must end an episode')
    reward, value, next_value = (np.asarray(x,np.float64) for x in (reward,value,next_value))
    advantage = np.empty(shape,np.float64)
    tail = np.zeros(shape[1],np.float64)
    for t in range(shape[0]-1,-1,-1):
        delta = reward[t] + gamma*np.where(terminal[t],0.,next_value[t])-value[t]
        tail = delta + gamma*lam*np.where(episode_end[t],0.,tail)
        advantage[t] = tail
    return advantage.astype(np.float32), (advantage+value).astype(np.float32)


def critic_calibration_metrics(reward, value, next_value, terminal, episode_end, gamma):
    """Pre-fit diagnostics; MC uses only observed terminal-reaching suffixes.

    No bootstrap in MC targets, no crossing episode resets. The subset is
    completion/time-to-end biased and must not be described as all-state EV.
    Predictions were frozen during collection, before fitting this window.
    """
    reward, value, next_value = [np.asarray(x, np.float64) for x in (reward, value, next_value)]
    terminal, episode_end = np.asarray(terminal, bool), np.asarray(episode_end, bool)
    if reward.ndim != 2 or any(x.shape != reward.shape for x in (value, next_value, terminal, episode_end)):
        raise ValueError('calibration requires matching [T,N] arrays')
    if not 0 <= gamma <= 1 or np.any(terminal & ~episode_end):
        raise ValueError('calibration requires valid discount and terminal episode ends')
    targets = np.zeros_like(reward)
    valid = np.zeros_like(terminal)
    tail = np.zeros(reward.shape[1])
    known = np.zeros(reward.shape[1], bool)
    for t in range(len(reward)-1, -1, -1):
        known = np.where(episode_end[t], terminal[t], known)
        tail = reward[t] + gamma * np.where(episode_end[t], 0., tail)
        targets[t], valid[t] = tail, known
    td = reward + gamma*np.where(terminal, 0., next_value) - value
    result = dict(critic_prefit_td_mean=float(td.mean()),
                  critic_prefit_td_rms=float(np.sqrt(np.mean(td**2))),
                  critic_mc_observed_samples=int(valid.sum()),
                  critic_mc_observed_fraction=float(valid.mean()))
    if valid.any():
        error = value[valid]-targets[valid]
        variance = targets[valid].var()
        result.update(critic_mc_prefit_bias=float(error.mean()),
                      critic_mc_prefit_rmse=float(np.sqrt(np.mean(error**2))))
        if variance > 1e-8:
            result['critic_mc_prefit_explained_variance'] = float(1-error.var()/variance)
    return result


def critic_warmup_settings(settings, cycle):
    """Skip actor updates on fresh windows, never recompute stale advantages."""
    count = int(settings.get('ppo_critic_warmup_cycles', 0))
    if count < 0:
        raise ValueError('critic warmup cycles must be nonnegative')
    active = cycle <= count
    return (
        {
            **settings,
            'actor_epochs': 0,
            # A deliberate critic-only warmup is the sole supported exception
            # to the complete actor epoch contract.
            'require_complete_actor_epoch': False,
        }
        if active else settings
    ), active


def release_rollout_storage(rollout):
    """Drop consumed samples and return freed glibc arenas at a cycle boundary.

    Python/CUDA wrappers can contain cycles, and freeing NumPy/Torch buffers
    alone need not return CPU heap pages to the OS. This changes no PPO math.
    """
    import ctypes
    import gc

    rollout.clear()
    gc.collect()
    trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if trim is not None:
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)


class WindowBuffer:
    def __init__(self, horizon, lanes):
        self.horizon, self.lanes = horizon, lanes
        self.data = {}
        self.index = 0

    def append(self, fields):
        if self.index >= self.horizon:
            raise ValueError('window overflow')
        if self.data and set(fields) != set(self.data):
            raise ValueError('window schema changed')
        for key, value in fields.items():
            value = np.asarray(value)
            if value.shape[0] != self.lanes:
                raise ValueError('window lane count changed')
            if key not in self.data:
                self.data[key] = np.empty((self.horizon,*value.shape),value.dtype)
            self.data[key][self.index] = value
        self.index += 1


@torch.no_grad()
def collect_window(collector, critic, *, episodes, seed_base):
    from scripts.train_privileged_racing import (ppo_critic_features, SquashedGaussian,
        configured_ppo_exploration_correlation, correlated_ppo_distribution,
        ppo_reliability_reward, completed_episode_is_terminal, ppo_episode_tracks, TASK_DIM)
    from starscream.ppo_collection import flush_lanes
    c = collector
    s = c.settings
    lap_constraints = getattr(c, 'lap_time_constraints', None)
    horizon = int(s['ppo_rollout_window_steps'])
    validate_window_settings(s, c.parallel)
    if not 1 <= horizon <= 16384 or episodes != c.parallel:
        raise ValueError('window mode requires horizon1..16384 and episodes_per_cycle==rollout_envs (lane slots, not episode quota)')
    if c.policy.action_head_type != 'mlp':
        raise ValueError('window mode requires direct Gaussian PPO')
    if c._paper_switcher is not None or s.get('ppo_group_constraints',{}).get('enabled',False):
        raise ValueError('window mode does not yet support episode-based switchers/failure-return constraints')
    c.policy.eval(); critic.eval()
    start = time.perf_counter()
    if not getattr(c,'_window_started',False):
        c._schedule(c.parallel, seed_base)
        c._window_started = True
        c._window_episode_index = c.parallel
        c._window_seed_base = seed_base
        c._window_adjusted_returns = np.zeros(c.parallel,np.float64)
        c._window_weights = c.track_weights.copy()
    reassignments = {}
    if not np.array_equal(c.track_weights, c._window_weights):
        desired = ppo_episode_tracks(c.stage.tracks,c.parallel,s,seed_base,
                                     weights_override=c.track_weights)
        reassignments = lane_reassignments([slot.track for slot in c.slots],desired)
        # The preceding window already bootstrapped these live final states.
        # A quota change is censoring, NEVER an invented failure or completion.
        for lane, track in reassignments.items():
            c._reset_slot(c.slots[lane],c._window_episode_index,c._window_seed_base,track)
            c._window_episode_index += 1
            c._window_adjusted_returns[lane] = 0.
        c._window_weights = c.track_weights.copy()
    active = c.slots
    n = len(active)
    sample = c.normalizer.numpy(active[0].history.array())
    check_window_storage(horizon,n,sample.size,len(ppo_critic_features(sample,active[0].context,s)),
                         TASK_DIM,int(s.get('ppo_window_max_host_bytes',2*1024**3)))
    buffer = WindowBuffer(horizon,n)
    results = []
    gamma, lam = float(s.get('discount',.997)), float(s.get('gae_lambda',.95))
    track_ids = np.asarray([c.stage.tracks.index(slot.track) for slot in active],np.int64)
    # Explicit per-course transition quotas, adjustable between frozen windows.
    # Every assigned lane contributes exactly horizon transitions on every call.
    for tick in range(horizon):
        raw_h = np.stack([slot.history.array() for slot in active])
        normalized = c.normalizer.numpy(raw_h)
        speeds = np.asarray([slot.target_speed for slot in active],np.float32)
        ci = np.stack([ppo_critic_features(hist,slot.context,s) for hist,slot in zip(normalized,active)])
        correlation = configured_ppo_exploration_correlation(s)
        cohorts = int(s.get('ppo_inference_cohorts',1))
        stride = n // cohorts
        packed_parts = []
        # Draw once in lane order, independent of scheduling. CPU stepping of
        # earlier cohorts overlaps GPU inference of later cohorts; actor weights
        # remain frozen throughout. No learner/rollout policy staleness.
        noise = torch.randn((n,4),device=c.device,dtype=torch.float32) if cohorts > 1 else None
        for first in range(0,n,stride):
            last = first+stride
            group = active[first:last]
            offset_array = np.stack([correlation*slot.exploration_residual for slot in group])
            if s.get('ppo_fused_window_inputs', False):
                if not hasattr(c, '_window_input_transfer'):
                    from .inference_graph import PPOHostInputTransfer
                    c._window_input_transfer = PPOHostInputTransfer(c.device)
                h, speed_tensor, cit, offsets = c._window_input_transfer.transfer(
                    normalized[first:last], speeds[first:last], ci[first:last], offset_array)
            else:
                h = torch.from_numpy(normalized[first:last]).to(c.device)
                speed_tensor = torch.from_numpy(speeds[first:last]).to(c.device)
                cit = torch.from_numpy(ci[first:last]).to(c.device)
                offsets = torch.as_tensor(offset_array, device=c.device)
            if c.ppo_inference_graphs is None:
                base = c.policy.distribution(h,speed_tensor)
                value = critic(cit)
            else:
                location, logstd, value = c.ppo_inference_graphs(h,speed_tensor,cit,critic)
                base = SquashedGaussian(location,logstd)
            offsets = offsets.to(base.location.dtype)
            distribution = correlated_ppo_distribution(base,offsets,correlation)
            if noise is None:
                action, raw = distribution.rsample()
            else:
                raw = distribution.location + distribution.std * noise[first:last].to(base.location.dtype)
                action = torch.tanh(raw)
            logprob = distribution.log_prob(action,raw)
            part = torch.cat([value[:,None],action,raw,logprob[:,None],base.location,
                             offsets,distribution.location,distribution.log_std.expand_as(raw)],-1).float().cpu().numpy()
            packed_parts.append(part)
            for i,slot in enumerate(group):
                slot.exploration_residual = (part[i,5:9]-part[i,10:14]).astype(np.float32)
                slot.connection.send(('advance',part[i,1:5]))
            flush_lanes(group)
        packed = packed_parts[0] if cohorts == 1 else np.concatenate(packed_parts)
        values, actions = packed[:,0],packed[:,1:5]
        if tick:
            previous_alive = ~buffer.data['episode_end'][tick-1]
            buffer.data['next_value'][tick-1,previous_alive] = values[previous_alive]
        contexts = np.stack([slot.context for slot in active])
        responses = [c._receive(slot,'step') for slot in active]
        reward = np.empty(n,np.float64)
        terminal, ended = np.zeros(n,bool),np.zeros(n,bool)
        next_value = np.zeros(n,np.float64)
        features = np.stack([r[2] for r in responses])
        for i,(slot,response) in enumerate(zip(active,responses)):
            _, r, feature, context, done, result = response
            slot.history.append_feature(feature)
            slot.context = np.asarray(context,np.float32)
            ended[i] = done
            terminal[i] = completed_episode_is_terminal(bool(done),result,
                timeout_is_terminal=bool(s.get('ppo_timeout_is_terminal',False)))
            reward[i] = ppo_reliability_reward(float(r),float(contexts[i,0]),float(context[0]),bool(done),result,s)
            if lap_constraints is not None:
                # Every completed attempt is terminal under this finite-lap
                # objective, including timeout and non-collision termination.
                terminal[i] = bool(done)
                reward[i] = lap_constraints.reward(slot, result, bool(done), c.stage.max_steps)
            c._window_adjusted_returns[i] += reward[i]
        # Nonterminal time-limit: value of final observation, BEFORE reset.
        bootstrap = np.flatnonzero(ended & ~terminal)
        if len(bootstrap):
            bh = c.normalizer.numpy(np.stack([active[i].history.array() for i in bootstrap]))
            bc = np.stack([ppo_critic_features(hist,active[i].context,s) for hist,i in zip(bh,bootstrap)])
            next_value[bootstrap] = critic(torch.from_numpy(bc).to(c.device)).float().cpu().numpy()
        buffer.append(dict(history=normalized,critic_input=ci,action=actions,
            raw_action=packed[:,5:9],old_log_prob=packed[:,9],old_location=packed[:,18:22],
            old_log_std=packed[:,22:26],exploration_offset=packed[:,14:18],
            speed_command=speeds,track_id=track_ids,
            track_weight=c.track_weights[track_ids].astype(np.float32),
            rollout_laps=np.asarray([slot.rollout_laps for slot in active],np.float32),
            task_delta=(features[:,:TASK_DIM]-raw_h[:,-1,:TASK_DIM]),
            dynamics_valid=np.asarray([slot.context[0]<=contexts[i,0]+1e-7 for i,slot in enumerate(active)],np.float32),
            reward=reward,value=values.astype(np.float64),next_value=next_value,
            terminal=terminal,episode_end=ended))
        for i in np.flatnonzero(ended):
            result = dict(responses[i][5])
            result['ppo_adjusted_return'] = float(c._window_adjusted_returns[i])
            if lap_constraints is not None:
                result['environment_return'] = result['return']
                result['return'] = result['ppo_adjusted_return']
                result['lap_time_dual'] = active[i].lap_time_dual
                from pathlib import Path
                reference = lap_constraints.courses[Path(active[i].track).stem]['reference_seconds']
                elapsed_cost = -result['steps'] / (lap_constraints.hz * reference)
                result['environment_reward_component_sums'] = result.get('reward_component_sums', {})
                result['reward_component_sums'] = dict(time=elapsed_cost,
                    terminal_failure=result['ppo_adjusted_return'] - elapsed_cost)
                del active[i].lap_time_dual
            c._window_adjusted_returns[i] = 0.
            results.append(result)
            c._reset_slot(active[i],c._window_episode_index,c._window_seed_base,active[i].track)
            c._window_episode_index += 1
    # Artificial collection boundary is NOT an MDP terminal; bootstrap the
    # still-live final states using the frozen critic, and retain their histories.
    live = np.flatnonzero(~buffer.data['episode_end'][-1])
    if len(live):
        bh = c.normalizer.numpy(np.stack([active[i].history.array() for i in live]))
        bc = np.stack([ppo_critic_features(hist,active[i].context,s) for hist,i in zip(bh,live)])
        buffer.data['next_value'][-1,live] = critic(torch.from_numpy(bc).to(c.device)).float().cpu().numpy()
    data = buffer.data
    adv, returns = window_gae(data['reward'],data['value'],data['next_value'],data['terminal'],data['episode_end'],gamma,lam)
    discard = {'reward','value','next_value','terminal','episode_end'}
    rollout = {key:torch.from_numpy(array.reshape(horizon*n,*array.shape[2:])) for key,array in data.items() if key not in discard}
    rollout['advantage'] = torch.from_numpy(adv.reshape(-1))
    rollout['return'] = torch.from_numpy(returns.reshape(-1))
    rollout['old_value'] = torch.from_numpy(data['value'].reshape(-1).astype(np.float32))
    rollout['failure_return'] = torch.zeros(horizon*n,dtype=torch.float32)
    rollout['family_id'] = torch.from_numpy(c.family_ids[rollout['track_id'].numpy()])
    c.last_window_metrics = dict(window_steps=horizon,window_transitions=horizon*n,
        **critic_calibration_metrics(data['reward'], data['value'], data['next_value'],
                                     data['terminal'], data['episode_end'], gamma),
        reward_mean=float(data['reward'].mean()),
        reward_std=float(data['reward'].std()),
        reward_min=float(data['reward'].min()),
        reward_max=float(data['reward'].max()),
        true_terminal_count=int(data['terminal'].sum()),
        episode_end_count=int(data['episode_end'].sum()),
        completed_episodes=len(results),active_batch_mean=n,
        quota_reassigned_lanes=len(reassignments),
        inference_cohorts=int(s.get('ppo_inference_cohorts',1)),
        rollout_buffer_bytes=sum(array.nbytes for array in buffer.data.values()),
        course_transition_counts=np.bincount(rollout['track_id'].numpy(),minlength=len(c.stage.tracks)).tolist())
    for track in np.unique(track_ids):
        lanes = track_ids == track
        calibration = critic_calibration_metrics(
            data['reward'][:,lanes], data['value'][:,lanes], data['next_value'][:,lanes],
            data['terminal'][:,lanes], data['episode_end'][:,lanes], gamma)
        c.last_window_metrics.update({f'track_{track}_{key}': value for key,value in calibration.items()})
        c.last_window_metrics[f'track_{track}_raw_advantage_mean'] = float(adv[:,lanes].mean())
        c.last_window_metrics[f'track_{track}_raw_advantage_std'] = float(adv[:,lanes].std())
    # Cheap end-of-window diagnostics in already stored CPU arrays. These are
    # commanded normalized perturbations, NOT applied thrust/attitude responses.
    innovations = data['raw_action'] - data['old_location']
    base_location = data['old_location'] - data['exploration_offset']
    commanded_delta = data['action'] - np.tanh(base_location)
    for axis, name in enumerate(('collective', 'roll', 'pitch', 'yaw')):
        c.last_window_metrics[f'exploration_{name}_innovation_std'] = float(innovations[...,axis].std())
        c.last_window_metrics[f'exploration_{name}_normalized_rms'] = float(np.sqrt(np.mean(commanded_delta[...,axis]**2)))
        c.last_window_metrics[f'exploration_{name}_saturation_fraction'] = float(np.mean(np.abs(data['action'][...,axis]) > .99))
    return rollout, results, horizon*n/max(time.perf_counter()-start,1e-6)
