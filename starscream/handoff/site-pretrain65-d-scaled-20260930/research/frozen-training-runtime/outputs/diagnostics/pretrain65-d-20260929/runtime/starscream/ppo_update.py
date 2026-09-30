"""PPO update scheduling and instrumentation, independent of simulator contracts."""
from contextlib import contextmanager
import math
import time

import numpy as np
import torch


class PPOBatchPlan:
    """Cache immutable membership; retain CUDA RNG ordering for legacy draws.

    weighted_single_pass visits every row once and uses explicit objective
    weights instead of repeating other groups until the last group is exhausted.
    """
    def __init__(self, strata, weights=None):
        host = strata.detach().cpu().numpy()
        self.ids, inverse, counts = np.unique(host, return_inverse=True, return_counts=True)
        if not len(self.ids):
            raise ValueError('PPO batch plan requires samples')
        self.count = len(host)
        self.device = strata.device
        self.counts = counts
        self.weights = np.asarray([float((weights or {}).get(int(k), 1.)) for k in self.ids], np.float64)
        if not np.all(np.isfinite(self.weights)) or np.any(self.weights <= 0):
            raise ValueError('PPO group weights must be finite and positive')
        self.groups = [torch.from_numpy(np.flatnonzero(inverse == i)).to(strata.device)
                       for i in range(len(self.ids))]
        row_weights = self.count * (self.weights / self.weights.sum())[inverse] / counts[inverse]
        self.loss_weights = torch.from_numpy(row_weights.astype(np.float32)).to(strata.device)
        self.quotas = {}

    def quota(self, batch_size):
        size = max(int(batch_size), len(self.ids))
        if size not in self.quotas:
            quotas = np.ones(len(self.ids), np.int64)
            remaining = size - len(self.ids)
            raw = remaining * self.weights / self.weights.sum()
            extra = np.floor(raw).astype(np.int64)
            quotas += extra
            residual = remaining - int(extra.sum())
            quotas[np.argsort(-(raw-extra), kind='stable')[:residual]] += 1
            self.quotas[size] = quotas
        return self.quotas[size]

    def stratified(self, batch_size, maximum_batches=None):
        quotas = self.quota(batch_size)
        batch_count = max(math.ceil(n/int(q)) for n,q in zip(self.counts, quotas))
        if maximum_batches is not None:
            if int(maximum_batches) < 1:
                raise ValueError('maximum PPO minibatches must be positive')
            batch_count = min(batch_count, int(maximum_batches))
        draws = []
        for group, quota in zip(self.groups, quotas):
            remaining = batch_count * int(quota)
            pieces = []
            while remaining > 0:
                shuffled = group[torch.randperm(len(group), device=group.device)]
                pieces.append(shuffled[:remaining])
                remaining -= min(remaining, len(shuffled))
            draws.append(torch.cat(pieces))
        batches = []
        for i in range(batch_count):
            indices = torch.cat([draw[i*int(q):(i+1)*int(q)] for draw,q in zip(draws,quotas)])
            batches.append(indices[torch.randperm(len(indices),device=self.device)])
        return batches

    def single_pass(self, batch_size):
        if batch_size < 1:
            raise ValueError('PPO minibatch size must be positive')
        return list(torch.randperm(self.count, device=self.device).split(batch_size))


class UpdateTimer:
    """Opt-in synchronized phase attribution; disabled in normal training."""
    def __init__(self, enabled, device):
        self.enabled, self.device = enabled, torch.device(device)
        self.metrics = {}
        if enabled and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.started = time.perf_counter()

    def mark(self, name):
        if self.enabled:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            self.metrics[name] = now-self.started
            self.started = now

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.metrics[name] = self.metrics.get(name, 0.) + time.perf_counter()-start


def configure_critic_acceleration(critic, settings):
    """Compile only gradient-enabled value calls; preserve graph/eager collection.

    BF16 is opt-in for critic fitting only. Loss and value outputs stay FP32;
    collection/bootstrap use the FP32 critic, avoiding any actor precision change.
    """
    compile_update = bool(settings.get('compile_ppo_critic', False))
    precision = str(settings.get('ppo_critic_update_precision', 'fp32'))
    if precision not in {'fp32', 'bf16'}:
        raise ValueError('ppo_critic_update_precision must be fp32 or bf16')
    signature = (compile_update, precision)
    previous = getattr(critic, '_ppo_update_acceleration', None)
    if previous is not None:
        if previous != signature:
            raise ValueError('recreate critic before changing update acceleration')
        return
    if not compile_update and precision == 'fp32':
        return
    if next(critic.parameters()).device.type != 'cuda':
        raise ValueError('PPO critic acceleration requires CUDA')
    original = critic.forward

    def update(value):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
            return original(value).float()

    if compile_update:
        import torch._inductor.config as config
        config.compile_threads = int(settings.get('compile_threads', 1))
        update = torch.compile(update, fullgraph=True, dynamic=False,
                               options={'triton.cudagraphs': False})

    def forward(value):
        return update(value) if torch.is_grad_enabled() else original(value)

    critic.forward = forward
    critic._ppo_update_acceleration = signature


def configure_actor_acceleration(policy, settings):
    """Compile the complete FP32 actor distribution on the update path only."""
    if not settings.get('compile_ppo_actor', False) or getattr(policy, '_ppo_actor_compiled', False):
        return
    if next(policy.parameters()).device.type != 'cuda':
        raise ValueError('PPO actor compilation requires CUDA')
    import torch._inductor.config as config
    config.compile_threads = int(settings.get('compile_threads', 1))
    original = policy.distribution
    compiled = torch.compile(original, fullgraph=True, dynamic=False,
                             options={'triton.cudagraphs': False})

    def distribution(*args, **kwargs):
        return (compiled if torch.is_grad_enabled() else original)(*args, **kwargs)

    policy.distribution = distribution
    policy._ppo_actor_compiled = True
