"""One-window DAgger lookahead with isolated CUDA ownership and bounded handoff.

The learner alone owns replay and its sampler. The collector owns a frozen actor
and publishes complete batches through read-only file-backed arrays. Checkpoints
record the pending request and actor, never an uncommitted replay generation.
"""
from __future__ import annotations

import copy
from dataclasses import fields
import multiprocessing as mp
from pathlib import Path
import pickle
import shutil
import tempfile
import time
import traceback

import numpy as np

from starscream.dagger_row_buffer import NumericRowBuffer


def lookahead_request(settings, collector_settings, *, window, seed, tracks,
                      expert_refresh_end):
    """Freeze collection schedules; feedback weights intentionally lag one window."""
    from scripts import train_privileged_racing as t
    from starscream.dagger_schedule import recovery_anneal, speed_fraction_weights, dart_collection_round
    permanent = int(settings.get('dagger_permanent_expert_rounds', 0))
    coverage = int(settings.get('dagger_successful_coverage_rounds', 0))
    if window <= max(permanent + 1, coverage + 1, expert_refresh_end + 1):
        return None  # Keep coverage/refresh transitions synchronous.
    frozen = copy.deepcopy(collector_settings)
    speed = speed_fraction_weights(window, settings)
    if speed:
        frozen['dagger_teacher_speed_fraction_weights'] = speed
    recovery = recovery_anneal(window, settings)
    frozen['dagger_pure_expert_collection'] = bool(recovery.get('pure_expert'))
    frozen['dagger_require_successful_episodes'] = t.dagger_successful_episode_filter(
        window, settings, refresh_permanent_expert=False)
    beta = t.dagger_teacher_beta(window, settings, refresh_permanent_expert=False)
    dart = dart_collection_round(window, settings)
    kwargs = dict(episodes=int(settings.get('episodes_per_round', 96)), beta=beta,
                  seed_base=seed + window * 100000, dart=dart, tracks=tuple(tracks))
    return frozen, kwargs


class MappedRows:
    """List-compatible immutable row chunks; array views retain their mapping."""
    def __init__(self, array):
        self.parts = [array]

    def __len__(self):
        return sum(len(part) for part in self.parts)

    def __iter__(self):
        for part in self.parts:
            yield from part

    def __array__(self, dtype=None, copy=None):
        if copy is False and len(self.parts) != 1:
            raise ValueError('Multiple row chunks require a copy')
        array = (self.parts[0] if len(self.parts) == 1 else
                 np.concatenate(self.parts) if self.parts else np.empty(0))
        if copy is False and dtype is not None and np.dtype(dtype) != array.dtype:
            raise ValueError('Dtype conversion requires a copy')
        result = np.asarray(array, dtype=dtype)
        return result.copy() if copy else result

    def __getitem__(self, index):
        return np.asarray(self)[index]

    def clear(self):
        # Do not close mmap: the caller can hold an array view after clear().
        self.parts.clear()

    def extend(self, rows):
        if isinstance(rows, MappedRows):
            self.parts.extend(rows.parts)
        elif len(rows):
            self.parts.append(np.asarray(rows).copy())


def write_batch(batch, directory, max_bytes):
    """Publish arrays first, metadata last. Paths are private to this process tree."""
    directory = Path(directory)
    directory.mkdir()
    metadata, total = {}, 0
    for field in fields(batch):
        value = getattr(batch, field.name)
        if field.name == 'tracks':
            names = list(dict.fromkeys(value))
            ids = {name: index for index, name in enumerate(names)}
            array = np.fromiter((ids[x] for x in value), dtype=np.int32, count=len(value))
            metadata[field.name] = ('tracks', names)
        elif isinstance(value, (list, NumericRowBuffer, MappedRows)) and len(value):
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise TypeError(f'Non-numeric DAgger field {field.name}')
            metadata[field.name] = ('rows', None)
        else:
            metadata[field.name] = ('value', value)
            continue
        total += array.nbytes
        if total > max_bytes:
            raise MemoryError(f'DAgger handoff exceeds {max_bytes} bytes')
        np.save(directory / f'{field.name}.npy', array, allow_pickle=False)
        # Batch no longer needs this field; reduce the peak while serializing.
        value.clear()
    payload = pickle.dumps(metadata, protocol=5)
    total += len(payload)
    if total > max_bytes:
        raise MemoryError(f'DAgger handoff exceeds {max_bytes} bytes')
    (directory / 'metadata.pkl').write_bytes(payload)
    return total


def read_batch(directory, batch_type):
    directory = Path(directory)
    metadata = pickle.loads((directory / 'metadata.pkl').read_bytes())
    batch = batch_type()
    for name, (kind, value) in metadata.items():
        if kind == 'rows':
            value = MappedRows(np.load(directory / f'{name}.npy', mmap_mode='r', allow_pickle=False))
        elif kind == 'tracks':
            ids = np.load(directory / f'{name}.npy', mmap_mode='r', allow_pickle=False)
            value = [value[int(index)] for index in ids]
        elif kind != 'value':
            raise ValueError(f'Unknown handoff field kind {kind}')
        setattr(batch, name, value)
    return batch


def _collector_main(connection, initial_path, settings, stage, device, root, precision):
    import signal
    import torch
    from scripts import train_privileged_racing as t
    collector = None
    try:
        t._terminate_worker_with_parent()
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision(precision)
        policy, normalizer, _, _ = t.load_policy_checkpoint(initial_path, device)
        from starscream.training_acceleration import configure_policy_acceleration
        configure_policy_acceleration(policy, settings)
        policy.eval()
        collector = t.ProcessDaggerCollector(policy, normalizer, settings, stage, device)
        connection.send(('ready',))
        while True:
            request = connection.recv()
            if request[0] == 'close':
                break
            _, token, snapshot_path, frozen_settings, kwargs, maximum = request
            policy.load_state_dict(torch.load(snapshot_path, map_location=device, weights_only=True))
            policy.eval()
            collector.settings.update(frozen_settings)
            started = time.perf_counter()
            started_unix = time.time()
            batch = collector.collect(**kwargs)
            collect_seconds = time.perf_counter() - started
            handoff_started = time.perf_counter()
            size = write_batch(batch, Path(root) / token, maximum)
            profile = dict(getattr(collector, 'last_profile', {}), pipeline_started_unix=started_unix)
            connection.send(('result', token, profile,
                             collect_seconds, time.perf_counter()-handoff_started, size))
            del batch
            t.release_process_heap()
    except (EOFError, SystemExit):
        pass
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if collector is not None:
            collector.close()
        connection.close()


class AsyncDaggerCollector:
    """Single pending request; no learner arrays or CUDA tensors enter the child."""
    def __init__(self, policy, normalizer, settings, stage, device):
        import torch
        from starscream.privileged_racing import checkpoint_payload
        if str(settings.get('collector_backend', 'process')) != 'process':
            raise ValueError('Async DAgger requires the process collector')
        if (policy.action_head_type != 'mlp'
                or getattr(policy, 'action_mixture_mode_head', None) is not None):
            raise ValueError('Async DAgger is qualified for the deterministic MLP actor')
        self.settings = copy.deepcopy(dict(settings))
        self.policy = policy
        self.parallel = int(settings.get('rollout_envs', 12))
        self.pending = None
        self.last_profile = {}
        self.window_collect_seconds = 0.0
        self.window_provenance = []
        self.closed = False
        self.sequence = 0
        self.actor = None
        self.actor_version = -1
        self.timeout = float(settings.get('dagger_pipeline_timeout_seconds', 900))
        self.maximum = int(settings.get('dagger_pipeline_max_handoff_bytes', 2 * 1024**3))
        if self.timeout <= 0 or self.maximum <= 0:
            raise ValueError('DAgger pipeline timeout and handoff budget must be positive')
        self.directory = tempfile.TemporaryDirectory(prefix='starscream-dagger-',
            dir=settings.get('dagger_pipeline_spool_directory'))
        root = Path(self.directory.name)
        self.result_directories = []
        initial = checkpoint_payload(policy, normalizer, stage='dagger', track='async-bootstrap')
        initial['model'] = {name: value.detach().cpu().clone() for name, value in initial['model'].items()}
        torch.save(initial, root / 'initial.pt')
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_collector_main,
            args=(child, str(root/'initial.pt'), self.settings, stage, device,
                  str(root), torch.get_float32_matmul_precision()), name='dagger-actor')
        self.process.start()
        child.close()
        try:
            if self._receive()[0] != 'ready':
                raise RuntimeError('DAgger actor failed startup handshake')
        except BaseException:
            self.close()
            raise

    def _receive(self):
        deadline = time.monotonic() + self.timeout
        while not self.connection.poll(.1):
            if not self.process.is_alive():
                raise RuntimeError(f'DAgger actor exited ({self.process.exitcode})')
            if time.monotonic() > deadline:
                raise TimeoutError('DAgger actor response timed out')
        response = self.connection.recv()
        if response[0] == 'error':
            raise RuntimeError('DAgger actor failed:\n' + response[1])
        return response

    def publish(self, policy, version):
        if self.pending is not None:
            raise RuntimeError('Cannot replace a pending actor snapshot')
        self.actor = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
        self.actor_version = int(version)

    def begin(self, kwargs, *, window, settings=None):
        import torch
        if self.pending is not None or self.actor is None:
            raise RuntimeError('DAgger begin requires a published actor and an empty queue')
        self.sequence += 1
        token = f'window-{self.sequence:06d}'
        frozen = copy.deepcopy(self.settings if settings is None else settings)
        snapshot = Path(self.directory.name) / 'actor.pt'
        torch.save(self.actor, snapshot)
        self.pending = dict(window=int(window), actor_version=self.actor_version,
                            settings=frozen, kwargs=copy.deepcopy(kwargs), token=token)
        self.connection.send(('collect', token, str(snapshot), frozen, kwargs, self.maximum))

    def checkpoint_state(self):
        if self.pending is None:
            return None
        state = {key: copy.deepcopy(value) for key, value in self.pending.items() if key != 'token'}
        state['actor'] = self.actor
        state['schema'] = 1
        return state

    def restore_pending(self, state):
        if state.get('schema') != 1:
            raise ValueError('Unsupported asynchronous DAgger checkpoint schema')
        if int(state['window']) - int(state['actor_version']) != 2:
            raise ValueError('Pending DAgger actor must lag its collection window by one update')
        self.actor = {key: value.detach().cpu().clone() for key, value in state['actor'].items()}
        self.actor_version = int(state['actor_version'])
        self.begin(state['kwargs'], window=state['window'], settings=state['settings'])

    def collect(self, **kwargs):
        from scripts.train_privileged_racing import DaggerBatch
        if self.pending is None:
            self.begin(kwargs, window=getattr(self, 'logical_window', -1))
        if self.pending['kwargs'] != kwargs:
            raise ValueError('Collection request differs from frozen lookahead; discard explicitly')
        response = self._receive()
        if response[:2] != ('result', self.pending['token']):
            raise RuntimeError('DAgger result generation mismatch')
        _, token, profile, seconds, transfer_seconds, size = response
        self.window_collect_seconds += seconds
        frozen = self.pending['settings']
        self.window_provenance.append(dict(
            window=self.pending['window'], actor_version=self.pending['actor_version'],
            request=copy.deepcopy(self.pending['kwargs']),
            family_weights=copy.deepcopy(frozen.get('track_sampling_family_weights', {})),
            gate_weights=copy.deepcopy(frozen.get('dagger_transition_start_gate_weights_by_family', {})),
            speed_weights=copy.deepcopy(frozen.get('dagger_teacher_speed_fraction_weights', {})),
            successful_only=bool(frozen.get('dagger_require_successful_episodes', False))))
        self.last_profile = dict(profile, pipeline_collect_seconds=seconds,
            pipeline_handoff_seconds=transfer_seconds, pipeline_handoff_bytes=size,
            pipeline_actor_version=self.pending['actor_version'], pipeline_window=self.pending['window'])
        path = Path(self.directory.name) / token
        result = read_batch(path, DaggerBatch)
        self.result_directories.append(path)
        self.pending = None
        return result

    def release_results(self):
        # Linux mappings keep data alive after unlink, even if a caller retains a view.
        for path in self.result_directories:
            shutil.rmtree(path)
        self.result_directories.clear()

    def discard_pending(self):
        if self.pending is not None:
            response = self._receive()
            if response[:2] != ('result', self.pending['token']):
                raise RuntimeError('DAgger discard generation mismatch')
            shutil.rmtree(Path(self.directory.name) / self.pending['token'])
            self.pending = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.is_alive():
            try:
                self.connection.send(('close',))
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
        self.connection.close()
        self.directory.cleanup()
