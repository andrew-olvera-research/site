"""Audit the plant-aware live pipeline smoke before starting the long experiment."""
import json
from pathlib import Path
import numpy as np
import h5py
import torch

root = Path('outputs/checkpoints/starscream-v62111-plant-pipeline-smoke-v2')
states = {}
for path in root.glob('best-step-*.pt'):
    value = torch.load(path, map_location='cpu', weights_only=False)
    states[int(value['round'])] = value
final = torch.load(root/'latest.pt', map_location='cpu', weights_only=False)
pending = states[2]['dagger_async_pending']
checks = dict(final_round=final['round'] == 3,
    final_pending_empty=final['dagger_async_pending'] is None,
    pending_actor_version=pending['actor_version'] == 1 and pending['window'] == 3,
    pending_actor_exact=all(torch.equal(v, states[1]['model'][k]) for k,v in pending['actor'].items()),
    eval_matches_model=final['metrics']['dagger_policy_version'] == final['round'],
    model_schema=final['model_config']['input_dim'] == 167 and len(final['plant_settings_schema']['names']) == 64,
    settings_learned=not torch.equal(final['model']['step_embedding.settings_projection.input.weight'],
                                    states[0]['model']['step_embedding.settings_projection.input.weight']))
sizes = []
for path in sorted((root/'dagger-replay').glob('round-*.h5')):
    with h5py.File(path) as file:
        for pool in ('online','permanent'):
            history = file[pool]['histories']
            if len(history):
                indices = np.unique(np.linspace(0,len(history)-1,256,dtype=int))
                sample = history[indices]
                assert sample.shape[1:] == (3,167) and sample.dtype == np.float16
                assert np.isfinite(sample).all()
                # No reset can leak another episode's constants into a history.
                np.testing.assert_array_equal(sample[:,0,103:152],sample[:,-1,103:152])
                assert np.unique(sample[:,-1,103]).size > 1
            sizes.append(dict(shard=path.name,pool=pool,rows=len(history)))
if not all(checks.values()):
    raise AssertionError(checks)
events = [json.loads(line)['metrics'] for line in Path(
    'outputs/logs/starscream-v62111-plant-pipeline-smoke-v2.full.events.jsonl').read_text().splitlines()]
remote = [json.loads(line)['metrics'] for line in Path(
    'outputs/logs/starscream-v62111-plant-pipeline-smoke-v2.events.jsonl').read_text().splitlines()]
checks['local_course_metrics'] = any(any(k.startswith('eval/track/') for k in e) for e in events)
checks['remote_no_course_or_family_metrics'] = not any(any(k.startswith(('eval/track/','eval/family/')) for k in e) for e in remote)
assert all(checks.values())
report = dict(checks=checks,shards=sizes,environment_steps=final['environment_steps'],
              optimizer_updates=96,remote_metric_count=len(set().union(*(set(e) for e in remote))))
destination = Path('outputs/dagger-throughput/plant-smoke-audit.json')
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
