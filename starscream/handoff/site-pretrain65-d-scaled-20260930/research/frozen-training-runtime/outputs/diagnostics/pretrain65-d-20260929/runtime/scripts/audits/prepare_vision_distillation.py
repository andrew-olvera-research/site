#!/usr/bin/env python3
"""Print deterministic proposed files; does not launch or mutate experiments."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def select(records, size=25):
    """Population family quotas + rarity-weighted marginal cell coverage.

    Cell-kind normalization prevents abundant chain cells dominating approach,
    entry and vertical cells. No teacher/student outcomes enter selection.
    """
    counts = Counter(r['family'] for r in records)
    quota = {f: int(size*n/len(records)) for f,n in counts.items()}
    for f in sorted(counts, key=lambda f: (-(size*counts[f]/len(records)-quota[f]), f))[:size-sum(quota.values())]:
        quota[f] += 1
    frequency = Counter(c for r in records for c in set(r['cells']))
    kinds = Counter(c.split(':')[0] for c in frequency)
    weight = {c: 1/(frequency[c]**.5 * kinds[c.split(':')[0]]) for c in frequency}
    chosen, covered = [], set()
    used = Counter()
    while len(chosen) < size:
        eligible = [r for r in records if r not in chosen and used[r['family']] < quota[r['family']]]
        best = min(eligible, key=lambda r: (-sum(weight[c] for c in set(r['cells'])-covered), r['slot']))
        chosen.append(best); covered.update(best['cells']); used[best['family']] += 1
    return sorted(chosen, key=lambda r:r['slot']), dict(quota), dict(counts), covered


def proposed_files():
    source = ROOT/'configs/eval/v6_22_real100_hard_v2.manifest.json'
    records = json.loads(source.read_text())['records']
    chosen, quota, counts, covered = select(records)
    manifest = dict(schema='starscream-vision-selection-v1', source=str(source.relative_to(ROOT)),
        source_sha256=digest(source), method='family-proportional rarity-weighted cell coverage; deterministic slot tie break',
        family_quotas=quota, family_weights={f:n/len(records) for f,n in counts.items()},
        covered_cells=len(covered), total_cells=len({c for r in records for c in r['cells']}),
        records=[{k:r[k] for k in ('name','path','slot','family','fingerprint','cells')} for r in chosen],
        report_slots=[r['slot'] for r in records if r not in chosen])
    text = json.dumps(manifest, indent=2)+'\n'
    selection_path = 'configs/eval/v6211_vision_selection25.json'
    files = {selection_path: text}
    training = 'configs/exp/v6.21.1/update_fix_dagger.yaml'
    teachers = dict(base='starscream-v6.21.1.update-fix-pretrain65-dagger',
                    rl2='starscream-v6.21.1.rl.2-new-course-critic-100m')
    for lineage, directory in teachers.items():
        folder = ROOT/'outputs/checkpoints'/directory
        rank = json.loads((folder/'top-k.json').read_text())
        # Freeze one exact ranked checkpoint now; never silently follow latest.
        item = sorted(rank['checkpoints'], key=lambda r:(-r['score'],r['step']))[0]
        teacher = folder/item['path']
        for arm in ['raw','ekf']:
            config = dict(schema='starscream-vision-distillation-v1',
                teacher_checkpoint=str(teacher.relative_to(ROOT)), teacher_sha256=digest(teacher),
                training_source=training, training_source_sha256=digest(ROOT/training),
                selection_manifest=selection_path, selection_sha256=hashlib.sha256(text.encode()).hexdigest(),
                observation=arm, seed=2026092301, evaluation_seed=2034091462,
                model=dict(width=256,depth=3,heads=8,feedforward=504,history=3),
                output=f'outputs/vision-distillation/v6211-{lineage}-{arm}-3m-seed1',
                image_delay=.033, speed_command=16.5, vision_augmentation=True,
                rounds=24, beta_decay_rounds=12, rollout_envs=16,
                updates_per_round=1024, batch_size=256, replay_capacity=260000,
                learning_rate=.00012, dynamics_weight=.15,
                evaluation_episodes=4, evaluation_interval=2)
            for explicit in [False, True]:
                for conditioning in [True, False]:
                    variant = copy.deepcopy(config)
                    suffix = ('_interface' if explicit else '') + ('' if conditioning else '_unconditioned')
                    variant['model'].update(explicit_estimation=explicit, conditioning=conditioning,
                        feedforward=504 if conditioning else 618)
                    variant.update(state_weight=.5 if explicit else 0., interface_weight=.25 if explicit else 0.)
                    variant['output'] = f'outputs/vision-distillation/v6211-{lineage}-{arm}{suffix}-3m-seed1'
                    variant['wandb'] = dict(enabled=True, mode='online', project='starscream',
                        run_name=f'v6211-{lineage}-{arm}{suffix}-3m-seed1',
                        train_metric_allowlist=['started','round','loss','action','dynamics','state','interface',
                            'beta','replay','collection_seconds','update_seconds','round_seconds',
                            'updates_per_second','learning_rate'],
                        eval_metric_allowlist=['success_rate','teacher_success_rate','success_retention',
                            'successful_lap_seconds','mean_episode_steps','teacher/*'],
                        tags=['vision-distillation','v6.21.1',lineage,arm,
                              'explicit-estimation' if explicit else 'implicit-estimation',
                              'conditioned' if conditioning else 'unconditioned'],
                        local_event_path=variant['output']+'/wandb-events.jsonl')
                    files[f'configs/exp/v6.21.1.distill/{lineage}_{arm}{suffix}_3m.json'] = json.dumps(variant, indent=2)+'\n'
    return files


if __name__ == '__main__':
    print(json.dumps(proposed_files()))
