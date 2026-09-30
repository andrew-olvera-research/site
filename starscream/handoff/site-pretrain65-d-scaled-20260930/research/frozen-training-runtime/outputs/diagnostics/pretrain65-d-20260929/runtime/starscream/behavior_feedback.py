"""Project real100 behavior evidence onto existing training requirement cells.

Validation coordinates define requirements, never replacement training geometry.
Training shard/family identities are retained; competence and gate weights now
come from the real100 behavior taxonomy and matching geometric cells.
"""
from collections import Counter, defaultdict
import hashlib
import json

import numpy as np

from starscream.evaluation_suite import repository_path
from starscream.env.tracks import load_track
from starscream.env.racing_manifold.benchmark_v22 import requirement_cells, requirement_cells_by_gate


def build_behavior_map(settings):
    suite_path = repository_path(settings['evaluation_suite_manifest'])
    suite = json.loads(suite_path.read_text())
    population_path = repository_path(suite['source'])
    if hashlib.sha256(population_path.read_bytes()).hexdigest() != suite['source_sha256']:
        raise ValueError('real100 population manifest changed')
    population = json.loads(population_path.read_text())['records']
    frequency = Counter(c for r in population for c in set(r['cells']))
    kinds = Counter(c.split(':')[0] for c in frequency)
    cell_weights = {c:1/(frequency[c]**.5*kinds[c.split(':')[0]]) for c in frequency}
    family_count = Counter(r['family'] for r in population)
    support = {f:Counter() for f in family_count}
    for record in population:
        support[record['family']].update(set(record['cells']))
    manifest_path = repository_path(settings['track_manifest'])
    manifest = json.loads(manifest_path.read_text())
    records = {str(repository_path(r['path']).resolve()):r for r in manifest['records']}
    training = []
    for value in settings['curriculum']['tracks']:
        path = repository_path(value).resolve()
        record = records[str(path)]
        track = load_track(path)
        cells = requirement_cells(track)
        scores = {f:suite['family_weights'][f]*sum(cell_weights[c]*support[f][c]/family_count[f]
                  for c in cells if c in cell_weights) for f in family_count}
        total = sum(scores.values())
        if total <= 0:
            raise ValueError(f'No real100 requirement support for training course {path}')
        training.append(dict(path=str(path), name=track.name, family=record['family'],
            source=record['source_family'], behavior_mix={f:v/total for f,v in scores.items()},
            gate_cells=[sorted(c) for c in requirement_cells_by_gate(track,finite_route=True)]))
    validation = []
    for record in suite['records']:
        track = load_track(repository_path(record['path']))
        validation.append(dict(name=track.name,family=record['family'],
            gate_cells=[sorted(c) for c in requirement_cells_by_gate(track,finite_route=True)]))
    return dict(schema='real100-behavior-feedback-v1',
        suite_sha256=hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        population_sha256=suite['source_sha256'],
        training_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        method='population-prior, rarity/kind-normalized shared requirement support',
        cell_weights=cell_weights,training=training,validation=validation)


def behavior_feedback(evaluation, mapping, *, route_horizon=6, gate_floor=.2):
    competence = defaultdict(list)
    cell_reached, cell_failed = Counter(), Counter()
    for record in mapping['validation']:
        prefix = f"track/{record['name']}/"
        n = len(record['gate_cells'])
        survival = [float(evaluation[prefix+f'p{k}']) for k in range(1,n+1)]
        if (not all(np.isfinite(p) and 0 <= p <= 1 for p in survival)
                or any(b > a + 1e-6 for a,b in zip(survival,survival[1:]))):
            raise ValueError(f'Invalid survival curve for {record["name"]}')
        competence[record['family']].append(float(np.mean(survival[:route_horizon])))
        episodes = float(evaluation[prefix+'episodes'])
        if not np.isfinite(episodes) or episodes <= 0:
            raise ValueError('Behavior feedback requires evaluated episodes')
        # Gate zero is a cold spawn, not the cyclic arrival described by the
        # geometry cells. Unreached transitions provide no failure evidence.
        for gate in range(1,n):
            reached = episodes*survival[gate-1]
            if reached > 0:
                for cell in record['gate_cells'][gate]:
                    horizon = int(cell[5]) if cell.startswith('chain') else 1
                    # Ordered requirements are evaluated through the entire
                    # sequence, not inferred from its first gate alone.
                    end = gate+horizon-1
                    if end >= n-1:
                        continue  # final outgoing chord is a cyclic closure
                    failed = episodes*max(0.,survival[gate-1]-survival[end])
                    cell_reached[cell] += reached
                    cell_failed[cell] += failed
    behavior_competence = {f:float(np.mean(v)) for f,v in competence.items()}
    sources, gates = defaultdict(list), defaultdict(lambda:defaultdict(list))
    observed_training_gates = total_training_gates = 0
    for record in mapping['training']:
        value = sum(w*behavior_competence[f] for f,w in record['behavior_mix'].items())
        sources[record['source']].append(value)
        for gate,cells in enumerate(record['gate_cells']):
            if gate == 0:
                continue
            evidence = [(mapping['cell_weights'].get(c,0),
                         (cell_failed[c]+.5)/(cell_reached[c]+1))
                        for c in cells if cell_reached[c] > 0]
            mass = sum(w for w,_ in evidence)
            total_training_gates += 1
            observed_training_gates += int(mass > 0)
            # No observations means no special transition target. A uniform
            # positive prior preserves sampling instead of inventing failures.
            risk = sum(w*r for w,r in evidence)/mass if mass else .5
            gates[record['family']][gate].append(gate_floor+risk)
    return ({s:float(np.mean(v)) for s,v in sources.items()},
            {f:{g:float(np.mean(v)) for g,v in rows.items()} for f,rows in gates.items()},
            behavior_competence,
            dict(observed_cells=len(cell_reached),training_courses=len(mapping['training']),
                 behavior_families=len(behavior_competence),
                 observed_training_gate_fraction=observed_training_gates/max(total_training_gates,1)))


def sampler_feedback(evaluation, settings, manifest_path, route_horizon):
    config = settings.get('dagger_dynamic_sampling',{})
    if not config.get('behavior_map'):
        from scripts.train_privileged_racing import dagger_family_survival_statistics
        return dagger_family_survival_statistics(evaluation,manifest_path,route_horizon=route_horizon)
    path = repository_path(config['behavior_map'])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['behavior_map_sha256']:
        raise ValueError('Behavior sampler map changed')
    mapping = json.loads(raw)
    if mapping['suite_sha256'] != settings['evaluation_suite_sha256']:
        raise ValueError('Behavior sampler map belongs to a different selection suite')
    if mapping['training_manifest_sha256'] != hashlib.sha256(repository_path(manifest_path).read_bytes()).hexdigest():
        raise ValueError('Behavior sampler training manifest changed')
    competence,gates,behaviors,health = behavior_feedback(evaluation,mapping,
        route_horizon=route_horizon,gate_floor=float(config.get('gate_weight_floor',.2)))
    config['behavior_gate_weights'] = gates
    config['behavior_competence'] = behaviors
    config['behavior_health'] = health
    config['behavior_training_mix'] = {r['family']:r['behavior_mix'] for r in mapping['training']}
    config['behavior_family_weights'] = settings['evaluation_suite_family_weights']
    return competence, {}


def update_behavior_sampler(update, *, base_family_weights, source_by_family,
                            observed_losses, previous_state, config):
    """Learn priorities in the 15 real100 families, then project onto fixed shards."""
    mixes = config['behavior_training_mix']
    if set(mixes) != set(base_family_weights):
        raise ValueError('Behavior map must cover every training shard exactly')
    families = config['behavior_family_weights']
    losses = {}
    for behavior in families:
        evidence = [(mix[behavior]*base_family_weights[f], observed_losses[source_by_family[f]])
                    for f,mix in mixes.items() if source_by_family[f] in observed_losses]
        mass = sum(w for w,_ in evidence)
        if mass:
            losses[behavior] = sum(w*v for w,v in evidence)/mass
    legacy_config = dict(config)
    legacy_config.pop('behavior_map')
    state, _, _ = update(base_family_weights=families,
        source_by_family={f:f for f in families},
        observed_competence=config['behavior_competence'], observed_frontiers={},
        observed_losses=losses, previous_state=previous_state, config=legacy_config)
    weights = {f:base_family_weights[f]*sum(w*state['multipliers'][b] for b,w in mix.items())
               for f,mix in mixes.items()}
    previous_gates = (previous_state or {}).get('gate_weights',{})
    alpha = float(config.get('competence_ema',.3))
    floor = float(config.get('gate_weight_floor',.2))
    gates = {}
    for family, targets in config['behavior_gate_weights'].items():
        old = previous_gates.get(family,{})
        gates[family] = {}
        for gate,target in targets.items():
            if not np.isfinite(target) or not floor <= target <= floor+1:
                raise ValueError('Invalid behavior gate target')
            prior = float(old.get(gate,old.get(str(gate),floor+.5)))
            gates[family][int(gate)] = (1-alpha)*prior+alpha*target
    state.update(frontiers={},family_weights=weights,gate_weights=gates,
        behavior_health=config['behavior_health'], feedback_schema='real100-behavior-feedback-v1',
        behavior_map_sha256=config['behavior_map_sha256'])
    return state, weights, gates
