"""Small, budgeted PPO curriculum over a qualified, append-only course bank.

No goal-track geometry or validation outcomes enter admission. The generator
publishes feasibility; the trainer owns policy reachability and active tasks.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path


def phase_at(config, steps):
    phases = config['phases']
    ends = [int(p['until_steps']) for p in phases]
    if not ends or ends != sorted(set(ends)) or ends[0] <= 0:
        raise ValueError('curriculum phase boundaries must increase')
    return next((i for i, end in enumerate(ends) if steps < end), len(ends)-1)


def balanced_weights(core, additions, new_mass):
    if not 0 <= new_mass < 1:
        raise ValueError('source rehearsal must retain positive mass')
    def group_weights(rows, mass):
        groups = Counter(str(r.get('parent_id', r['name'])) for r in rows)
        return {r['name']: mass / len(groups) / groups[str(r.get('parent_id',r['name']))]
                for r in rows} if rows else {}
    return {**group_weights(core, 1-new_mass if additions else 1.),
            **group_weights(additions, new_mass)}


def reachable(metrics, gates):
    # Small admission probe, not a statistically precise success estimate.
    return (float(metrics.get('full_course_success', 0)) >= .25 or
            float(metrics.get('mean_gates', 0)) / gates >= .5)


class PoolCurriculum:
    def __init__(self, config, raw_stage, state=None):
        self.config = deepcopy(config)
        self.base = deepcopy(raw_stage)
        self.core = json.loads(Path(config['core_manifest']).read_text())['records']
        self.core = [r for r in self.core if r['split']=='train']
        self.state = deepcopy(state or {'accepted': [], 'last_probe': -1, 'phase': -1})
        self.forbidden = set(config['excluded_parent_ids'])
        self._validate(self.core)

    def _validate(self, rows):
        names = set(); fingerprints = set()
        for r in rows:
            if (r['split'] != 'train' or not r.get('qualified') or
                r.get('parent_id') in self.forbidden or not Path(r['path']).is_file()):
                raise ValueError('unsafe course bank record: '+r['name'])
            fp = r.get('geometry_fingerprint',r.get('fingerprint'))
            if not fp or fp in fingerprints or r['name'] in names:
                raise ValueError('duplicate or missing course identity')
            names.add(r['name']); fingerprints.add(fp)

    def update(self, steps, probe):
        phase = phase_at(self.config, steps)
        p = self.config['phases'][phase]
        changed = phase != self.state['phase']
        path = Path(self.config['bank_manifest'])
        if (phase > 0 and path.exists() and
            steps-self.state['last_probe'] >= self.config['refresh_steps']):
            rows = json.loads(path.read_text())['records']
            self._validate(self.core+rows)
            accepted = {r['name'] for r in self.state['accepted']}
            pending = [r for r in rows if r['name'] not in accepted]
            # Retry shelved tasks on later refreshes, but rotate which are probed.
            if pending:
                offset = (steps//self.config['refresh_steps']) % len(pending)
                pending = pending[offset:]+pending[:offset]
            added = 0
            for r in pending[:self.config['probes_per_refresh']]:
                if probe(r):
                    self.state['accepted'].append(r); added += 1; changed=True
                if added >= self.config['additions_per_refresh']: break
            self.state['last_probe'] = steps
        self.state['phase'] = phase
        rows = self.core+self.state['accepted']
        self._validate(rows)
        stage = deepcopy(self.base)
        stage.update(p['stage'])
        stage['tracks'] = [r['path'] for r in rows]
        weights = balanced_weights(self.core,self.state['accepted'],p['new_course_mass'])
        return changed, stage, weights, p
