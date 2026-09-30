"""Reproducible v6.22 benchmark proposals, MPCC search and release gate.

Run in the starscream container. Subcommands never modify training or real50.
Candidates and failed teacher attempts remain in an append-only audit trail.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from starscream.env.procedural_tracks import geometry_fingerprint, save_track_yaml
from starscream.env.tracks import load_track
from starscream.env.racing_manifold.benchmark_v22 import (
    CloneIndex, FAMILIES, VERSION, generate, requirement_cells, validate_geometry,
)
from starscream.env.racing_manifold.corpus_coverage import clone_distance, geometry_record

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/course-pools/v622-benchmarks'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    p = Path(path)
    if str(p).startswith('/workspace/'):
        return ROOT/p.relative_to('/workspace')
    return p if p.is_absolute() else ROOT/p


def protected_tracks():
    paths = {}
    for manifest in ('outputs/course-pools/v621-additions/manifest.json',
                     'outputs/diagnostics/v6201-teacher-speed/frozen/manifest.json'):
        for row in json.loads((ROOT/manifest).read_text())['records']:
            paths[str(resolve(row['path']))] = None
    suite = yaml.safe_load((ROOT/'configs/eval/v6_21_holdout_eval_50.yaml').read_text())
    for row in suite['active']:
        paths[str(resolve(row['track']))] = None
    # Geometry already used by any saved online bank is also protected.
    for path in (ROOT/'outputs/checkpoints').glob('*/bank/**/*.yaml'):
        paths[str(path)] = None
    tracks = {}
    for path in paths:
        t = load_track(path)
        tracks[geometry_fingerprint(t)] = t
    return list(tracks.values())


def cell_weight(cell):
    if cell.startswith('chain4:'): return .125
    if cell.startswith('chain3:'): return .25
    if cell.startswith('chain2:'): return .5
    return 1.


def propose(args):
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = OUT/'candidates.json'
    if ledger.exists():
        raise ValueError('candidate ledger exists; use qualify/report or a new version')
    protected = protected_tracks()
    clones = CloneIndex(protected)
    selected_tracks = []
    rows, rejected = [], []
    suite = yaml.safe_load((ROOT/'configs/eval/v6_21_holdout_eval_50.yaml').read_text())
    for entry in suite['active'][:8]:
        t = load_track(resolve(entry['track']))
        path = save_track_yaml(t, OUT/'references'/f'{t.name}.yaml')
        rows.append(dict(name=t.name, path=str(path), suite='public_reference', slot=t.name,
                         family='public_reference', source='real50_public_reference',
                         fingerprint=geometry_fingerprint(t), prior_exposure='real50 validation and distribution design',
                         cells=sorted(requirement_cells(t)), geometry=geometry_record(t),
                         static_reasons=validate_geometry(t)))
    for split, total in [('real60', 52), ('real100-hard', 92)]:
        support = Counter()
        for slot in range(total):
            family = FAMILIES[slot % len(FAMILIES)]
            hard = split == 'real100-hard'
            options = []
            for attempt in range(args.proposals):
                seed = 622000000 + int(hard)*10000000 + slot*10000 + attempt
                count = (int(np.random.default_rng(seed).integers(8, 19 if hard else 13))
                         if family in ('long_low', 'long_braking', 'ordered_3d', 'flow')
                         else int(np.random.default_rng(seed).integers(5, 11)))
                name = f'v622_{"hard" if hard else "champ"}_{family}_{slot:03d}_{attempt:03d}'
                try:
                    t = generate(family, count, seed, name, hard)
                    reasons = validate_geometry(t)
                    if reasons:
                        rejected.append(dict(name=name, reasons=reasons)); continue
                    nearest = clones.distance(t)
                    if nearest < .12:
                        rejected.append(dict(name=name, reasons=['clone'], distance=nearest)); continue
                    cells = requirement_cells(t)
                    # Repeated witnesses, diminishing returns; no train deficits
                    # or policy failures influence this independent benchmark.
                    score = sum(cell_weight(c)/(1+support[c]) for c in cells)/len(t.gates)**.5
                    options.append((score, t, cells, nearest, seed))
                except ValueError as exc:
                    rejected.append(dict(name=name, reasons=[str(exc)]))
            if not options:
                raise RuntimeError(f'no proposals for {split}/{slot}/{family}')
            options.sort(key=lambda x: (-x[0], x[1].name))
            for rank, (score, t, cells, nearest, seed) in enumerate(options[:args.alternates]):
                path = save_track_yaml(t, OUT/'candidates'/f'{t.name}.yaml')
                rows.append(dict(name=t.name, path=str(path), suite=split, slot=f'{split}-{slot:03d}',
                                 family=family, rank=rank, seed=seed, source=VERSION,
                                 fingerprint=geometry_fingerprint(t), prior_exposure='fresh geometry',
                                 cells=sorted(cells), geometry=geometry_record(t),
                                 minimum_protected_clone_distance=nearest, selection_score=score, static_reasons=[]))
                # Reserve every alternate across splits and slots, not only rank 0.
                selected_tracks.append(t)
                clones.add(t)
            support.update(options[0][2])
            print('proposed', split, slot, family, len(options), flush=True)
    atomic_json(ledger, dict(version=VERSION, records=rows, rejections=rejected,
                            generator_sha256=digest(ROOT/'starscream/env/racing_manifold/benchmark_v22.py'),
                            protected_count=len(protected), independent_generated_splits=True))


def teacher_contract():
    cfg = yaml.safe_load((ROOT/'configs/exp/v6.20.1/qualification_r6.yaml').read_text())
    cfg.update(output=str(OUT), max_steps=6000)
    sources = ['scripts/audits/build_v622_benchmarks.py', 'scripts/audits/audit_dagger_teacher_labels.py',
               'scripts/audits/prepare_v619_pool.py', 'scripts/train_privileged_racing.py',
               'starscream/mpcc/controller.py', 'starscream/mpcc/acados_backend.py',
               'starscream/mpcc/config.py', 'starscream/mpcc/model.py', 'starscream/mpcc/racing_line.py']
    from scripts.train_privileged_racing import load_config
    payload = dict(config=cfg, sources={s: digest(ROOT/s) for s in sources},
                   teacher=load_config(resolve(cfg['teacher_config']))['dagger'])
    contract = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return cfg, contract, payload


def usable(e):
    return (e['success'] and not e['prefix_failed'] and e['solver_failure_fraction'] <= .06
            and e['recovery_fraction'] <= .10)


def qualify_job(job):
    row, cfg, contract, full = job
    import torch
    torch.set_num_threads(1)
    from scripts.audits.prepare_v619_pool import settings_for
    from scripts.audits.audit_dagger_teacher_labels import audit_track
    from scripts.audits.prepare_v620_transitions import finite_report
    from starscream.env.racing_manifold.transition_corpus import executed_transition_metrics
    root = OUT/'qualification'/row['name']/contract[:12]
    root.mkdir(parents=True, exist_ok=True)
    result_path = root/('full.json' if full else 'screen.json')
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['fingerprint'] != geometry_fingerprint(load_track(row['path'])):
            raise ValueError('qualification geometry drift')
        return result
    with (root/'solver.log').open('a') as log:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
    settings, stage = settings_for(cfg, row['path'])
    settings.update(deepcopy(cfg['teacher_overrides']))
    settings['mpcc_build_root'] = f'/tmp/starscream-v622-{os.getpid()}'
    cache = []
    # Search command AND acceleration/braking envelope, not cap alone.
    profiles = []
    for tag, command, authority in [('base', 16.5, False), ('fast', 24., True),
                                     ('controlled', 12., False), ('slow', 8., False)]:
        s = deepcopy(settings); s['mpcc_nominal_speed'] = command
        if authority:
            s['mpcc_config'].update(max_progress_speed=30., maximum_acceleration=30.,
                maximum_longitudinal_acceleration=24., maximum_braking_acceleration=26.,
                collective_slew_limit=16., body_rate_slew_limit=6., corridor_margin=.15)
        profiles.append((tag, command, s))

    def evaluate(tag, speed, settings, domain, mode, repeats, seed):
        dest = root/f'{tag}-{domain}'
        report_path = dest/'report.json'
        if report_path.exists(): return json.loads(report_path.read_text())
        s = deepcopy(settings)
        if domain.startswith('nominal'):
            s['dynamics_randomization'] = {'enabled': False}
        report = audit_track(s, stage, row['path'], 0, 1, repeats, seed,
            start_mode=mode, repeats_per_start=repeats, start_perturbation_scale=1.,
            dart_action_noise_scale=1. if domain == 'dart' else 0.,
            dart_episode_fraction=1. if domain == 'dart' else 0., speed_fractions=(1.,),
            frontier_speed=speed, trace_directory=dest, backend_cache=cache,
            qualification_limits={'solver': .06, 'recovery': .10})
        transitions = []
        for e, path in zip(report['episodes'], sorted(dest.glob('episode-*.npz'))):
            with np.load(path) as trace:
                if len(trace['states']) > 1:
                    ts = executed_transition_metrics(trace)
                    for t in ts:
                        t['physical_gate'] = (e['start_gate_index']+t['relative_gate_phase']) % row['geometry']['gate_count']
                    transitions.append(dict(start_gate=e['start_gate_index'], success=e['success'], transitions=ts))
        result = finite_report(dict(report=report, executed=transitions))
        atomic_json(report_path, result)
        return result

    screens = {}
    for tag, speed, s in profiles:
        screens[tag] = evaluate(tag, speed, s, 'nominal-screen', 'canonical', 2, 62217001)
    ranked = sorted((p for p in profiles if all(usable(e) for e in screens[p[0]]['report']['episodes'])),
                    key=lambda p: np.mean([e['lap_time_seconds'] for e in screens[p[0]]['report']['episodes']]))
    result = dict(name=row['name'], fingerprint=row['fingerprint'], contract=contract,
                  screen_passed=bool(ranked), qualified=False, selected_profile=None,
                  screens={k:v['report']['summary'] for k,v in screens.items()}, trials={})
    if ranked:
        result['selected_profile'] = ranked[0][0]
    if full:
        for tag, speed, s in ranked:
            cohorts = {}
            for domain, mode, repeats, seed in [
                ('nominal-all-starts', 'all', 1, 62218001),
                ('randomized', 'all', 1, 62219001),
                ('dart', 'all', 1, 62220001),
                ('prefix', 'expert-prefix-all', 1, 62221001)]:
                cohorts[domain] = evaluate(tag, speed, s, domain, mode, repeats, seed)
                if not all(usable(e) for e in cohorts[domain]['report']['episodes']): break
            result['trials'][tag] = {k:v['report']['summary'] for k,v in cohorts.items()}
            if len(cohorts) == 4 and all(all(usable(e) for e in c['report']['episodes']) for c in cohorts.values()):
                result.update(qualified=True, selected_profile=tag, speed_command=speed,
                              teacher_overrides={k:s[k] for k in ('mpcc_nominal_speed', 'mpcc_config')})
                break
    atomic_json(result_path, result)
    return result


def qualify(args):
    data = json.loads((OUT/'candidates.json').read_text())
    cfg, contract, payload = teacher_contract()
    (OUT/'contracts').mkdir(parents=True, exist_ok=True)
    atomic_json(OUT/'contracts'/f'{contract}.json', payload)
    ledger = OUT/('qualification.json' if args.full else 'screen.json')
    results = {}
    if ledger.exists():
        prior = json.loads(ledger.read_text())
        if prior['contract'] == contract:
            results = {r['name']:r for r in prior['records']}
    rows = [r for r in data['records'] if r.get('rank', 0) <= args.rank]
    if args.limit: rows = rows[:args.limit]
    pending = [r for r in rows if r['name'] not in results]
    print('qualify', len(pending), 'full', args.full, 'contract', contract[:12], flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
        futures = {pool.submit(qualify_job, (r, cfg, contract, args.full)):r for r in pending}
        for future in as_completed(futures):
            row = futures[future]
            try: result = future.result()
            except Exception as exc:
                result = dict(name=row['name'], contract=contract, qualified=False, screen_passed=False, error=repr(exc))
            results[row['name']] = result
            atomic_json(ledger, dict(contract=contract, records=list(results.values())))
            print('result', row['name'], result.get('screen_passed'), result['qualified'], result.get('error', ''), flush=True)


def report(args):
    data = json.loads((OUT/'candidates.json').read_text())
    _, contract, _ = teacher_contract()
    evidence = {}
    for row in data['records']:
        root = OUT/'qualification'/row['name']/contract[:12]
        path = root/'full.json' if (root/'full.json').exists() else root/'screen.json'
        if path.exists(): evidence[row['name']] = json.loads(path.read_text())
    selected, missing = [], []
    for slot in sorted({r['slot'] for r in data['records']}):
        options = [r for r in data['records'] if r['slot'] == slot]
        passed = [r for r in options if evidence.get(r['name'], {}).get('qualified')]
        if passed: selected.append(min(passed, key=lambda r:r.get('rank', 0)))
        else: missing.append(slot)
    summary = dict(contract=contract, candidate_count=len(data['records']), slots=len({r['slot'] for r in data['records']}),
                   screened=len(evidence), screen_passed=sum(r.get('screen_passed', False) for r in evidence.values()),
                   qualified=len(selected), missing_slots=missing, release_ready=not missing)
    atomic_json(OUT/'status.json', summary)
    print(json.dumps(summary, indent=2))
    if args.freeze:
        if missing: raise RuntimeError('release blocked: unqualified slots')
        # A final cross-split and within-suite check includes chosen alternates.
        generated = [r for r in selected if r['suite'] != 'public_reference']
        for i, row in enumerate(generated):
            a = load_track(row['path'])
            for other in generated[:i]:
                b = load_track(other['path'])
                distance = clone_distance(a, b)
                if distance is not None and distance < .12:
                    raise ValueError(f'clone at release: {row["name"]} / {other["name"]}')
        for split, count in [('real60', 60), ('real100-hard', 100)]:
            records = [r for r in selected if r['suite'] in (split, 'public_reference')]
            assert len(records) == count
            active = []
            for row in records:
                path = ROOT/'starscream/assets/tracks/v622'/split/f'{row["name"]}.yaml'
                save_track_yaml(load_track(row['path']), path)
                active.append(dict(name=row['name'], track=str(path.relative_to(ROOT)),
                                   geometry_fingerprint=row['fingerprint'], prior_exposure=row['prior_exposure']))
            suite = dict(schema='starscream-real-course-suite-v1', description=f'v6.22 {split}; public anchors plus independent generated geometry',
                         active=active, pending_geometry=[])
            target = ROOT/'configs/eval'/f'v6_22_{split.replace("-", "_")}.yaml'
            target.write_text(yaml.safe_dump(suite, sort_keys=False))
            atomic_json(OUT/f'{split}-manifest.json', dict(records=records, contract=contract,
                course_witnesses=dict(Counter(c for r in records for c in r['cells']))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['propose', 'qualify', 'report'])
    parser.add_argument('--proposals', type=int, default=40)
    parser.add_argument('--alternates', type=int, default=3)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    {'propose':propose, 'qualify':qualify, 'report':report}[args.command](args)


if __name__ == '__main__':
    main()
