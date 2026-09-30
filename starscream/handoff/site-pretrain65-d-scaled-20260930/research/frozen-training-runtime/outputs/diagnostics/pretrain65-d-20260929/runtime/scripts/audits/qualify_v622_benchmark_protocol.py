"""Benchmark admission under the established four-cohort r6 protocol.

Cold-start-all-gates stress remains separate diagnostic evidence. Canonical
nominal/randomized/DART and randomized expert-prefix-all are admission gates.
No thresholds, dynamics, or course geometry are relaxed after a failure.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.audits.build_v622_benchmarks import OUT, ROOT, teacher_contract, usable
from starscream.course_model.training import atomic_json
from starscream.env.procedural_tracks import geometry_fingerprint
from starscream.env.tracks import load_track


def protocol_contract():
    cfg, base, payload = teacher_contract()
    contract = hashlib.sha256((base+Path(__file__).read_text()).encode()).hexdigest()
    return cfg, contract, dict(base=base, base_contract=payload,
        protocol='canonical-nominal2-randomized2-dart2-expert-prefix-all1',
        qualification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def profiles(settings, reference):
    output = []
    for tag, speed, fast in [('base',16.5,False), ('fast',24.,True), ('controlled',12.,False), ('slow',8.,False)]:
        s = deepcopy(settings)
        if fast:
            s['mpcc_config'].update(max_progress_speed=30., maximum_acceleration=30.,
                maximum_longitudinal_acceleration=24., maximum_braking_acceleration=26.,
                collective_slew_limit=16., body_rate_slew_limit=6., corridor_margin=.15)
        s['mpcc_nominal_speed'] = speed
        output.append((tag, speed, s))
    if reference:
        for acceleration, longitudinal, braking in [(10.,6.,10.), (16.,10.,16.), (24.,16.,24.)]:
            for aperture in (0., .35):
                for speed in (10.,20.):
                    s = deepcopy(settings)
                    s['mpcc_config'].update(maximum_acceleration=acceleration,
                        maximum_longitudinal_acceleration=longitudinal,
                        maximum_braking_acceleration=braking, corridor_margin=.15)
                    s['mpcc_planner_config'].update(aperture_fraction=aperture, aperture_margin=.12, offset_iterations=40)
                    s['mpcc_nominal_speed'] = speed
                    output.append((f'a{acceleration:g}-p{aperture:g}-v{speed:g}', speed, s))
    return output


def job(args):
    row, cfg, contract, base_contract = args
    import torch
    torch.set_num_threads(1)
    from scripts.audits.prepare_v619_pool import settings_for
    from scripts.audits.audit_dagger_teacher_labels import audit_track
    from scripts.audits.prepare_v620_transitions import finite_report
    root = OUT/'benchmark-qualification'/row['name']/contract[:12]
    root.mkdir(parents=True, exist_ok=True)
    dest = root/'result.json'
    if dest.exists():
        result = json.loads(dest.read_text())
        if result['fingerprint'] != geometry_fingerprint(load_track(row['path'])): raise ValueError('geometry drift')
        return result
    stronger = OUT/'qualification'/row['name']/base_contract[:12]/'full.json'
    if stronger.exists():
        prior = json.loads(stronger.read_text())
        if prior['qualified'] and prior['fingerprint'] == row['fingerprint']:
            result = dict(prior, contract=contract, evidence_contract=base_contract,
                acceptance_basis='stronger all-physical-starts nominal/randomized/DART plus expert-prefix-all',
                upstream_evidence=str(stronger))
            atomic_json(dest, result)
            return result
    search_contract = hashlib.sha256((base_contract+(ROOT/'scripts/audits/tune_v622_reference_teachers.py').read_text()).encode()).hexdigest()
    for path in (OUT/'reference-search'/row['name']/search_contract[:12]).glob('*/full.json'):
        prior = json.loads(path.read_text())
        if prior['qualified'] and prior['fingerprint'] == row['fingerprint']:
            result = dict(prior, contract=contract, evidence_contract=search_contract,
                acceptance_basis='stronger all-physical-starts extended teacher search', upstream_evidence=str(path))
            atomic_json(dest, result)
            return result
    with (root/'solver.log').open('a') as log:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
    settings, stage = settings_for(cfg, row['path'])
    settings.update(deepcopy(cfg['teacher_overrides']))
    settings['mpcc_build_root'] = f'/tmp/starscream-v622-benchmark-{os.getpid()}'
    choices = profiles(settings, row['suite'] == 'public_reference')
    cache, reports = [], {}

    def evaluate(tag, speed, settings, domain):
        path = root/f'{tag}-{domain}'/'report.json'
        if path.exists(): return json.loads(path.read_text())
        # Reuse only exactly matching nominal source evidence. Different
        # arena fingerprints/names never inherit the old qualification.
        if domain == 'nominal':
            old = OUT/'qualification'/row['name']/base_contract[:12]/f'{tag}-nominal-screen'/'report.json'
            provenance = old.parent.parent/'screen.json'
            if old.exists() and provenance.exists() and json.loads(provenance.read_text())['fingerprint'] == row['fingerprint']:
                report = json.loads(old.read_text())['report']
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(path, dict(report=report, reused_trace_directory=str(old.parent),
                                      source_contract=base_contract, fingerprint=row['fingerprint']))
                return json.loads(path.read_text())
            search_contract = hashlib.sha256((base_contract+(ROOT/'scripts/audits/tune_v622_reference_teachers.py').read_text()).encode()).hexdigest()
            old = OUT/'reference-search'/row['name']/search_contract[:12]/tag/'screen.json'
            if old.exists():
                prior = json.loads(old.read_text())
                if prior['fingerprint'] == row['fingerprint']:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(path, dict(report=prior['cohorts']['nominal-screen'],
                        reused_trace_directory=str(old.parent/'nominal-screen'),
                        source_contract=search_contract, fingerprint=row['fingerprint']))
                    return json.loads(path.read_text())
        s = deepcopy(settings)
        if domain == 'nominal': s['dynamics_randomization'] = {'enabled':False}
        seeds = {'nominal':62217001,'randomized':62219001,'dart':62220001,'prefix':62221001}
        result = audit_track(s, stage, row['path'], 0, 1, 2, seeds[domain],
            start_mode='expert-prefix-all' if domain == 'prefix' else 'canonical',
            repeats_per_start=1 if domain == 'prefix' else 2, start_perturbation_scale=1.,
            dart_action_noise_scale=float(domain == 'dart'), dart_episode_fraction=float(domain == 'dart'),
            speed_fractions=(1.,), frontier_speed=speed, trace_directory=path.parent,
            backend_cache=cache, qualification_limits={'solver':.06, 'recovery':.10})
        result = finite_report(dict(report=result, fingerprint=row['fingerprint']))
        atomic_json(path, result)
        return result

    screened = []
    for tag, speed, s in choices:
        evidence = evaluate(tag, speed, s, 'nominal')
        episodes = evidence['report']['episodes']
        if episodes and all(usable(e) for e in episodes):
            screened.append((float(np.mean([e['lap_time_seconds'] for e in episodes])), tag, speed, s))
    result = dict(name=row['name'], fingerprint=row['fingerprint'], contract=contract,
                  qualified=False, attempted_profiles={}, nominal_profiles_passed=len(screened))
    for _, tag, speed, s in sorted(screened, key=lambda x:x[0]):
        cohorts = {}
        for domain in ('nominal', 'randomized', 'dart', 'prefix'):
            evidence = evaluate(tag, speed, s, domain)
            cohorts[domain] = evidence['report']['summary']
            episodes = evidence['report']['episodes']
            if not episodes or not all(usable(e) for e in episodes): break
        result['attempted_profiles'][tag] = cohorts
        if len(cohorts) == 4 and all(cohorts[d]['success_rate'] == 1. for d in cohorts):
            # Re-check per-episode limits: aggregate success alone is insufficient.
            if all(all(usable(e) for e in evaluate(tag,speed,s,d)['report']['episodes']) for d in cohorts):
                result.update(qualified=True, selected_profile=tag, speed_command=speed,
                    teacher_overrides={k:s[k] for k in ('mpcc_nominal_speed','mpcc_config','mpcc_planner_config')})
                break
    atomic_json(dest, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    data = json.loads((OUT/'candidates.json').read_text())
    plan = json.loads((OUT/'independence-selection.json').read_text())
    cfg, contract, payload = protocol_contract()
    (OUT/'contracts').mkdir(exist_ok=True)
    atomic_json(OUT/'contracts'/f'{contract}.json', payload)
    preferences = {c['name']:i for choices in plan['preferences'].values() for i,c in enumerate(choices)}
    rows = [r for r in data['records'] if r['suite'] == 'public_reference' or r['name'] in preferences]
    if args.limit: rows = rows[:args.limit]
    selected, attempted = {}, {}
    for row in rows:
        path = OUT/'benchmark-qualification'/row['name']/contract[:12]/'result.json'
        if path.exists():
            result = json.loads(path.read_text()); attempted[row['name']] = result
            if result['qualified']: selected[row['slot']] = row['name']
    for rank in range(max(preferences.values(), default=0)+1):
        pending = [r for r in rows if preferences.get(r['name'],0) == rank and r['slot'] not in selected and r['name'] not in attempted]
        print('BENCHMARK', contract[:12], 'rank',rank, 'pending',len(pending),flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
            jobs = {pool.submit(job, (r,cfg,contract,payload['base'])):r for r in pending}
            for future in as_completed(jobs):
                row = jobs[future]; result = future.result(); attempted[row['name']] = result
                if result['qualified']: selected[row['slot']] = row['name']
                atomic_json(OUT/'benchmark-progress.json', dict(contract=contract, selected=selected,
                    attempted=len(attempted), qualified=len(selected),
                    missing_slots=sorted({r['slot'] for r in rows}-set(selected))))
                print('ADMISSION', row['name'], result['qualified'], 'selected',len(selected),flush=True)


if __name__ == '__main__':
    main()
