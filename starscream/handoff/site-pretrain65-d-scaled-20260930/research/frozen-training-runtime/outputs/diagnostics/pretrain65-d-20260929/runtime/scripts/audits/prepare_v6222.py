"""Deterministic pretrain45 design, r6 qualification, and v6.21 recipe reproduction.

Run inside /workspace: python scripts/audits/prepare_v6222.py build|qualify|freeze.
Failures remain in an immutable candidate ledger; qualification never relaxes
the DART, solver, recovery, speed, or 3000-step admission contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starscream.course_model.training import atomic_json
from starscream.course_model.schema import static_reasons
from starscream.env.tracks import Gate, load_track, forward_up_quaternion
from starscream.env.procedural_tracks import geometry_fingerprint, save_track_yaml
from starscream.env.racing_manifold.benchmark_v22 import CloneIndex, _track, requirement_cells
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.env.racing_manifold.transition_corpus_v21 import generate_course, _bezier_return
from scripts.audits.review_v622_geometry import resampled

OUT = ROOT / 'outputs/v6222-pretrain45'
BASE = ROOT / 'configs/exp/v6.21/scratch_5m_dagger_r218.yaml'
CONFIG = ROOT / 'configs/exp/v6.22.2/pretrain45_dagger.yaml'


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def support_course(kind, variant, seed, name):
    rng = np.random.default_rng(seed)
    hand = 1 if variant % 2 == 0 else -1
    if kind.startswith('utt_'):
        # One faithful long/low arrival, not a full lap of compounded stress.
        z = rng.uniform(.77, .90)
        length = rng.uniform(25, 32) if variant == 0 else rng.uniform(34, 42)
        turn = {'utt_straight': rng.uniform(0, 18), 'utt_brake90': rng.uniform(80, 100),
                'utt_reversal': rng.uniform(140, 160)}[kind]
        lengths = [length, rng.uniform(7, 10), rng.uniform(7, 11)]
        turns = [turn, rng.uniform(25, 55), rng.uniform(15, 45)]
        rises = [0, .35, .35]
        count = 8
    else:
        z = rng.uniform(5.8, 6.3)
        count = 9 if kind != 'composition' else 11
        if kind == 'a2rl_drop':
            lengths = list(rng.uniform(6.5, 9, 3))
            turns = [rng.uniform(100, 120), rng.uniform(75, 105), 30]
            rises = [-rng.uniform(1.5, 2.1), rng.uniform(.2, .5), rng.uniform(.3, .6)]
        elif kind == 'a2rl_exit':
            lengths = [rng.uniform(7, 9), rng.uniform(4, 6), rng.uniform(7, 10)]
            turns = [rng.uniform(105, 125), -rng.uniform(105, 130), 35]
            rises = [rng.uniform(.2, .5), -rng.uniform(1.5, 2.1), 0]
        else:
            lengths = [*rng.uniform(6.5, 9, 3), rng.uniform(4.5, 6)]
            turns = [*rng.uniform(100, 120, 3), -rng.uniform(100, 125)]
            rises = [-rng.uniform(1.5, 2), -rng.uniform(1.5, 2), rng.uniform(2.5, 3), -1.2]
    points = [np.array([0., 0., z])]
    heading = 0.
    for length, turn, rise in zip(lengths, turns, rises):
        points.append(points[-1] + [length*np.cos(heading), length*np.sin(heading), rise])
        heading += np.radians(turn) * hand
    motif = np.asarray(points)
    returned = _bezier_return(motif[-1], heading, motif[0], 0., count-len(motif), rng)
    p = np.vstack((motif, returned))
    gates = []
    for i, point in enumerate(p):
        incoming = point - p[i-1]
        outgoing = p[(i+1) % len(p)] - point
        normal = incoming / np.linalg.norm(incoming) + outgoing / np.linalg.norm(outgoing)
        normal[2] = 0
        if np.linalg.norm(normal) < 1e-6:
            normal = outgoing * [1, 1, 0]
        # Exact ground-gate scale at the target; comfortable connectors.
        width = rng.uniform(1.50, 1.60) if kind.startswith('utt_') and i <= 1 else rng.uniform(1.9, 2.3)
        height = min(width, 2*point[2])
        gates.append(Gate(point.astype(np.float32), forward_up_quaternion(normal),
                          np.array([width, height], np.float32), name=f'gate_{i:02d}'))
    # Keep motifs at known corresponding phase for adaptive frontier transfer.
    return _track(name, gates, kind, seed, False)


def build(args):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'candidates.json').exists():
        raise FileExistsError('candidate ledger exists; qualify or create a versioned revision')
    old = read(ROOT/'outputs/course-pools/v621-additions/manifest.json')
    train = [r for r in old['records'] if r['split']=='train']
    core = [r for r in train if r['gate_count'] in (4,6,8,10) and r['name'].startswith('v6201_')]
    additions = [r for r in train if r['name'].startswith('v621_')]
    retained = deepcopy(core + additions)
    assert len(core)==20 and len(additions)==10
    existing_val = [deepcopy(r) for r in old['records'] if r['split']=='validation' and r['family'] in {x['family'] for x in core}]
    specs = []
    for r in additions:
        specs.append(dict(split='validation', family=r['family'], grammar=r['grammar'],
                          count=r['gate_count'], low=r.get('low_gates',False), narrow=r.get('narrow_gates',False)))
    for kind in ('utt_straight','utt_brake90','utt_reversal','a2rl_drop','a2rl_chain','a2rl_exit','composition'):
        for variant in range(3 if kind=='composition' else 2):
            for split in ('train','validation'):
                specs.append(dict(split=split, family=f'{kind}_{variant}', grammar=kind, variant=variant))
    assert len(specs)==40
    protected = [load_track(r['path']) for r in old['records']]
    for filename in ('v6_22_real60.manifest.json','v6_22_real100_hard.manifest.json','v6_22_real100_hard_v2.manifest.json'):
        protected.extend(load_track(str(ROOT / r['path'].removeprefix('/workspace/'))) for r in read(ROOT/'configs/eval'/filename)['records'])
    clones = CloneIndex([resampled(t) for t in protected])
    rows, rejects = [], []
    for slot_index, spec in enumerate(specs):
        slot = spec['split']+'_'+spec['family']
        accepted = 0
        for trial in range(400):
            seed = 622200000 + slot_index*10000 + trial
            name = f'v6222_{slot}_{trial:03d}'
            try:
                if 'count' in spec:
                    t = generate_course(spec['grammar'], spec['count'], seed, name,
                                        low_gates=spec['low'], narrow_gates=spec['narrow'])
                else:
                    t = support_course(spec['grammar'], spec['variant'], seed, name)
                reasons = static_reasons(t)
                g = geometry_record(t)
                distance = clones.distance(resampled(t))
                if distance < .06:
                    reasons.append('protected_shape_clone')
                if max(x['incoming_m'] for x in g['transitions']) > 45:
                    reasons.append('leg_over_45m')
                if reasons:
                    rejects.append(dict(slot=slot,seed=seed,reasons=reasons)); continue
                path = save_track_yaml(t, OUT/'candidates'/f'{name}.yaml').resolve()
                rows.append(dict(spec, name=name, path=str(path), slot=slot, rank=accepted,
                                 source_family=spec['family'], source_stratum='canonical',
                                 seed=seed, gate_count=len(t.gates), geometry=g,
                                 fingerprint=geometry_fingerprint(t), cells=sorted(requirement_cells(t)),
                                 flight_mimic_start_gates=list(range(len(t.gates))),
                                 minimum_protected_clone_distance=distance))
                accepted += 1
                if accepted == args.alternates:
                    break
            except ValueError as exc:
                rejects.append(dict(slot=slot,seed=seed,reasons=[str(exc)]))
        if accepted != args.alternates:
            raise RuntimeError(f'{slot}: only {accepted} candidates')
        print('GENERATED',slot,accepted,flush=True)
    atomic_json(OUT/'candidates.json', dict(records=rows,rejections=rejects,retained=retained,
        retained_validation=existing_val, generator_sha256=digest(__file__), base_sha256=digest(BASE),
        design=dict(train=45,validation=45,retained=30,new_support=15,max_steps=3000,
                    speed_command=16.5,dart_exemptions=0,alternates=args.alternates)))


def qualification_config():
    base = read(BASE)['dagger']
    cfg = dict(output=str(OUT),teacher_config=str(BASE),seed=2026091922,
               max_steps=3000,speed_command=16.5,maximum_solver_failure=.06,maximum_recovery=.10,
               canonical_repeats=2,prefix_repeats=1,teacher_overrides={})
    sources = ['scripts/audits/prepare_v6201.py','scripts/audits/audit_dagger_teacher_labels.py',
               'scripts/audits/prepare_v619_pool.py','scripts/train_privileged_racing.py',
               'starscream/mpcc/controller.py','starscream/mpcc/acados_backend.py',
               'starscream/mpcc/model.py','starscream/mpcc/racing_line.py']
    payload = dict(config=cfg,teacher=base,sources={p:digest(ROOT/p) for p in sources})
    cfg['contract_sha256'] = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    atomic_json(OUT/'qualification-contract.json',payload)
    return cfg


def qualify(args):
    from scripts.audits.prepare_v6201 import qualification_job
    data = read(OUT/'candidates.json'); cfg = qualification_config()
    slots = sorted({r['slot'] for r in data['records']})
    ledger_path = OUT/'qualification.json'
    ledger = read(ledger_path) if ledger_path.exists() else dict(selected=[],records=[],contract=cfg['contract_sha256'])
    assert ledger['contract']==cfg['contract_sha256']
    selected = {r['slot']:r for r in ledger['selected']}
    attempted = {r['name'] for r in ledger['records']}
    clones = CloneIndex([resampled(load_track(r['path'])) for r in data['retained']+data['retained_validation']+list(selected.values())])
    def save():
        atomic_json(ledger_path,dict(contract=cfg['contract_sha256'],selected=list(selected.values()),
            records=ledger['records'],missing_slots=[s for s in slots if s not in selected],complete=len(selected)==len(slots)))
    for rank in range(max(r['rank'] for r in data['records'])+1):
        pending = [r for r in data['records'] if r['rank']==rank and r['slot'] not in selected and r['name'] not in attempted]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as pool:
            futures={pool.submit(qualification_job,(cfg,r)):r for r in pending}
            results={}
            for future in as_completed(futures):
                row=futures[future]; result=future.result(); results[row['name']]=result
                print('QUALIFIED',row['name'],result['qualified'],flush=True)
            # Selection order is deterministic, independent of worker completion.
            for row in pending:
                result=results[row['name']]; attempted.add(row['name'])
                distance=clones.distance(resampled(load_track(row['path'])))
                result['selected_shape_distance']=distance
                if result['qualified'] and distance>=.06:
                    selected[row['slot']]=dict(row,qualification=result)
                    clones.add(resampled(load_track(row['path'])))
                ledger['records'].append(result)
            save()
        print('ADMITTED',len(selected),'/',len(slots),'rank',rank,flush=True)
        if len(selected)==len(slots):break
    if len(selected)!=len(slots):raise SystemExit('unfilled cells; inspect failures and design a versioned repair')


def freeze(args):
    data=read(OUT/'candidates.json'); q=read(OUT/'qualification.json')
    assert q['complete']
    records=deepcopy(data['retained']+data['retained_validation'])
    for r in q['selected']:
        r=deepcopy(r); evidence=r.pop('qualification')
        r.update(qualified=True,status='qualified',qualified_speed_mps=16.5,dart_eligible=True,
            geometry_fingerprint=r['fingerprint'],qualification=dict(contract_sha256=q['contract'],
            selected=dict(teacher_profile='baseline'),evidence_path=str(OUT/'qualification'/r['name']/q['contract'][:12]/'result.json')))
        records.append(r)
    for r in records:
        r.setdefault('dart_eligible',True)
        t=load_track(r['path'])
        assert r['flight_mimic_start_gates']==list(range(len(t.gates)))
        r['geometry']=geometry_record(t)
        r['cells']=sorted(requirement_cells(t))
    train=[r for r in records if r['split']=='train']; val=[r for r in records if r['split']=='validation']
    assert len(train)==len(val)==45
    assert Counter(r['family'] for r in train)==Counter(r['family'] for r in val)
    assert all(n==1 for n in Counter(r['family'] for r in train).values())
    cfg=read(BASE); s=cfg['dagger']; name='starscream-v6.22.2-pretrain45'
    s.update(run_name=name,track_manifest=str(OUT/'manifest.json'),
             dagger_initialization_stats_checkpoint=str(OUT/'normalization.pt'),
             racing_line_cache=str(OUT/'racing-lines'),mpcc_build_root='/tmp/starscream-v6222-pretrain45',
             evaluation_episodes=180,reporting_evaluation_episodes=180,
             tags=['v6.22.2','v6.21-reproduction','pretrain45','prefix-verified','dart-required'])
    weights={r['family']:1. for r in train}
    s.update(track_sampling_family_weights=weights,dagger_replay_family_weights=weights,
             dagger_permanent_expert_required_families=sorted(weights),
             reliability_family_aliases={r['family']:r['grammar'] for r in train})
    for key,rows in [('curriculum',train),('evaluation_curriculum',val),('reporting_evaluation_curriculum',val)]:
        s[key].update(name='v6222_'+key,tracks=[r['path'] for r in rows])
    # Independent seed panel periodically checks whether screening seeds overfit.
    # Final release selection uses a separate 32-episode/course confirmation.
    s['reporting_evaluation_seed']=2036091922
    cfg['checkpoint']['run_name']=name
    cfg['wandb'].update(run_name=name,group='starscream-v6.22.2',
                        local_event_path=f'/workspace/outputs/logs/{name}.events.jsonl')
    cfg['experiment_notes']=dict(recipe='exact v6.21 5M optimizer/collection recipe; new corpus and development panels',
        budget='218 rounds; 360 episodes, 2026 updates, 27001 retained online rows per round; raw steps measured, not a stopping rule',
        validation='45 courses x4 fixed seeds each round for screening and competence; independent x4 every ten rounds; final x32 disjoint seeds per course',
        selection='in-loop top5 are candidates only; release winner requires independent confirmation with uncertainty',
        target='SWIFT/CDRA foundation plus UTT and A2RL support; 60-70% SR is an objective, not an admission claim')
    CONFIG.parent.mkdir(parents=True,exist_ok=True)
    CONFIG.write_text(json.dumps(cfg,indent=2)+'\n')
    atomic_json(OUT/'manifest.json',dict(schema='starscream-procedural-track-manifest-v1',records=records,
        mandatory_teacher_overrides=read(ROOT/'outputs/course-pools/v621-additions/manifest.json')['mandatory_teacher_overrides']))
    atomic_json(OUT/'manifest.lock.json',{r['path']:digest(r['path']) for r in records})
    atomic_json(OUT/'design-summary.json',dict(train=45,validation=45,retained_train=30,new_train=15,
        mean_train_gates=float(np.mean([r['gate_count'] for r in train])),
        train_grammar_counts=dict(Counter(r['grammar'] for r in train)),
        train_commands=dict(Counter(r['qualified_speed_mps'] for r in train)),
        dart_exemptions=0,full_prefix_metadata=True,
        changed_dagger_keys=[k for k in sorted(set(s)|set(read(BASE)['dagger'])) if s.get(k)!=read(BASE)['dagger'].get(k)]))
    print('FROZEN',CONFIG,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['build','qualify','freeze'])
    parser.add_argument('--alternates',type=int,default=8)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    globals()[args.mode](args)
