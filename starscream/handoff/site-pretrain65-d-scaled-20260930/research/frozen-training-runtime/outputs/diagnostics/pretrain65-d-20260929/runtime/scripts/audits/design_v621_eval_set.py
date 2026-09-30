"""Design the v6.21 50-track held-out evaluation set by weighted transition-cell set cover.

Candidates: real courses (Swift, CDRA, A2RL, MultiGP UTT/GQ), the r4 v6201 transition
candidates that were never selected into the rl.4+ sampling pool, the v6.19 Green-style
spline pool, and the named reference shapes. Every candidate is described by ordered
transition cells (turn bin x rise x wrong-side x incoming length) and ordered bigrams;
the greedy picks tracks that add the most uncovered cells, with extra weight on cells the
v6.20.1 training corpus supports weakly. Balance quotas keep grammar/gate-count spread.
Run inside the container (needs starscream).
"""
import argparse, collections, json, shutil
from pathlib import Path
import numpy as np
import yaml
from starscream.env.tracks import load_track
from starscream.env.procedural_tracks import geometry_fingerprint, save_track_yaml
from starscream.env.racing_manifold.corpus_coverage import geometry_record, clone_distance
from starscream.env import multigp_tracks as mt

ROOT = Path('/workspace')
POOL_MANIFEST = ROOT/'outputs/diagnostics/v6201-teacher-speed/frozen/manifest.json'
R4 = ROOT/'outputs/course-pools/v6201-transitions-r4/candidates.json'
R6Q = ROOT/'outputs/course-pools/v6201-transitions-r6/qualification.json'
V619 = ROOT/'outputs/course-pools/v619-green-inspired/manifest.json'
NAMED_REAL = ROOT/'outputs/procedural-tracks/named-real-v4.2-dagger-candidates-v2/tracks'
ASSETS = ROOT/'starscream/assets/tracks'
OUT = ROOT/'outputs/course-pools/v621-holdout-eval'
SUITE = ROOT/'configs/eval/v6_21_holdout_eval_50.yaml'

REAL = {  # name -> (path, note)
    'swift_champion_2022_exact': (ASSETS/'swift_champion_2022_exact.yaml', 'exact released Swift champion course; diving hairpins into 1.45 m gates, stacked split-S'),
    'multigp_cdra_2026_reconstructed': (ASSETS/'multigp_cdra_2026_reconstructed.yaml', 'official 2026 CDRA diagram; three consecutive >120 deg planar hairpins at 0.9 m'),
    'a2rl_s2_2026_source_consistent_v2': (ASSETS/'a2rl_s2_2026_source_consistent_v2.yaml', 'A2RL season-2 metric reconstruction; 12 gates, climbs to 5.4 m, stacked reverse entries'),
    'multigp_utt01': (NAMED_REAL/'canonical_train/multigp_utt01.yaml', 'UTT1: 56-71 m straights into 150+ deg flag hairpins'),
    'multigp_utt02_tsunami': (NAMED_REAL/'canonical_train/multigp_utt02_tsunami.yaml', 'UTT2: long straights plus 3 m-spaced 4x3 m gate clusters'),
    'multigp_utt03_bessel_run': (NAMED_REAL/'canonical_train/multigp_utt03_bessel_run.yaml', 'UTT3: straight run with two 153 deg hairpins'),
    'multigp_utt04_high_voltage': (NAMED_REAL/'pure_holdout/multigp_utt04_high_voltage.yaml', 'UTT4: 3.2 m over-gate on a hairpin oval'),
    'multigp_utt05_nautilus': (NAMED_REAL/'pure_holdout/multigp_utt05_nautilus.yaml', 'UTT5: same-sign tightening spiral (149/90/79/90 deg)'),
    'multigp_utt06_fury': (NAMED_REAL/'canonical_train/multigp_utt06_fury.yaml', 'UTT6: dense 3D 90-deg corners, 3 m gates, 180 deg reversal, 2.3 m climb'),
    'multigp_utt08_revenge': (NAMED_REAL/'canonical_train/multigp_utt08_revenge.yaml', 'UTT8: 2 m-spaced climbing ladder gates and 3 m drop'),
    'multigp_utt10_prairie_rage': (NAMED_REAL/'canonical_train/multigp_utt10_prairie_rage.yaml', 'UTT10: 1.5 m flag slalom chains'),
    'multigp_global_qualifier_2026': (NAMED_REAL/'canonical_train/multigp_global_qualifier_2026.yaml', 'MultiGP 2026 Global Qualifier: 260 m 3D, 180 deg climbs to 6.3 m'),
}
FORCED_REAL = ['swift_champion_2022_exact', 'multigp_cdra_2026_reconstructed', 'a2rl_s2_2026_source_consistent_v2']
# Route-checkpoint artifacts: consecutive checkpoints under 2 m apart or 22-point routes
# exceed the six-gate route horizon and would be scored on bookkeeping, not flight.
EXCLUDED_REAL = {'multigp_global_qualifier_2026', 'multigp_utt10_prairie_rage', 'multigp_utt02_tsunami'}
NAMED_SHAPES = ['figure8', 'big_s', 'kidney', 'split_s', 'vertical_3d', 'swift_eval_inspired']

TURN_EDGES = [0, 30, 60, 90, 120, 150, 181]


def symbolize(record):
    """Per-transition unigram cells and ordered bigram cells."""
    ts = record['transitions']; n = len(ts)
    uni, sym = [], []
    for t in ts:
        tb = next(i for i in range(6) if TURN_EDGES[i] <= t['turn_deg'] < TURN_EDGES[i+1])
        rise = -1 if t['height_change_m'] < -0.75 else (1 if t['height_change_m'] > 0.75 else 0)
        exit_side = int(t['preceding_gate_on_exit_side'])
        inc = 0 if t['incoming_m'] < 5 else (1 if t['incoming_m'] < 12 else 2)
        width = 0 if t['width_m'] <= 1.55 else (1 if t['width_m'] <= 2.2 else 2)
        low = int(t['gate_center_height_m'] <= 1.2)
        uni.append(f'u:{tb}:{rise}:{exit_side}:{inc}')
        uni.append(f'w:{width}'); uni.append(f'low:{low}')
        if t['reverse_entry']: uni.append('rev')
        sym.append(f'{tb}:{rise}')
    bi = [f'b:{sym[i]},{sym[(i+1)%n]}' for i in range(n)]
    tri = [f't:{sym[i]},{sym[(i+1)%n]},{sym[(i+2)%n]}' for i in range(n)]
    return set(uni) | set(bi) | set(tri)


def load_record(name, path, **meta):
    track = load_track(path)
    rec = geometry_record(track)
    rec.update(name=name, path=str(path), fingerprint=geometry_fingerprint(track), **meta)
    rec['cells'] = symbolize(rec)
    p = np.array([g.position for g in track.gates]); rec['length_m'] = float(np.linalg.norm(np.roll(p, -1, 0)-p, axis=1).sum())
    return rec, track


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--real', type=int, default=8); ap.add_argument('--generated', type=int, default=42)
    ap.add_argument('--min-spline', type=int, default=8); ap.add_argument('--max-spline', type=int, default=12)
    ap.add_argument('--max-named', type=int, default=3)
    ap.add_argument('--max-per-grammar', type=int, default=8); ap.add_argument('--max-per-gate-count', type=int, default=7)
    ap.add_argument('--qualified-bonus', type=float, default=3.0)
    ap.add_argument('--screen', type=Path, default=OUT/'screen.json', help='raceability screen ledger: failed courses are excluded, passed ones kept')
    ap.add_argument('--gap-bonus', type=float, default=2.0, help='extra weight for cells with zero train support')
    ap.add_argument('--weak-bonus', type=float, default=1.0, help='extra weight for cells with <=3 train support')
    a = ap.parse_args()

    pool = json.loads(POOL_MANIFEST.read_text())['records']
    pool_names = {r['name'] for r in pool if r['split'] in ('train', 'validation')}
    pool_tracks = {}
    train_cells = collections.Counter()
    for r in pool:
        if r['split'] == 'report': continue
        rec, track = load_record(r['name'], r['path'])
        pool_tracks[r['name']] = track
        if r['split'] == 'train':
            for c in rec['cells']: train_cells[c] += 1
    print('train cells', len(train_cells))

    cands = {}
    r6 = json.loads(R6Q.read_text()); qualified = {x['name'] for x in r6['records'] if str(x.get('qualified')) == 'True'}
    screened_ok, screened_bad = set(), set()
    if a.screen.exists():
        for x in json.loads(a.screen.read_text())['records']:
            (screened_ok if x['qualified'] else screened_bad).add(x['name'])
        print('screen ledger: passed', len(screened_ok), 'failed', len(screened_bad))
    qualified |= screened_ok
    for r in json.loads(R4.read_text())['records']:
        if r['name'] in pool_names or r['name'] in screened_bad: continue
        rec, track = load_record(r['name'], r['path'], source='v6201_r4', grammar=r['grammar'], gate_count=int(r['gate_count']), qualified=r['name'] in qualified)
        # anti-clone against the rl sampling pool (same gate count, reflection-aware)
        dmin = min((clone_distance(track, pool_tracks[n]) or 9 for n in pool_names if len(pool_tracks[n].gates) == len(track.gates)), default=9)
        if dmin < 0.12: continue
        rec['pool_clone_distance'] = float(dmin); cands[r['name']] = rec
    for r in json.loads(V619.read_text())['records']:
        if r['name'] in NAMED_SHAPES: continue
        rec, _ = load_record(r['name'], r['path'], source='v619_spline', grammar='spline', gate_count=len(load_track(r['path']).gates), qualified=str(r.get('qualified')) == 'True', v619_split=r['split'])
        cands[r['name']] = rec
    for n in NAMED_SHAPES:
        rec, _ = load_record(n, ASSETS/f'{n}.yaml', source='named_shape', grammar='named', gate_count=len(load_track(ASSETS/f'{n}.yaml').gates), qualified=n in {'figure8', 'big_s', 'kidney', 'split_s'})
        cands[n] = rec
    real = {}
    for n, (p, note) in REAL.items():
        rec, _ = load_record(n, p, source='real', grammar='real', gate_count=len(load_track(p).gates), qualified=False, note=note)
        real[n] = rec
    print('generated candidates', len(cands), collections.Counter(c['source'] for c in cands.values()))

    def weight(cell):
        if cell.startswith(('w:', 'low:')): return 0.0  # aperture/height are not behaviours to cover
        s = train_cells.get(cell, 0)
        base = 0.5 if cell.startswith('t:') else 1.0
        return base * (1.0 + (a.gap_bonus if s == 0 else 0.0) + (a.weak_bonus if s <= 3 else 0.0))

    covered = set(); chosen = []
    def gain(rec): return sum(weight(c) for c in rec['cells'] - covered)
    def take(rec, why):
        covered.update(rec['cells']); chosen.append((rec, why))
    # --- real courses
    for n in FORCED_REAL: take(real[n], 'forced')
    while sum(1 for r, _ in chosen if r['source'] == 'real') < a.real:
        best = max((r for r in real.values() if r['name'] not in {c['name'] for c, _ in chosen} and r['name'] not in EXCLUDED_REAL), key=gain)
        take(best, f'greedy gain {gain(best):.1f}')
    # --- generated: quotas first (each v6201 grammar x parity of gate count), then free greedy
    selected_names = lambda: {c['name'] for c, _ in chosen}
    def ok(rec):
        if rec['name'] in selected_names(): return False
        picks = [c for c, _ in chosen if c['source'] != 'real']
        if rec['source'] == 'v619_spline' and sum(1 for c in picks if c['source'] == 'v619_spline') >= a.max_spline: return False
        if rec['source'] == 'named_shape' and sum(1 for c in picks if c['source'] == 'named_shape') >= a.max_named: return False
        if rec['source'] == 'v6201_r4':
            if sum(1 for c in picks if c['grammar'] == rec['grammar']) >= a.max_per_grammar: return False
            if sum(1 for c in picks if c['source'] == 'v6201_r4' and c['gate_count'] == rec['gate_count']) >= a.max_per_gate_count: return False
        for c in picks:  # anti-clone inside the eval set
            if c['gate_count'] == rec['gate_count'] and c['source'] == rec['source'] == 'v6201_r4':
                if clone_distance(load_track(c['path']), load_track(rec['path'])) < 0.12: return False
        return True
    for n in sorted(screened_ok):
        if n in cands and ok(cands[n]): take(cands[n], 'kept: passed raceability screen')
    grammars = ('banking', 'slalom', 'braking_reversal', 'stacked_reversal', 'go_around')
    quota = [(g, (k,)) for g in grammars for k in (4, 6, 8, 10)] + [(g, (5, 7, 9)) for g in grammars]
    # A quota cell whose candidates keep failing MPCC qualification (>= 3 screened failures)
    # is relaxed to the neighbouring gate counts so the set can converge.
    failed_cells = collections.Counter()
    for n in screened_bad:
        parts = n.split('_'); failed_cells[('_'.join(parts[3:-1]), int(parts[2]))] += 1
    for g, ks in quota:
        if any(c['source'] == 'v6201_r4' and c['grammar'] == g and c['gate_count'] in ks for c, _ in chosen): continue
        if all(failed_cells[(g, k)] >= 3 for k in ks):
            ks = tuple(sorted({max(4, min(10, k + d)) for k in ks for d in (-1, 1, -2)}))
            print(f'quota {g}/{ks} relaxed after repeated qualification failures')
        opts = [c for c in cands.values() if c['source'] == 'v6201_r4' and c['grammar'] == g and c['gate_count'] in ks and ok(c)]
        if not opts: continue
        best = max(opts, key=lambda c: gain(c) + (a.qualified_bonus if c['qualified'] else 0.0))
        take(best, f'quota {g}/{"|".join(map(str, ks))}: gain {gain(best):.1f}')
    while sum(1 for c, _ in chosen if c['source'] == 'v619_spline') < a.min_spline:
        opts = [c for c in cands.values() if c['source'] == 'v619_spline' and ok(c)]
        best = max(opts, key=gain); take(best, f'spline quota: gain {gain(best):.1f}')
    while sum(1 for r, _ in chosen if r['source'] != 'real') < a.generated:
        opts = [c for c in cands.values() if ok(c)]
        best = max(opts, key=lambda c: gain(c) + (a.qualified_bonus if c['qualified'] else 0.0))
        take(best, f'free greedy: gain {gain(best):.1f}')

    # --- write outputs
    if (OUT/'tracks').exists(): shutil.rmtree(OUT/'tracks')
    (OUT/'tracks').mkdir(parents=True)
    records = []
    for rec, why in chosen:
        dst = OUT/'tracks'/f"{rec['name']}.yaml"; shutil.copy(rec['path'], dst)
        assert geometry_fingerprint(load_track(dst)) == rec['fingerprint']
        records.append({k: v for k, v in rec.items() if k not in ('cells',)} | dict(path=str(dst), selection=why, cells=sorted(rec['cells'])))
    eval_cells = collections.Counter(c for rec, _ in chosen for c in rec['cells'])
    coverage = dict(train_cells=dict(train_cells), eval_cells=dict(eval_cells),
        eval_only_cells=sorted(c for c in eval_cells if c not in train_cells),
        weak_train_cells_in_eval=sorted(c for c in eval_cells if 0 < train_cells.get(c, 0) <= 3),
        counts=dict(real=sum(1 for r, _ in chosen if r['source'] == 'real'), by_source=dict(collections.Counter(r['source'] for r, _ in chosen)),
                    by_grammar=dict(collections.Counter(r['grammar'] for r, _ in chosen)), by_gate_count=dict(collections.Counter(r['gate_count'] for r, _ in chosen)),
                    qualified=sum(1 for r, _ in chosen if r.get('qualified'))))
    (OUT/'manifest.json').write_text(json.dumps(dict(schema='starscream-v621-holdout-eval-v1', records=records, coverage=coverage), indent=1, default=float))
    suite = dict(schema='starscream-real-course-suite-v1',
        description='v6.21 50-track held-out evaluation set: 8 real courses + 42 generated, chosen by weighted transition-cell set cover (scripts/audits/design_v621_eval_set.py). Never used for training or checkpoint selection.',
        active=[dict(name=rec['name'], track=str(Path(rec['path']).relative_to(ROOT)), geometry_fingerprint=rec['fingerprint']) for rec, _ in chosen], pending_geometry=[])
    SUITE.write_text(yaml.safe_dump(suite, sort_keys=False))
    print('\nSELECTED')
    for rec, why in chosen:
        print(f"  {rec['name']:<48} {rec['source']:<12} {rec['grammar']:<17} g{rec['gate_count']:<3} L{rec['length_m']:5.0f}m q={int(bool(rec.get('qualified')))} {why}")
    print('\ncoverage', json.dumps(coverage['counts']))
    print('cells: train', len(train_cells), 'eval', len(eval_cells), 'eval-only', len(coverage['eval_only_cells']), 'weak-train-in-eval', len(coverage['weak_train_cells_in_eval']))


if __name__ == '__main__':
    main()
