"""Per-transition gap analysis on the v6.21 held-out suite and minimal train-set additions.

For every evaluated checkpoint: join per-gate survival (track/<name>/p{k}) with the
transition cells of the suite manifest, compute conditional survival per transition and
per cell, and rank cells by failure mass against v6.20.1 training support. Then choose
the fewest candidate courses (r4 pool minus eval set minus rl pool) that cover the
failing cells, and list failing cells no candidate covers (needs a new grammar).
Host-side: JSON only.
"""
import argparse, collections, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITE_MANIFEST = ROOT/'outputs/course-pools/v621-holdout-eval/manifest.json'
POOL_MANIFEST = ROOT/'outputs/diagnostics/v6201-teacher-speed/frozen/manifest.json'
R4 = ROOT/'outputs/course-pools/v6201-transitions-r4/candidates.json'
TURN_EDGES = [0, 30, 60, 90, 120, 150, 181]


def cells_for_transition(t, prev, nxt):
    """Behavioural unigram + ordered bigram cells for one arrival transition (host re-implementation)."""
    def sym(x):
        tb = next(i for i in range(6) if TURN_EDGES[i] <= x['turn_deg'] < TURN_EDGES[i+1])
        rise = -1 if x['height_change_m'] < -0.75 else (1 if x['height_change_m'] > 0.75 else 0)
        return tb, rise
    tb, rise = sym(t); inc = 0 if t['incoming_m'] < 5 else (1 if t['incoming_m'] < 12 else 2)
    out = {f'u:{tb}:{rise}:{int(t["preceding_gate_on_exit_side"])}:{inc}', f'b:{sym(prev)[0]}:{sym(prev)[1]},{tb}:{rise}'}
    if t.get('reverse_entry'): out.add('rev')
    return out


def describe(cell):
    names = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180']
    rise = {'-1': 'drop', '0': 'level', '1': 'climb'}
    if cell == 'rev': return 'reverse (wrong-side) entry'
    if cell.startswith('u:'):
        tb, r, ex, inc = cell[2:].split(':')
        return f"turn {names[int(tb)]}deg, {rise[r]}, {'wrong-side approach, ' if ex == '1' else ''}incoming {['<5', '5-12', '>=12'][int(inc)]} m"
    if cell.startswith('b:'):
        a, b = cell[2:].split(','); (ta, ra), (tb, rb) = a.split(':'), b.split(':')
        return f"{names[int(ta)]}deg {rise[ra]} -> {names[int(tb)]}deg {rise[rb]}"
    return cell


def transition_rows(name, transitions, metrics, n):
    p = [metrics.get(f'track/{name}/p{i}') for i in range(1, n+1)]
    if p[0] is None: return []
    rows = []
    for k in range(1, n):
        if not p[k-1]: continue
        t, prev, nxt = transitions[k], transitions[k-1], transitions[(k+1) % n]
        rows.append(dict(course=name, gate=k, reached=p[k-1], cond=p[k]/p[k-1], lost=p[k-1]-p[k], cells=cells_for_transition(t, prev, nxt), t=t))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', nargs='+', required=True, help='label=path/to/eval.json')
    ap.add_argument('--episodes', type=int, default=32)
    ap.add_argument('--fail-threshold', type=float, default=0.85, help='conditional survival below this marks a cell as failing')
    ap.add_argument('--max-additions', type=int, default=8)
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/course-pools/v621-holdout-eval/gap_analysis.json')
    a = ap.parse_args()
    suite = json.loads(SUITE_MANIFEST.read_text())
    pool = json.loads(POOL_MANIFEST.read_text())['records']
    train_support = collections.Counter()
    for r in pool:
        if r['split'] != 'train': continue
        ts = r['transitions']; n = len(ts)
        for k in range(n):
            for c in cells_for_transition(ts[k], ts[k-1], ts[(k+1) % n]): train_support[c] += 1
    report = {}
    for spec in a.results:
        label, path = spec.split('=', 1); m = json.loads(Path(path).read_text())['metrics']
        rows = []
        for rec in suite['records']:
            rows += transition_rows(rec['name'], rec['transitions'], m, rec['gate_count'])
        # per-course summary
        courses = []
        for rec in suite['records']:
            pre = f"track/{rec['name']}/"
            courses.append(dict(name=rec['name'], source=rec['source'], grammar=rec['grammar'], gates=rec['gate_count'], qualified=bool(rec.get('qualified')),
                success=m.get(pre+'full_course_success'), crash=m.get(pre+'crash_rate'), timed_out=m.get(pre+'timed_out'), gate_fraction=(m.get(pre+'mean_gates') or 0)/rec['gate_count']))
        by_source = collections.defaultdict(list)
        for c in courses: by_source[c['source']].append(c['success'])
        # per-cell aggregation: failure mass = episodes lost at transitions carrying the cell
        cell = collections.defaultdict(lambda: dict(reached=0.0, lost=0.0, transitions=0, courses=set()))
        for r in rows:
            for c in r['cells']:
                cell[c]['reached'] += r['reached']; cell[c]['lost'] += r['lost']; cell[c]['transitions'] += 1; cell[c]['courses'].add(r['course'])
        table = []
        for c, v in cell.items():
            surv = 1 - v['lost']/v['reached'] if v['reached'] > 0 else None
            table.append(dict(cell=c, meaning=describe(c), conditional_survival=surv, episodes_lost=v['lost']*a.episodes, transitions=v['transitions'], courses=len(v['courses']), train_support=train_support.get(c, 0)))
        table.sort(key=lambda x: -x['episodes_lost'])
        failing = [x for x in table if x['conditional_survival'] is not None and x['conditional_survival'] < a.fail_threshold and x['transitions'] >= 2]
        worst = sorted(rows, key=lambda r: r['cond'])[:25]
        report[label] = dict(macro=dict(success=sum(c['success'] for c in courses)/len(courses), by_source={k: sum(v)/len(v) for k, v in by_source.items()},
            by_grammar={g: sum(c['success'] for c in courses if c['grammar'] == g)/max(1, sum(1 for c in courses if c['grammar'] == g)) for g in sorted({c['grammar'] for c in courses})}),
            courses=courses, cells=table, failing_cells=failing,
            worst_transitions=[dict(course=r['course'], gate=r['gate'], cond=r['cond'], reached=r['reached'], turn=r['t']['turn_deg'], dz=r['t']['height_change_m'], incoming=r['t']['incoming_m'], align=r['t']['incoming_alignment'], cells=sorted(r['cells'])) for r in worst])
    # --- minimal train additions for the FIRST label (base model): weighted set cover over failing cells
    base = report[a.results[0].split('=')[0]]
    weights = {x['cell']: x['episodes_lost'] / (1 + x['train_support']) for x in base['failing_cells']}
    eval_names = {r['name'] for r in suite['records']}; pool_names = {r['name'] for r in pool}
    cands = []
    for r in json.loads(R4.read_text())['records']:
        if r['name'] in eval_names or r['name'] in pool_names: continue
        ts = r['transitions']; n = len(ts); cs = set()
        for k in range(n): cs |= cells_for_transition(ts[k], ts[k-1], ts[(k+1) % n])
        cands.append(dict(name=r['name'], grammar=r['grammar'], gate_count=r['gate_count'], cells=cs))
    covered = set(); additions = []
    for _ in range(a.max_additions):
        best = max(cands, key=lambda c: sum(weights.get(x, 0) for x in c['cells'] - covered))
        g = sum(weights.get(x, 0) for x in best['cells'] - covered)
        if g <= 0: break
        newly = sorted((best['cells'] - covered) & set(weights)); covered |= best['cells']
        additions.append(dict(name=best['name'], grammar=best['grammar'], gate_count=best['gate_count'], gain=g, covers=[dict(cell=c, meaning=describe(c)) for c in newly]))
        cands.remove(best)
    uncoverable = [dict(cell=c, meaning=describe(c), episodes_lost=w*(1+train_support.get(c, 0))) for c, w in weights.items() if not any(c in x['cells'] for x in cands) and c not in covered]
    remaining = sum(w for c, w in weights.items() if c not in covered)
    report['train_additions'] = dict(additions=additions, covered_weight=sum(weights.values())-remaining, total_weight=sum(weights.values()), uncoverable_cells=uncoverable)
    a.out.write_text(json.dumps(report, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else float(o)))
    for label, rep in report.items():
        if label == 'train_additions': continue
        print(f"\n### {label}: macro {100*rep['macro']['success']:.1f}%  by source", {k: f'{100*v:.1f}%' for k, v in rep['macro']['by_source'].items()})
        print('  by grammar', {k: f'{100*v:.1f}%' for k, v in rep['macro']['by_grammar'].items()})
        print('  courses:'); [print(f"    {c['name']:<48} {c['source']:<12} g{c['gates']:<3} success {100*c['success']:5.1f}% crash {100*c['crash']:5.1f}% timeout {100*(c['timed_out'] or 0):4.1f}% gates {100*c['gate_fraction']:5.1f}%") for c in sorted(rep['courses'], key=lambda c: c['success'])]
        print('  failing cells (survival < threshold, ranked by episodes lost):')
        for x in rep['failing_cells'][:25]: print(f"    {x['cell']:<24} surv {100*x['conditional_survival']:5.1f}%  lost {x['episodes_lost']:6.1f} eps over {x['transitions']:2d} transitions/{x['courses']:2d} courses  train support {x['train_support']:3d}  | {x['meaning']}")
        print('  worst transitions:')
        for r in rep['worst_transitions'][:15]: print(f"    {r['course']:<44} gate {r['gate']:2d} cond {100*r['cond']:5.1f}% (reached {100*r['reached']:5.1f}%) turn {r['turn']:5.0f} dz {r['dz']:+5.2f} in {r['incoming']:5.1f} align {r['align']:+.2f}")
    ta = report['train_additions']
    print(f"\n### minimal train additions (greedy, weighted by base failure mass / (1+train support)): cover {ta['covered_weight']:.1f} of {ta['total_weight']:.1f}")
    for x in ta['additions']: print(f"  + {x['name']:<40} {x['grammar']:<17} g{x['gate_count']:<3} gain {x['gain']:6.1f} covers {[c['cell'] for c in x['covers']]}")
    print('  uncoverable by any r4 candidate (needs new generator grammar):'); [print(f"    {x['cell']:<24} {x['meaning']}  (~{x['episodes_lost']:.0f} eps lost)") for x in ta['uncoverable_cells']]


if __name__ == '__main__':
    main()
