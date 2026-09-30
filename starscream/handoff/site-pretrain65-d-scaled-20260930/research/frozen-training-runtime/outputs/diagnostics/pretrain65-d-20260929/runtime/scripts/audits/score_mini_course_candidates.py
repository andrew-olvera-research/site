"""Rank one-train/two-selection-course family triplets by requirement overlap."""
import json
import itertools
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.env.tracks import load_track
from starscream.env.racing_manifold.benchmark_v22 import requirement_cells
from starscream.evaluation_suite import repository_path

ROOT = Path(__file__).resolve().parents[2]
train = json.loads((ROOT/'outputs/diagnostics/v62111-train65-teacher-v4/frozen/approved/manifest.json').read_text())['records']
val = json.loads((ROOT/'configs/eval/v6211_vision_selection25.json').read_text())['records']
byfamily = defaultdict(list)
for r in val: byfamily[r['family']].append(r)
cache = {}
def cells(path):
    if path not in cache:
        cache[path] = set(requirement_cells(load_track(repository_path(path))))
    return cache[path]
rows=[]
for family, group in byfamily.items():
    if len(group)!=2: continue
    for t in train:
        if t.get('grammar') != family or not t.get('qualified'): continue
        a = cells(t['path'])
        scores=[]
        for v in group:
            b=cells(v['path']);scores.append(len(a&b)/len(a|b))
        rows.append((sum(scores)/2,min(scores),family,t['name'],tuple(r['slot'] for r in group),scores))
for row in sorted(rows,reverse=True)[:25]: print(row)

slalom_train=[r for r in train if r.get('grammar')=='slalom' and r.get('split')=='train' and r.get('qualified')]
slalom_val=byfamily['slalom']
print('TWO_TRAIN_UNION')
for i,a in enumerate(slalom_train):
    for b in slalom_train[i+1:]:
        union=cells(a['path']) | cells(b['path'])
        scores=[len(union&cells(v['path']))/len(cells(v['path'])) for v in slalom_val]
        print((sum(scores)/2,min(scores),a['name'],b['name'],scores))
print('NESTED_CANDIDATES')
for n in (2,4,6):
    ranked=[]
    for group in itertools.combinations(slalom_train,n):
        union=set().union(*(cells(r['path']) for r in group))
        scores=[len(union&cells(v['path']))/len(cells(v['path'])) for v in slalom_val]
        ranked.append((min(scores),sum(scores)/2,-sum(r['gate_count'] for r in group),
                       tuple(r['name'] for r in group),scores))
    for row in sorted(ranked,reverse=True)[:3]: print(n,row)
