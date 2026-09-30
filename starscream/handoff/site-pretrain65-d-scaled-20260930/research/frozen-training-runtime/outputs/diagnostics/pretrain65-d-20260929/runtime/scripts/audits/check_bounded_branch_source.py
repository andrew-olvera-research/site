"""Verify selected bounded checkpoint provenance and referenced replay files."""
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'outputs/checkpoints/starscream-v6.21.1.1-plant-selection25-dagger-recovery-fix-scale/best-step-141977167-selection_suite_success-0.2625.pt'
p=torch.load(path,map_location='cpu',weights_only=False)
meta=p['dagger_replay']
sources=[]
for source in meta['sources']:
    lo=int(source.get('min_round',1)); hi=min(int(source['max_round']),int(meta['committed_round']))
    folder=Path(source['root'])
    missing=[r for r in range(lo,hi+1) if not (folder/f'round-{r:05d}.h5').is_file()]
    sources.append(dict(source, missing_rounds=missing))
result={'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'committed_round':meta['committed_round'],'sources':sources,
    'payload_keys':sorted(p.keys()),
    'quality_contract':{k:v for k,v in meta['contract']['trajectory_quality_v1'].items() if k!='track_bounds'},
    'note':'Checks checkpoint readability and shard existence only; not a full replay reconstruction or training resume.'}
out=ROOT/'outputs/diagnostics/pretraining-mode-decision-20260928/bounded-branch-source.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
