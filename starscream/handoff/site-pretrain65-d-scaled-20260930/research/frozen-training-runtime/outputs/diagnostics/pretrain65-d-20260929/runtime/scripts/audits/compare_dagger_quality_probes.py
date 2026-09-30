"""Summarize matched frozen-policy collector probes; no learning-effect claim."""
import json
from pathlib import Path
import numpy as np

base=Path('outputs/diagnostics')
rows=[]
for name in ['plant-quality-matched-unbounded65','plant-quality-independent65']:
 p=base/name
 report=json.loads((p/'report.json').read_text())
 for result in report['results']:
  data=np.load(p/f"beta-{result['beta']}.npz")
  q=data['trajectory']; tracks=data['tracks']; names=data['track_names']
  keep=data['retained_indices']
  summary={k:v for k,v in result.items() if k not in ['per_track','observed_speed_commands','seconds']}
  summary.update(arm='unbounded_audit' if report['quality']['audit_only'] else 'bounded',
   report=str(p/'report.json'), seed=report['seed_base'], canonical_starts=report['canonical_starts'],
   internal_recovery_query_fraction=result['counters']['internal_teacher_recovery_steps']/result['environment_steps'],
   valid_row_recovery_class_fraction=float(np.mean(q[:,0]==2)),
   retained_tracks=len(np.unique(tracks[keep])), valid_distance_p99=float(np.quantile(q[:,4],.99)),
   valid_distance_maximum=float(q[:,4].max()),
   retained_distance_p99=float(np.quantile(q[keep,4],.99)),
   retained_distance_maximum=float(q[keep,4].max()),
   qualification='audit recovery is observational and unqualified' if report['quality']['audit_only'] else 'recovery admitted only after bounded teacher continuation')
  per_track=[]
  for i,name in enumerate(names):
   x=q[tracks==i];bounds=report['quality']['track_bounds'][str(name)]
   retained=q[keep[tracks[keep]==i]]
   if not report['quality']['audit_only']:
    assert np.all(retained[:,4] <= bounds['maximum_distance'])
   per_track.append(dict(track=str(name),valid_rows=len(x),recovery_rows=int((x[:,0]==2).sum()),
    beyond_intervention_rows=int((x[:,4]>bounds['intervention_distance']).sum()),
    beyond_hard_bound_rows=int((x[:,4]>bounds['maximum_distance']).sum()),
    misses=int(x[:,1].sum())))
  summary['per_track']=per_track
  rows.append(summary)
result=dict(results=rows,limitations=[
 'Same checkpoint, initial seeds, courses, speeds, plant and worker count; intervention changes subsequent state/RNG visitation.',
 'Stopped attempts are collector truncations, not evaluation crashes; compare exposure and costs, not raw completion as policy efficacy.',
 'Valid-row statistics exclude invalid solver labels. All-query counters are reported separately.',
 'No learner update in this comparison; no claim of improved learned racing or optimal quotas.'])
p=base/'plant-trajectory-mix-v2/matched-collector-comparison.json'
p.write_text(json.dumps(result,indent=2)+'\n')
for row in rows:print(json.dumps({k:v for k,v in row.items() if k not in ['per_track','counters','qualification','report','seed','canonical_starts','audit_only','sampled_fractions','quota_shortfalls']}))
