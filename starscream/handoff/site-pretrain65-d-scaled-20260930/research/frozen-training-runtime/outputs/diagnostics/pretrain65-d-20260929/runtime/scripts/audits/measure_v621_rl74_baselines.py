import json
from pathlib import Path
for name in ['starscream-v6.21.rl.7.3-b-time-mild','starscream-v6.21-scratch-5m-dagger-r218']:
 e=[json.loads(l) for l in Path('outputs/logs',name+'.events.jsonl').read_text().splitlines()]
 key='train/performance/full_cycle_seconds' if 'rl.7' in name else 'train/round_seconds'
 rows=[r for r in e if key in r['metrics']];print(name,'n',len(rows),'steps',rows[-1]['step'],'seconds',sum(r['metrics'][key] for r in rows),'sps',rows[-1]['step']/sum(r['metrics'][key] for r in rows));print('stepkeys',{k:v for k,v in rows[-1]['metrics'].items() if 'step' in k and k.count('/')<3})
