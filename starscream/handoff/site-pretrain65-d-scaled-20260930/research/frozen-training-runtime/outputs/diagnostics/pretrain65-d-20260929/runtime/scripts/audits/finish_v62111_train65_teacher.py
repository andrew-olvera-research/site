"""Wait for durable searches, repair unqualified baselines, then freeze evidence.

Does not start tuners, train a policy, or change the experiment configuration.
Run only once after starting the resumable primary and refinement searches.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path('outputs/diagnostics/v62111-train65-teacher-v4')
while True:
    summaries=[json.loads(p.read_text()) for p in (root/'summary.json',root/'local-refinement/summary.json')]
    if all(s['complete'] for s in summaries):break
    time.sleep(15)
print('stage=reliability-repair',flush=True)
subprocess.run([sys.executable,'-u','scripts/audits/repair_v62111_train65_teacher.py','--workers','6'],check=True)
print('stage=reliability-rescue',flush=True)
subprocess.run([sys.executable,'-u','scripts/audits/rescue_v62111_train65_teacher.py','--workers','6'],check=True)
print('stage=freeze',flush=True)
subprocess.run([sys.executable,'scripts/audits/freeze_v62111_train65_teacher.py',
    '--search',str(root),'--refinement',str(root/'local-refinement'),
    '--repair',str(root/'reliability-repair'),
    '--rescue',str(root/'reliability-rescue')],check=True)
print('stage=frozen',flush=True)
