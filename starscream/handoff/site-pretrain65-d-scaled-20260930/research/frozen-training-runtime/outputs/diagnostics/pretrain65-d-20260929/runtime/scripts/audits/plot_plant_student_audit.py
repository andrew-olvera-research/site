"""Plot saved metrics and the bounded recovery diagnostic."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out=Path('outputs/diagnostics/plant-student-saved-audit')
d=json.loads((out/'audit.json').read_text())
student=[json.loads(l) for l in Path('outputs/vision-distillation/v62111-plant-ekf-readout-async-48r/metrics.jsonl').read_text().splitlines()]
teacher=[json.loads(l) for l in Path('outputs/logs/starscream-v6.21.1.1-plant-selection25-dagger.events.jsonl').read_text().splitlines()]
teacher=[r for r in teacher if 'eval/selection_suite_success' in r['metrics']]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
ax=axs[0,0]
x=np.array([r['step']/1e6 for r in teacher]);y=np.array([r['metrics']['eval/selection_suite_success']*100 for r in teacher])
ax.plot(x,y,color='#3577ad',alpha=.3,lw=1)
ax.plot(x[19:],np.convolve(y,np.ones(20)/20,'valid'),color='#3577ad',lw=2,label='20-evaluation mean')
ax.set(title='Teacher still improves late in training',xlabel='Collected steps (millions)',ylabel='Selection25 weighted completion (%)')
ax.legend(frameon=False)
ax=axs[0,1]
e=[r for r in student if 'selection_score' in r]
ax.plot([r['round']+1 for r in e],[r['selection_score']*100 for r in e],color='#d47735',marker='o',ms=3)
ax.scatter([40],[45.25],color='#9c3324',s=60,zorder=3,label='Saved best: round 40')
ax.set(title='Student loss falls, but completion peaks earlier',xlabel='Student round',ylabel='Selection25 weighted completion (%)');ax.legend(frameon=False)
ax=axs[1,0]
families=['behavior:wrong_side_incidence','behavior:radius_switch','behavior:long_low_braking','go_around','behavior:vertical_chain','behavior:compound_reversal']
yy=np.arange(len(families))
ax.barh(yy-.18,[100*d['outcomes']['families'][f]['teacher_success'] for f in families],height=.34,color='#3577ad',label='Teacher')
ax.barh(yy+.18,[100*d['outcomes']['families'][f]['student_success'] for f in families],height=.34,color='#d47735',label='Student')
ax.set_yticks(yy,[f.replace('behavior:','').replace('_',' ') for f in families]);ax.invert_yaxis();ax.set(xlabel='Paired full100 completion (%)',title='Transfer gap concentrates in wrong-side approaches',xlim=(0,105));ax.legend(frameon=False,loc='lower right')
ax=axs[1,1]
for arm,color in [('teacher','#3577ad'),('student','#d47735')]:
    z=np.load(out/f'recovery-{arm}-021-e0.npz')['rows']
    r=next(r for r in json.loads((out/'recovery-probe.json').read_text()) if r['arm']==arm)
    ax.step(np.r_[z[:,0]/130,r['steps']/130],np.r_[z[:,1],r['passed']],where='post',label=arm.capitalize(),color=color)
    ax.scatter([r['steps']/130],[r['passed']],color=color,s=40)
ax.set(xlabel='Time (s)',ylabel='Accepted gates',title='Diagnostic course 021: persistent retry without progress',ylim=(-.3,8.5));ax.legend(frameon=False)
fig.suptitle('Plant teacher → 3M EKF student: progress, transfer, and recovery',fontsize=17)
fig.savefig(out/'analysis.png',dpi=150)
print(out/'analysis.png')
