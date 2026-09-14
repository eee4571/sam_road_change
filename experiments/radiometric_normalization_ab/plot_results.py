"""Render measured A/B metrics (no new model inference or parameter tuning)."""
import os
import json
import numpy as np
import pandas as pd
from run_experiment import ROOT, environment
os.environ.update(environment())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

m=json.loads((ROOT/'metrics.json').read_text(encoding='utf-8'))
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
colors=['#456b9b','#c77336']
types=['added','removed','widened','narrowed']
for ax,metric,title in [(axes[0,0],'counts_in_extent','Fast2 objects'),(axes[0,1],'no_intersection','Objects with no GT intersection')]:
    for j,arm in enumerate(('A','B')):
        vals=[m['arms'][arm]['offline_gt'][metric][k] for k in types]
        bars=ax.bar(np.arange(4)+(j-.5)*.36,vals,.36,label=arm,color=colors[j])
        ax.bar_label(bars,padding=2,fontsize=8)
    ax.set_xticks(range(4),types);ax.set_title(title);ax.legend()
names=['Axis overlap\nwithin 3 m','Final surface\nIoU','Raw MoLRA\nIoU']
for j,arm in enumerate(('A','B')):
    r=m['arms'][arm]
    vals=[r['centerlines']['symmetric_overlap_3m'],r['surface']['iou'],r['molra_surface']['raw']['iou']]
    bars=axes[1,0].bar(np.arange(3)+(j-.5)*.36,vals,.36,label=arm,color=colors[j])
    axes[1,0].bar_label(bars,fmt='%.3f',padding=2,fontsize=8)
axes[1,0].set_xticks(range(3),names);axes[1,0].set_ylim(0,1);axes[1,0].set_title('Cross-period comparability')
table=pd.read_csv(ROOT/'evaluation/fixed_station_width_differences.csv')
for arm,color in zip(('A','B'),colors):
    values=np.sort(np.abs(table[arm+'_delta_m']))
    axes[1,1].plot(values,np.arange(1,len(values)+1)/max(len(values),1),label=arm,color=color)
axes[1,1].axvline(2,color='#888888',linestyle='--',linewidth=.8)
axes[1,1].set_xlim(0,15);axes[1,1].set_xlabel('Absolute width difference (m)')
axes[1,1].set_ylabel('Fraction of fixed matched stations');axes[1,1].set_title('Paired width difference CDF');axes[1,1].legend()
for ax in axes.flat:
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('20250118 to 20260203 | A: raw; B: conservative radiometric matching',fontsize=13)
fig.savefig(ROOT/'diagnostics/ab_metrics.png',dpi=160)
fig.savefig(ROOT/'diagnostics/ab_metrics.pdf')
print('plots complete')
