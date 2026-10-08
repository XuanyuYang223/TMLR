"""Standalone figures from locked diagnostic and confirmation outputs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('results/readout_null_confirmation')


def run():
    diag=json.loads(Path('results/downstream_harm_diagnostic_v2/results.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,domain in zip(axs,['permworld','matrix']):
        for direction,color,label in [('actual_null','tab:red','Actual error direction'),('random_null','tab:blue','Random null direction')]:
            rows=[r for r in diag['curves'] if (r['domain'],r['condition'],r['split'],r['direction'])==(domain,'both_correct','collisions',direction)]
            xx=sorted({r['amplitude'] for r in rows if r['amplitude']<=.25})
            ys=np.array([[np.mean([r['accuracy'] for r in rows if r['source']==s and r['amplitude']==x]) for x in xx] for s in range(3)])*100
            ax.plot(xx,ys.mean(0),'o-',color=color,label=label)
            ax.fill_between(xx,ys.min(0),ys.max(0),color=color,alpha=.15)
        ax.set_title(domain);ax.set_ylim(0,102);ax.set_xlabel('Perturbation norm / intermediate hidden RMS');ax.set_ylabel('Compound accuracy (%)');ax.legend(fontsize=8)
    fig.suptitle('Correct-pairing old models; shading is range of three source means',fontsize=10)
    fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(ROOT/f'null_perturbation_zoom.{ext}',dpi=160)
    plt.close(fig)
    files=[ROOT/d/'independent_results.json' for d in ['permworld','matrix']]
    if not all(f.exists() for f in files):return
    fig,axs=plt.subplots(1,2,figsize=(10,4));conditions=['full_correct','full_wrong','null_correct','null_wrong']
    labels=['Full/correct','Full/wrong','Null/correct','Null/wrong']
    for ax,file in zip(axs,files):
        r=json.loads(file.read_text());ys=[]
        for c in conditions:
            ys.append([100*s['accuracy'] for s in r['records'] if s['condition']==c and s['split']=='collisions'])
        ys=np.asarray(ys);ax.bar(np.arange(4),ys.mean(1),color=['#377eb8','#999999','#4daf4a','#bbbbbb'])
        for j in range(3):ax.scatter(np.arange(4)+.07*(j-1),ys[:,j],s=18,color='black',zorder=3)
        ax.set_xticks(np.arange(4),labels,rotation=25,ha='right');ax.set_ylabel('Collision compound accuracy (%)');ax.set_title(r['domain']);ax.set_ylim(0,max(ys.max()+3,15))
        cc=r['contrasts']['accuracy']['interaction_null_minus_full'];lo,hi=cc['source_and_whole_pair_interval_pp']
        ax.text(.98,.97,f'Interaction {cc["mean_pp"]:+.2f} pp\n95% interval [{lo:+.2f}, {hi:+.2f}]',transform=ax.transAxes,va='top',ha='right',fontsize=8)
    fig.suptitle('New-source confirmation; dots are three independently trained source models',fontsize=10);fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(ROOT/f'direction_confirmation.{ext}',dpi=160)

if __name__=='__main__':run()
