from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
summary=json.loads((root/'timing/summary.json').read_text())
trials=json.loads((root/'timing/trial-stats.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
labels=['baseline','packed','combined']
colors=['#8b98a5','#4496c9','#138970']
for ax,key,title,ylabel in zip(axes,['ttfa_p50_ms','rtfx_p50'],['First complete 80 ms frame · lower is better','Generation speed · higher is better'],['TTFA (milliseconds)','Audio seconds / generation second']):
    for j,(variant,color) in enumerate(zip(labels,colors)):
        xs=[i+(j-1)*.25 for i in range(3)]
        vals=[next(r[key] for r in summary if r['text']==t and r['variant']==variant) for t in ['short','release','long']]
        ax.bar(xs,vals,width=.23,color=color,label={'baseline':'Flags off','packed':'Packed linears','combined':'+ Initial convolution'}[variant])
        for x,t,v in zip(xs,['short','release','long'],vals):
            dots=[r[key] for r in trials if r['text']==t and r['variant']==variant]
            ax.scatter([x]*len(dots),dots,s=14,color='black',zorder=3)
            ax.text(x,v+(.12 if key=='ttfa_p50_ms' else 1.4),f'{v:.2f}',ha='center',fontsize=8)
    ax.set_xticks(range(3),['Short','Release','Long'])
    ax.set_ylim(0,6.5 if key=='ttfa_p50_ms' else 96)
    ax.set_title(title,fontsize=11)
    ax.set_ylabel(ylabel)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
    ax.set_axisbelow(True)
fig.suptitle('Focused Phonon PR · Apple M4 Pro · CPU Q8 · 4 tensor threads',fontsize=13)
handles,leglabels=axes[0].get_legend_handles_labels()
fig.legend(handles,leglabels,loc='lower center',bbox_to_anchor=(.5,.035),ncol=3,frameon=False)
fig.text(.5,.015,'Bars: median of 3 trial medians. Dots: individual trials. 20 requests + 5 warmups per process.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.13,1,.94])
fig.savefig(root/'results.png',dpi=180)
fig.savefig(root/'results.svg')

p = root / 'results.svg'
p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines()) + '\n')
