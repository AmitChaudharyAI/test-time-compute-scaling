"""Publication figures from saved aggregate results only, in isolated environment."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
P=Path(__file__).resolve().parents[1]
r=json.loads((P/'results/analysis_summary.json').read_text())
a=r['metrics'][0]; fixed=r['metrics'][1:]; stops=r['stopping_distribution']
BLUE='#0072B2'; RED='#D55E00'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'xtick.direction':'out','ytick.direction':'out',
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':400})
def save(fig,stem):
 for ext in ('pdf','svg','png'): fig.savefig(P/'figures'/f'{stem}.{ext}',bbox_inches='tight',facecolor='white')
 plt.close(fig)
def trade(ax,field,label):
 x=[v[field] for v in fixed]; y=[100*v['accuracy'] for v in fixed]
 err=np.array([[100*(v['accuracy']-v['accuracy_ci_low']) for v in fixed],[100*(v['accuracy_ci_high']-v['accuracy']) for v in fixed]])
 ax.errorbar(x,y,yerr=err,color=BLUE,marker='o',markersize=5,lw=1,capsize=3,label='Fixed-budget SC',zorder=2)
 ax.errorbar(a[field],100*a['accuracy'],yerr=[[100*(a['accuracy']-a['accuracy_ci_low'])],[100*(a['accuracy_ci_high']-a['accuracy'])]],fmt='D',color=RED,markersize=7,capsize=4,label='Frozen adaptive',zorder=3)
 for v in fixed:
  n=int(v['method'].split('fixed_n')[1])
  # Preset annotation placement by budget identity, independent of accuracy.
  offset={1:(-3,-17),2:(-3,9),4:(5,-17),8:(5,9),16:(-28,9)}[n]
  ax.annotate(f'N={n}',(v[field],100*v['accuracy']),xytext=offset,textcoords='offset points',fontsize=9,color=BLUE)
 ax.annotate('Adaptive',(a[field],100*a['accuracy']),xytext=(8,-19),textcoords='offset points',color=RED,fontsize=9)
 ax.set_xlabel(label); ax.set_ylabel('Accuracy (%)'); ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
 ax.grid(axis='y',alpha=.2); ax.set_xlim(left=0,right=max(x)*1.08)
 lower=min(100*v['accuracy_ci_low'] for v in r['metrics'])-3; upper=max(100*v['accuracy_ci_high'] for v in r['metrics'])+3
 ax.set_ylim(max(0,lower),min(100,upper)); ax.legend(frameon=False,loc='lower right',fontsize=9)
for field,label,stem in [('mean_samples','Mean generated samples per question','accuracy_vs_mean_samples'),('mean_output_tokens','Mean output tokens per question','accuracy_vs_output_tokens')]:
 fig,ax=plt.subplots(figsize=(6.4,4.3),layout='constrained'); trade(ax,field,label)
 ax.set_title('Prospective accuracy and compute · MATH-500')
 fig.supxlabel('Error bars: 95% question-level bootstrap intervals; same panel for every method.',fontsize=8)
 save(fig,stem)
fig,ax=plt.subplots(figsize=(7,4.1),layout='constrained')
x=[s['stopping_sample'] for s in stops]; counts=[s['count'] for s in stops]
ax.bar(x,counts,color=BLUE,width=.75)
for t,c in zip(x,counts): ax.annotate(f'{c}\n{100*c/500:.1f}%',(t,c),xytext=(0,4),textcoords='offset points',ha='center',va='bottom',fontsize=7)
ax.set_xticks(x); ax.set_xlabel('Generated samples at STOP'); ax.set_ylabel('Questions (of 500)'); ax.set_ylim(0,max(counts)*1.18)
ax.yaxis.set_major_locator(MaxNLocator(integer=True)); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
ax.set_title('Frozen adaptive stopping distribution')
save(fig,'stopping_distribution')
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
trade(axes[0],'mean_samples','Mean samples per question'); trade(axes[1],'mean_output_tokens','Mean output tokens per question')
axes[0].set_title('(a) Sample cost'); axes[1].set_title('(b) Output-token cost')
fig.suptitle('Prospective adaptive versus fixed-budget self-consistency')
fig.supxlabel('Adaptive cost is counterfactual prefix cost; full experiment generated 8,000 samples.',fontsize=9)
save(fig,'adaptive_vs_fixed_budget_tradeoff')
fig,ax=plt.subplots(figsize=(6.5,3.7),layout='constrained')
for i,c in enumerate(r['paired_comparisons']):
 ax.errorbar(c['accuracy_difference_pp'],i,xerr=[[c['accuracy_difference_pp']-c['ci_low_pp']],[c['ci_high_pp']-c['accuracy_difference_pp']]],fmt='o',color=RED if i==0 else BLUE,capsize=4)
ax.set_yticks(range(3),['vs N=16 (primary)','vs N=8 (exploratory)','vs N=4 (exploratory)']); ax.invert_yaxis(); ax.axvline(0,color='.5',ls='--',lw=1)
ax.set_xlabel('Adaptive minus fixed accuracy (percentage points)'); ax.set_title('Paired 95% percentile bootstrap intervals'); ax.grid(axis='x',alpha=.2)
save(fig,'paired_accuracy_differences')
fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
ax.bar(np.array(x)-.2,[r['historical_separate_panel']['stopping_counts'][str(t)] for t in x],width=.4,color='.6',label='Historical offline')
ax.bar(np.array(x)+.2,counts,width=.4,color=BLUE,label='Fresh prospective')
ax.set_xticks(x); ax.set_xlabel('Samples at STOP'); ax.set_ylabel('Questions'); ax.set_title('Separate historical and prospective stopping distributions'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
save(fig,'historical_vs_fresh_stopping')
fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
for ax,field,label in zip(axes,['conditional_accuracy','mean_output_tokens','mean_margin'],['Conditional accuracy','Mean output tokens','Mean vote margin']):
 ss=[s for s in stops if s[field] is not None]
 ax.plot([s['stopping_sample'] for s in ss],[s[field] for s in ss],'o',color=BLUE,markersize=4)
 ax.set_xlabel('Actual STOP sample'); ax.set_ylabel(label); ax.grid(axis='y',alpha=.2); ax.set_xticks([2,4,8,12,16])
 if field!='mean_output_tokens': ax.set_ylim(0,1.05)
fig.suptitle('Stopping-conditioned outcomes and cost · descriptive')
save(fig,'conditional_stopping_behavior')
(P/'results/figure_metadata.json').write_text(json.dumps({'matplotlib_version':matplotlib.__version__,'rendering_only':True,'formats':['pdf','svg','png'],'png_dpi':400,'intervals':'Preregistered 95% question-level percentile bootstrap; individual-method intervals are not paired-difference intervals.'},indent=2)+'\n')
print('Publication figures rendered: PDF, SVG, 400-dpi PNG')
