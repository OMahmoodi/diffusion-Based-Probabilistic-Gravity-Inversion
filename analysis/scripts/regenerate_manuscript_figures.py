"""Regenerate manuscript result figures from frozen benchmark outputs."""
from pathlib import Path
import io, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'analysis'/'data'; ART=ROOT/'analysis'/'artifacts'/'04_conditional_results.zip'; OUT=ROOT/'figures'
METHODS=['SimPEG L2','Unconditional','Oracle conditional','Equal mixture']
LABEL={'SimPEG L2':'SimPEG L2','Unconditional':'Unconditional','Oracle conditional':'Known prior\n(oracle)','Equal mixture':'Scenario\nensemble'}
GEO=['blobs','horizontal_layers','dipping_layers','faulted_layers','dykes','intrusions']
metrics=pd.read_csv(DATA/'all_case_method_metrics.csv'); id_geo=pd.read_csv(DATA/'complete_id_rmse_by_geology.csv'); ood_geo=pd.read_csv(DATA/'complete_ood_rmse_by_geology.csv'); id_case=pd.read_csv(DATA/'complete_id_complete_case_metrics.csv'); ood_case=pd.read_csv(DATA/'complete_ood_complete_case_metrics.csv')
with zipfile.ZipFile(ART) as z:
 id_arrays=np.load(io.BytesIO(z.read('id_arrays.npz'))); ood_arrays=np.load(io.BytesIO(z.read('ood_arrays.npz')))
plt.rcParams.update({'font.size':16,'axes.titlesize':19,'axes.labelsize':18,'xtick.labelsize':14,'ytick.labelsize':15,'legend.fontsize':14})
def overall(metric,ylabel,phrase,fn):
 fig,axs=plt.subplots(1,2,figsize=(18,6.5),dpi=170,constrained_layout=True)
 for ax,split in zip(axs,['ID','OOD']):
  data=[metrics[(metrics.split==split)&(metrics.method==m)][metric].values for m in METHODS]; bp=ax.boxplot(data,patch_artist=True,widths=.45,showfliers=False)
  for p in bp['boxes']: p.set(facecolor='#f2f2f2',linewidth=1.8)
  for k in ['whiskers','caps','medians']:
   for x in bp[k]: x.set(linewidth=1.6)
  ax.scatter(range(1,5),[np.mean(x) for x in data],marker='^',s=120,zorder=3); ax.set_xticklabels([LABEL[m] for m in METHODS],rotation=18); ax.set_ylabel(ylabel); ax.set_title(f'{split}: {phrase}',pad=10); ax.grid(axis='y',alpha=.28)
  if metric=='rmse': ax.set_ylim(bottom=0)
 fig.savefig(OUT/fn,bbox_inches='tight'); plt.close(fig)
overall('rmse','Pixelwise RMSE','Pixelwise RMSE (lower is better)','figure4_overall_rmse.png'); overall('ssim','Fixed-range SSIM','Fixed-range SSIM (higher is better)','figure4b_overall_ssim.png')
def geofig(df,title,fn):
 fig,ax=plt.subplots(figsize=(18,7),dpi=170,constrained_layout=True); x=np.arange(len(GEO)); w=.19
 for i,m in enumerate(METHODS):
  s=df[df.method==m].set_index('geology_type').loc[GEO]; ax.bar(x+(i-1.5)*w,s.mean_rmse,w,yerr=s.sem_rmse,capsize=4,label=LABEL[m].replace('\n',' '))
 ax.set_xticks(x); ax.set_xticklabels([g.replace('_',' ') for g in GEO],rotation=22,ha='right'); ax.set_ylabel('Mean model RMSE'); ax.set_title(title,pad=10); ax.legend(ncol=2,loc='upper right'); ax.grid(axis='y',alpha=.28); ax.set_ylim(bottom=0); fig.savefig(OUT/fn,bbox_inches='tight'); plt.close(fig)
geofig(id_geo,'In-distribution: model RMSE by geology family','figure5a_id_rmse_by_geology.png'); geofig(ood_geo,'Out-of-distribution: model RMSE by geology family','figure5b_ood_rmse_by_geology.png')
def cov(df,title,fn):
 pref=['unconditional','oracle','mixture']; names=['Unconditional','Known prior (oracle)','Scenario ensemble']; a=[100*df[f'{p}_coverage_68'].mean() for p in pref]; b=[100*df[f'{p}_coverage_95'].mean() for p in pref]; x=np.arange(3); w=.36
 fig,ax=plt.subplots(figsize=(14,7.5),dpi=170,constrained_layout=True); b1=ax.bar(x-w/2,a,w,label='Nominal 68% interval'); b2=ax.bar(x+w/2,b,w,label='Nominal 95% interval'); l1=ax.axhline(68,ls='--',lw=1.8,label='Target 68%'); l2=ax.axhline(95,ls=':',lw=2.2,label='Target 95%'); ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0,105); ax.set_ylabel('Empirical pixel coverage (%)'); ax.set_title(title,pad=10)
 for bars in [b1,b2]:
  for bar in bars: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.2,f'{bar.get_height():.1f}%',ha='center',fontsize=14)
 ax.legend([l1,l2,b1,b2],['Target 68%','Target 95%','Nominal 68% interval','Nominal 95% interval'],loc='upper right'); ax.grid(axis='y',alpha=.28); fig.savefig(OUT/fn,bbox_inches='tight'); plt.close(fig)
cov(id_case,'In-distribution: posterior interval calibration','figure8a_id_coverage_calibration.png'); cov(ood_case,'Out-of-distribution: posterior interval calibration','figure8b_ood_coverage_calibration.png')
def grid(cases,fn):
 fig,axs=plt.subplots(len(cases),7,figsize=(22,8.8),dpi=180); plt.subplots_adjust(wspace=.06,hspace=.18,left=.06,right=.995,top=.93,bottom=.05); mx=max(id_arrays['oracle_std'].max(),id_arrays['mixture_std'].max())
 for i,case in enumerate(cases):
  arrs=[id_arrays['truth'][case],id_arrays['l2'][case],id_arrays['unconditional_mean'][case],id_arrays['oracle_mean'][case],id_arrays['mixture_mean'][case],id_arrays['oracle_std'][case],id_arrays['mixture_std'][case]]; s=metrics[(metrics.split=='ID')&(metrics.case==case)].set_index('method'); titles=['Truth',f"SimPEG\nRMSE={s.loc['SimPEG L2','rmse']:.3f}",f"Unconditional\nRMSE={s.loc['Unconditional','rmse']:.3f}",f"Known prior (oracle)\nRMSE={s.loc['Oracle conditional','rmse']:.3f}",f"Scenario ensemble\nRMSE={s.loc['Equal mixture','rmse']:.3f}",'Known prior std','Scenario ensemble std']
  for j,a in enumerate(arrs):
   ax=axs[i,j]; ax.imshow(a,cmap='RdBu_r' if j<5 else 'magma',vmin=-1 if j<5 else 0,vmax=1 if j<5 else mx,interpolation='nearest'); ax.set_title(titles[j],fontsize=12,pad=6); ax.set_xticks([]); ax.set_yticks([])
  axs[i,0].set_ylabel(f"ID case {case}\n{s.geology_type.iloc[0].replace('_',' ')}",fontsize=12)
 fig.savefig(OUT/fn,bbox_inches='tight'); plt.close(fig)
grid([20,40,39],'figure6_selected_known_prior_cases.png'); grid([47,18,42],'figure7_selected_scenario_ensemble_cases.png')
