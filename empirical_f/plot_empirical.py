from pathlib import Path
import csv,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parent
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def relation(folder,names,titles,filename,ylabel):
 fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 for col,(name,title) in enumerate(zip(names,titles)):
  curve=folder/(name+'_curve.csv')
  if not curve.exists():
   for ax in axs[:,col]:ax.text(.5,.5,'Insufficient support',ha='center',transform=ax.transAxes)
   continue
  rr=read(curve);xx=[int(r['heat_bin']) for r in rr];cap=max(xx)
  mean=[float(r['adjusted_mean']) for r in rr]
  axs[0,col].plot(xx,mean,'o-',ms=3,c='#1b7195',label='Adjusted mean')
  axs[0,col].fill_between(xx,[float(r['ci95_low']) for r in rr],[float(r['ci95_high']) for r in rr],alpha=.18,color='#1b7195',lw=0)
  raw=[r for r in read(folder/(name+'_support.csv')) if int(r['heat_bin']) in xx]
  axs[0,col].plot([int(r['heat_bin']) for r in raw],[float(r['mean_outcome']) for r in raw],':',c='#a25b2f',label='Raw mean')
  axs[1,col].plot([int(r['heat_bin']) for r in raw],[float(r['positive_probability']) for r in raw],'o-',ms=3,c='#a25b2f')
  axs[0,col].axhline(0,c='#777777',lw=.6);axs[0,col].set_title(title,loc='left');axs[0,col].set_ylabel(ylabel);axs[1,col].set_ylabel('At least one: observed proportion')
  axs[1,col].yaxis.set_major_formatter(PercentFormatter(1))
  ticks=([-1] if -1 in xx else [])+[k for k in range(0,cap+1,3) if k in xx and (k!=0 or -1 not in xx)]
  for ax in axs[:,col]:
   ax.set_xticks(ticks,['0' if k==-1 else '2^'+str(k) for k in ticks]);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
  axs[1,col].set_xlabel('Original popularity · bin lower bound')
 axs[0,0].legend(frameon=False,loc='upper left')
 fig.savefig(folder/filename,dpi=180);plt.close(fig)
def coverage():
 folder=ROOT/'02_age24';rr=read(folder/'coverage_by_last_observation_heat.csv')
 fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for win,color in [('25','#1b7195'),('27','#b85a30')]:
  rows=[r for r in rr if r['window_upper_hours']==win]
  ax[0].plot([int(r['last_observation_heat_bin']) for r in rows],[float(r['coverage']) for r in rows],'o-',ms=3,label=f'[24,{win}) h',c=color)
 ax[0].set(xlabel='Month-last original-like bin k (-1 = zero)',ylabel='Coverage of all April unique reposts')
 ax[0].yaxis.set_major_formatter(PercentFormatter(1));ax[0].legend(frameon=False)
 hist=read(folder/'actual_age_histogram.csv');r=[r for r in hist if r['window_upper_hours']=='25']
 ages=[int(x['age_s'])/3600 for x in r];weights=[int(x['reposts']) for x in r]
 ax[1].hist(ages,bins=60,range=(24,25),weights=weights,color='#1b7195')
 ax[1].set(xlabel='Actual observation age (hours)',ylabel='Selected reposts',title='Primary [24,25) h window')
 for a in ax:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.2)
 fig.savefig(folder/'age24_coverage.png',dpi=180);plt.close(fig)
if sys.argv[1]=='counts':
 relation(ROOT/'03_repost_counts',['reposts_vs_h','reposts_vs_h2'],['Original cumulative likes','Original cumulative reposts'],'repost_count_relations.png','Cumulative reposts per reposted post')
else:
 coverage()
 relation(ROOT/'02_age24',['age_25_h','age_25_hs'],['Heat at selected capture','Most recent pre-repost heat'],'age24_relations.png','Mean likes near 24 hours')
