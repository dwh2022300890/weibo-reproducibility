from pathlib import Path
from collections import defaultdict
import csv,sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parent
def read(name):
    with (out/name).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
def save(name,rows):
    with (out/name).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def estimate(values):
    v=np.asarray(values,dtype=float)
    mean=v.mean()
    half=1.95996398454*v.std(ddof=1)/np.sqrt(len(v))
    return dict(estimate=mean,ci95_low=mean-half,ci95_high=mean+half)
phase=sys.argv[1]
trials=read(phase+"_trials.csv")
groups=defaultdict(dict)
metrics=("m_started","no","nr_total","nr_per_planned","nr_per_started","active_original","active_repost")
for row in trials:
    k=(float(row["beta_o"]),float(row["beta_r"]),row["pool"],row["rule"])
    rep=int(row["replicate"])
    assert rep not in groups[k]
    assert int(row["m_started"])==int(row["m_planned"])
    groups[k][rep]=row
points=[]
for (bo,br,pool,rule),records in sorted(groups.items()):
    assert set(records)==set(range(100))
    p=dict(beta_o=bo,beta_r=br,pool=pool,rule=rule,repeats=len(records),m_planned=records[0]["m_planned"])
    for metric in metrics:
        for label,value in estimate([float(x[metric]) for x in records.values()]).items():
            p[metric+"_"+label]=value
    points.append(p)
save(phase+"_points.csv",points)
pairs=[]
for bo,br in sorted({key[:2] for key in groups}):
    if phase=="fixed_rate":
        comparisons=[
            ("M_in_independent",("independent","increasing"),("independent","fixed")),
            ("M_in_shared",("shared","increasing"),("shared","fixed")),
            ("pool_fixed",("shared","fixed"),("independent","fixed")),
            ("pool_increasing",("shared","increasing"),("independent","increasing"))]
    else:
        comparisons=[("M_in_shared",("shared","increasing"),("shared","fixed"))]
    for name,left,right in comparisons:
        a,b=groups[bo,br,*left],groups[bo,br,*right]
        assert all(a[i]["rng_seed"]==b[i]["rng_seed"] and a[i]["original_seed"]==b[i]["original_seed"] for i in a)
        for metric in ("nr_total","nr_per_planned","nr_per_started"):
            diff=[float(a[i][metric])-float(b[i][metric]) for i in range(100)]
            pairs.append(dict(beta_o=bo,beta_r=br,contrast=name,metric=metric,paired_repeats=100,**estimate(diff)))
save(phase+"_paired.csv",pairs)
if phase=="fixed_rate":
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout="constrained")
    for col,br in enumerate((.15,.3,.5)):
        for pool,color in (("independent","#1b7195"),("shared","#b65b29")):
            for rule,ls in (("fixed","--"),("increasing","-")):
                rr=sorted([x for x in points if x["beta_r"]==br and x["pool"]==pool and x["rule"]==rule],key=lambda x:x["beta_o"])
                x=[r["beta_o"] for r in rr]
                for i,m in enumerate(("nr_per_started","nr_total")):
                    ax=axes[i,col]
                    ax.plot(x,[r[m+"_estimate"] for r in rr],color=color,ls=ls,label=pool.title()+" / "+rule)
                    ax.fill_between(x,[r[m+"_ci95_low"] for r in rr],[r[m+"_ci95_high"] for r in rr],color=color,alpha=.10,lw=0)
                    ax.axvspan(.275,.675,color="0.96",zorder=-5)
        axes[0,col].set_title(r"$\beta_R$ = "+str(br))
        axes[1,col].set_xlabel(r"Original transmission probability $\beta_O$")
    axes[0,0].set_ylabel("Cumulative nodes per activated source")
    axes[1,0].set_ylabel("Total cumulative repost-layer nodes")
    axes[1,0].legend(frameon=False,fontsize=8)
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.2)
    fig.savefig(out/"fixed_rate_comparison.png",dpi=300)
else:
    # Resample replicate IDs jointly across all scan points and both M rules.
    rng=np.random.default_rng(2026111105)
    weights=rng.multinomial(100,[.01]*100,size=4000)
    def pooled(rule,lo,hi):
        rr=[r for r in trials if r["rule"]==rule and lo<=int(r["no"])<hi]
        if not rr:
            return dict(trials=0,replicates=0,mean="",ci95_low="",ci95_high="",bootstrap_draws=0),None
        ids=np.array([int(r["replicate"]) for r in rr])
        y=np.array([float(r["nr_per_started"]) for r in rr])
        count=np.bincount(ids,minlength=100)
        total=np.bincount(ids,weights=y,minlength=100)
        denom=weights@count
        boot=np.divide(weights@total,denom,out=np.full(len(weights),np.nan),where=denom>0)
        low,high=np.nanquantile(boot,[.025,.975])
        return dict(trials=len(rr),replicates=len(set(ids)),mean=y.mean(),ci95_low=low,ci95_high=high,
                    bootstrap_draws=int(np.isfinite(boot).sum())),boot
    bins=[]
    bands=[]
    shape=[]
    for rule in ("fixed","increasing"):
        for lo in range(0,901,20):
            row,_=pooled(rule,lo,lo+20)
            bins.append(dict(rule=rule,no_lower=lo,no_upper_exclusive=lo+20,**row))
        stats=[]
        for band in read("locked_NO_bands.csv"):
            lo,hi=int(band["lower"]),int(band["upper_exclusive"])
            row,boot=pooled(rule,lo,hi)
            bands.append(dict(rule=rule,**band,**row))
            stats.append((row,boot))
        for i,label in enumerate(("middle_minus_low","high_minus_middle")):
            a,b=stats[i:i+2]
            delta=b[1]-a[1]
            lo,hi=np.nanquantile(delta,[.025,.975])
            shape.append(dict(rule=rule,contrast=label,estimate=b[0]["mean"]-a[0]["mean"],
                              ci95_low=lo,ci95_high=hi,bootstrap_draws=int(np.isfinite(delta).sum())))
    for rule in ("fixed","increasing"):
        assert sum(r["trials"] for r in bins if r["rule"]==rule)==2100
    save("common_rate_NO_bins.csv",bins)
    save("common_rate_NO_bands.csv",bands)
    save("common_rate_shape.csv",shape)
    fig,ax=plt.subplots(figsize=(8,4.7),layout="constrained")
    for rule,color in (("fixed","#1b7195"),("increasing","#b65b29")):
        rr=[r for r in bins if r["rule"]==rule]
        x=[r["no_lower"]+10 for r in rr]
        y=[float(r["mean"]) if r["trials"] else np.nan for r in rr]
        lo=[float(r["ci95_low"]) if r["trials"] else np.nan for r in rr]
        hi=[float(r["ci95_high"]) if r["trials"] else np.nan for r in rr]
        ax.plot(x,y,"o-",ms=3,color=color,label="Fixed M = 3" if rule=="fixed" else "Increasing M = 1, 3, 5")
        ax.fill_between(x,lo,hi,color=color,alpha=.15,lw=0)
    ax.set(xlabel=r"Cumulative original-source nodes $N_O$ (20-node bins)",
           ylabel=r"Mean cumulative nodes per activated repost source $N_R$")
    ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.2)
    fig.savefig(out/"common_rate_NO_NR.png",dpi=300)
print("Verified summaries:",len(points),len(pairs),flush=True)
