import csv, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
base=Path(__file__).resolve().parent
with (base/"positive_like_bins.csv").open(encoding="utf-8-sig") as f:
    bins=list(csv.DictReader(f))
with (base/"sample_summary.csv").open(encoding="utf-8-sig") as f:
    sample=next(csv.DictReader(f))
n=int(sample["positive_reposts"])
assert sum(int(row["count"]) for row in bins) == n
for row in bins:
    count, width = (int(row[key]) for key in ("count", "width"))
    assert math.isclose(float(row["conditional_probability_mass"]), count / n)
    assert math.isclose(float(row["conditional_probability_density"]), count / (n * width))
fig,axes=plt.subplots(1,2,figsize=(7.5,3.05),layout="constrained")
for ax,key,label,title in zip(axes,
    ["conditional_probability_mass","conditional_probability_density"],
    ["Probability mass per bin","Probability density per like"],["(a) Mass","(b) Density"]):
    for row in bins:
        lo,hi=[int(row[k]) for k in ("lower_inclusive","upper_exclusive")]
        y=float(row[key])
        if y>0:
            ax.hlines(y,lo,hi,color="#176784",linewidth=1)
            ax.plot((lo*hi)**.5,y,"o",color="#176784",markersize=3)
    ax.set(xscale="log",yscale="log",xlabel="Cumulative likes per repost",ylabel=label,title=title)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.17)
fig.savefig(base/"S15_figure1_april_distribution.png",dpi=600,facecolor="white")
