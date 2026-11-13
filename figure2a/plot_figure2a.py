"""Reconcile and plot Fig. 2(a) from the saved Fig. 2(b) main analysis input."""
from pathlib import Path
import argparse
import csv
import json
import duckdb
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True, help="Saved original-post ordered shards")
parser.add_argument("--plan", type=Path, required=True, help="Existing Figure 2(b) model plan")
parser.add_argument("--reference", type=Path, required=True, help="Earlier Figure 2(a) reconciliation CSV")
args = parser.parse_args()
out = Path(__file__).resolve().parent
support = json.loads(args.plan.read_text(encoding="utf-8-sig"))["initial_supported_bins"]["main"]
assert support == list(range(-1, 21))
files = sorted(args.input.glob("???.parquet"))
assert len(files) == 128
resolution = 12
db = duckdb.connect()
db.execute("SET threads=2; SET memory_limit='240MB'; SET enable_external_file_cache=false")
cells = {}
totals = np.zeros(5, dtype=np.int64)
for path in files:
    result = db.execute("""
        WITH sample AS (
          SELECT h,z,CASE WHEN h>0 AND z>0 THEN floor(log2(h)*12)::INT ELSE -1 END bx,
                     CASE WHEN h>0 AND z>0 THEN floor(log2(z)*12)::INT ELSE -1 END y_bin
          FROM read_parquet(?) WHERE h < ?
        )
        SELECT bx,y_bin,count(*),count(*) FILTER(WHERE z<h),count(*) FILTER(WHERE z=h),
               count(*) FILTER(WHERE z>h),count(*) FILTER(WHERE bx>=0)
        FROM sample GROUP BY bx,y_bin
    """, [str(path), 2**(max(support)+1)]).fetchall()
    for bx, by, n, below, equal, above, positive in result:
        totals += (n, below, equal, above, positive)
        if positive:
            cells[bx, by] = cells.get((bx, by), 0) + positive
db.close()
with args.reference.open(encoding="utf-8-sig") as stream:
    reference = next(row for row in csv.DictReader(stream) if row["scope"] == "main_heat_support")
expected = tuple(int(reference[key]) for key in ("n", "below", "equal", "above"))
n, below, equal, above, positive = map(int, totals)
assert (n, below, equal, above) == expected == (315992251, 311390972, 4265161, 336118), (totals, expected)
assert below + equal + above == n
assert sum(cells.values()) == positive
zero = n-positive

def write_csv(name, header, rows):
    with (out/name).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)

write_csv("reconciliation.csv", ["category", "count", "denominator", "percent_of_full_sample"],
          [(label, count, n, 100*count/n) for label, count in
           [("full_main_sample",n),("repost_likes_below_original",below),
            ("repost_likes_equal_original",equal),("repost_likes_above_original",above),
            ("both_positive_plotted",positive),("either_zero_not_plotted",zero)]])
write_csv("density_cells.csv", ["original_bin_lower_inclusive","original_bin_upper_exclusive",
          "repost_bin_lower_inclusive","repost_bin_upper_exclusive","repost_count"],
          [(2**(x/resolution),2**((x+1)/resolution),2**(y/resolution),2**((y+1)/resolution),count)
           for (x,y),count in sorted(cells.items())])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.lines import Line2D

x_upper = 2**(max(support)+1)
y_octaves = (max(y for x,y in cells)+resolution)//resolution
y_upper = 2**y_octaves
x_edges = np.exp2(np.arange((max(support)+1)*resolution+1)/resolution)
y_edges = np.exp2(np.arange(y_octaves*resolution+1)/resolution)
grid = np.zeros((len(y_edges)-1, len(x_edges)-1), dtype=np.int64)
for (x,y),count in cells.items():
    grid[y,x] = count
fig, ax = plt.subplots(figsize=(4.75,4.15), layout="constrained")
mesh = ax.pcolormesh(x_edges,y_edges,np.ma.masked_equal(grid,0),cmap="viridis",
                     norm=LogNorm(vmin=1,vmax=grid.max()),shading="flat")
ax.plot([1,x_upper],[1,x_upper],color="white",lw=2.25,ls="--")
ax.plot([1,x_upper],[1,x_upper],color="#343434",lw=.9,ls="--")
ax.set(xscale="log",yscale="log",xlim=(1,x_upper),ylim=(1,y_upper),
       xlabel="Original post cumulative likes",ylabel="Repost cumulative likes")
ax.set_aspect("equal", adjustable="box")
for axis,upper in ((ax.xaxis,x_upper),(ax.yaxis,y_upper)):
    ticks=[10**k for k in range(0,int(np.log10(upper))+1,2)]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v,p: rf"$10^{{{round(np.log10(v))}}}$"))
ax.legend(handles=[Line2D([],[],color="#343434",lw=.9,ls="--",label=r"$y=x$")],
          loc="upper left",frameon=False,fontsize=9)
ax.tick_params(labelsize=9)
colorbar = fig.colorbar(mesh,ax=ax,pad=.03,shrink=.88)
colorbar.set_label("Reposts per bin", fontsize=9)
colorbar.ax.tick_params(labelsize=8)
fig.savefig(out/"figure2a.png",dpi=600,facecolor="white")
plt.close(fig)

caption = (
    "**Figure 2(a). Cumulative likes on original posts and their reposts.** "
    "Both counts come from the same crawl record. We use the same April 2023 main analysis sample as "
    "Fig. 2(b), with one last observation per repost within the crawl window and the same timing and "
    "popularity support filters. "
    f"Among all {n:,} reposts, including records with zero likes, "
    f"{100*below/n:.4f}% had fewer likes than the original post, {100*equal/n:.4f}% had the same "
    f"number, and {100*above/n:.4f}% had more. "
    f"The log–log panel displays all {positive:,} records with positive like counts on both posts. "
    f"The remaining {zero:,} records have zero likes on at least one side and are omitted from the panel. "
    "Color indicates the number of reposts in each two-dimensional bin, with 12 bins per doubling "
    "on each axis and a logarithmic color scale. The dashed line indicates equal like counts."
)
(out/"caption_en.md").write_text(caption+"\n",encoding="utf-8")
print(f"Verified full sample {n:,}: {below:,} / {equal:,} / {above:,}; plotted {positive:,}; zero {zero:,}; {len(cells):,} occupied cells")

