"""Recalculate Figure 1 from the exact April Figure 2 main-analysis population."""
import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import duckdb


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_frequency(ordered, upper):
    files = sorted(ordered.glob("???.parquet"))
    assert len(files) == 128 and [p.stem for p in files] == [f"{i:03}" for i in range(128)]
    scan = str(ordered / "???.parquet").replace("\\", "/").replace("'", "''")
    with duckdb.connect() as con:
        con.execute("SET threads=1; SET memory_limit='96MB'; SET enable_external_file_cache=false")
        return con.execute(f"""
            SELECT z, count(*) n,
                count(*) FILTER (WHERE bad_pair OR age IS NULL OR age<0 OR
                    delay IS NULL OR delay<0 OR oid IS NULL OR z IS NULL OR z<0) bad
            FROM read_parquet('{scan}', hive_partitioning=false)
            WHERE h>=0 AND h<?
            GROUP BY z ORDER BY z
        """, [upper]).fetchall()


def plot_bins(rows, sample, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(12.2, 5.1), layout="constrained")
    fig.suptitle("Figure 1 recalculation | Likes received by April 2023 reposts",
                 fontsize=15, fontweight="bold")
    for ax, key, title, ylabel, color in (
        (axes[0], "conditional_probability_mass", "A  Probability mass per bin",
         "Conditional probability mass", "#166887"),
        (axes[1], "conditional_probability_density", "B  Density per like",
         "Conditional probability density (per like)", "#8c492d"),
    ):
        x, y = [], []
        for row in rows:
            if row["count"]:
                lo, hi = row["lower_inclusive"], row["upper_exclusive"]
                x.append(math.sqrt(lo*hi))
                y.append(row[key])
                ax.hlines(row[key], lo, hi, color=color, alpha=.63, linewidth=1.45)
        ax.scatter(x, y, color=color, s=18, zorder=3)
        ax.set(xscale="log", yscale="log", xlim=(.91,rows[-1]["upper_exclusive"]*1.12),
               xlabel="Likes received by reposts (zhan)", ylabel=ylabel)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
        ax.grid(True, which="major", color="#76838a", alpha=.23)
    fig.supxlabel(f"Positive likes only in panels: N+ = {sample['positive_reposts']:,}. "
                  f"Zero likes: {sample['zero_reposts']:,} / {sample['valid_reposts']:,} "
                  f"({sample['zero_share']:.2%}).", fontsize=9.5)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def run(ordered, model_dir, out):
    out.mkdir(parents=True, exist_ok=True)
    names = ("sample_summary.csv", "positive_like_bins.csv", "tail_examples.csv",
             "figure1_april_distribution.png", "checks.json", "README_zh.md")
    if any((out / name).exists() for name in names):
        raise FileExistsError("Use a new output directory for a repeat run.")
    plan = json.loads((model_dir / "model_plan.json").read_text(encoding="utf-8"))
    support = plan["final_supported_bins"]["main"]
    assert support == list(range(-1, 21)), support
    upper = 2**(support[-1]+1)
    with (model_dir / "model_totals.csv").open(encoding="utf-8-sig", newline="") as stream:
        ref = next(row for row in csv.DictReader(stream) if row["model"] == "main")
    assert int(ref["reposts"]) == 315_992_251
    freq = source_frequency(ordered, upper)
    assert freq and freq[0][0] == 0 and all(z>=0 and bad==0 for z,n,bad in freq)
    total = sum(n for z,n,bad in freq)
    zero = freq[0][1]
    likes_sum = sum(z*n for z,n,bad in freq)
    assert (total,zero,likes_sum) == tuple(int(ref[k]) for k in ("reposts","zero_reposts","likes_sum"))
    positive = total-zero
    counts = [0] * int(freq[-1][0]).bit_length()
    for z,n,_ in freq[1:]:
        counts[int(z).bit_length()-1] += n
    rows = []
    for k,n in enumerate(counts):
        lo, hi = 2**k, 2**(k+1)
        width = hi-lo
        rows.append(dict(k=k, lower_inclusive=lo, upper_exclusive=hi, width=width,
                         count=n, positive_denominator=positive,
                         conditional_probability_mass=n/positive,
                         conditional_probability_density=n/(positive*width)))
    mass = math.fsum(r["conditional_probability_mass"] for r in rows)
    area = math.fsum(r["conditional_probability_density"]*r["width"] for r in rows)
    assert sum(counts)==positive and math.isclose(mass,1,rel_tol=0,abs_tol=1e-15)
    assert math.isclose(area,1,rel_tol=0,abs_tol=1e-15)
    sample = dict(valid_reposts=total, zero_reposts=zero, zero_share=zero/total,
                  positive_reposts=positive, positive_share=positive/total,
                  likes_sum=likes_sum, max_repost_likes=freq[-1][0],
                  original_heat_lower_inclusive=0, original_heat_upper_exclusive=upper)
    tails = [row for row in rows if row["count"]][-3:]
    assert len(tails)==3
    write_csv(out/"sample_summary.csv",[sample])
    write_csv(out/"positive_like_bins.csv",rows)
    write_csv(out/"tail_examples.csv",[{**r,
              "mass_calculation":f"{r['count']} / {positive}",
              "density_calculation":f"{r['count']} / ({positive} * {r['width']})"} for r in tails])
    (out/"checks.json").write_text(json.dumps(dict(total_matches_figure2=True,zero_matches_figure2=True,
               likes_sum_matches_figure2=True, positive_bin_count_sum=sum(counts),
               mass_sum=mass,density_times_bin_width_sum=area, source_shards=128), indent=2),encoding="utf-8")
    plot_bins(rows,sample,out/"figure1_april_distribution.png")
    print(json.dumps(sample,indent=2),flush=True)
    print("Tail examples:",tails,flush=True)
    print("Validation: count, mass and density integral reconcile.",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ordered",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.ordered,a.model_dir,a.out)

