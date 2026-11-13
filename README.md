# Weibo analysis reproduction files

Companion files for *Nonlinear reposting benefit and attention competition on Weibo* and the Supplementary Materials. See FILE_LIST.csv for the complete inventory.

## Revised Figure 1 and Supplementary Figure S2

With Python 3.12 and Matplotlib installed, run from the package root:

```sh
python figure1/plot_figure1.py
```

The script reads positive_like_bins.csv and sample_summary.csv and writes S15_figure1_april_distribution.png at 600 dpi. The output retains its legacy filename and corresponds to Supplementary Figure S2 in the current manuscript. For a positive-like bin with count n and width w, the two panels show n / N_positive and n / (N_positive × w), respectively. The 14,716,932 positive-like reposts are the denominator for both panels. The 301,275,319 zero-like reposts are reported separately out of 315,992,251 reposts. No power-law fit is performed.

figure1/recalculate.py is the upstream calculation. Running it requires DuckDB, Matplotlib, 128 ordered April Parquet shards, and the Figure 2 model_plan.json and model_totals.csv. Those inputs are not included. The aggregate CSVs are sufficient for the standalone plotting command above.

## Other included analyses

- **Main Figure 2(a)**: density_cells.csv contains the aggregated positive-positive cells and reconciliation.csv records the full and plotted denominators. The reference PNG is included. plot_figure2a.py is the original upstream script and needs the ordered shards, a model plan, and the earlier reconciliation reference passed through its command-line arguments. Use a working copy when rerunning it.
- **Appendix B.5**: snapshot_match_coverage_by_last_capture_heat.csv reports the 1- and 6-hour snapshot match counts and rates by last-capture original-post like bin. These rates use all eligible reposts in each bin, including unmatched reposts. Appendix B.6 uses separately defined joint support for formal contrasts.
- **Appendix E.6**: sir_d6 contains the simulation and summary scripts, locked schedules, synthetic trial outputs and plots for fixed and increasing M. Use Python 3.12, NumPy and Matplotlib. To regenerate summaries from the included trial CSVs, work in a copy of sir_d6 and run summarize.py for fixed_rate and common_rate. The trial CSVs are simulated node counts. For a fresh simulation, see run.py --help and the locked schedules.
- **Appendix F.2–F.3**: empirical_f contains analysis scripts and archived aggregate outputs for cumulative repost counts (F.2, empirical_f/03_repost_counts/) and 24-hour coverage (F.3, empirical_f/02_age24/). Re-estimation requires prepared Weibo data and upstream analysis inputs. Review the source scripts and runtime_versions.json before attempting an upstream run.

This package contains selected reproduction materials. The raw Weibo observations, full Figure 2(b) estimation inputs, original Figure 3(a) ledger, and legacy network run records are outside this package. Figure 9 and Figure S14 are not included as standalone reproduction workflows.

Selected reproduction scripts and aggregate results are publicly available on GitHub. The underlying Weibo observations were not uploaded and may be requested by emailing the corresponding author.
