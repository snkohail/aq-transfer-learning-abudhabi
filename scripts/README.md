# Supplementary analyses

These are the robustness checks and extra baselines that back up the main
results — the kind of thing a careful reviewer asks for. They aren't part of the
core `run.py` pipeline, but they use the same package (`aq`), the same leak-free
protocol, and the same `build_common_index()` + `score()` path, so their numbers
line up with the committed tables.

Each writes a CSV into [`../outputs/`](../outputs/) and prints a short summary.
All need the station workbook (see [`../data/README.md`](../data/README.md)); run
them from the repo root.

| Script | What it answers | Output |
|---|---|---|
| `leakfree_sensitivity.py` | Does the leak-free result survive turning gap interpolation **off** entirely? (Yes — every conclusion holds; ~1.2 % of station-days drop.) | `no_interp_sensitivity.csv` |
| `local_baselines.py` | The two local baselines the methods mention but never tabulate: an adaptation-window mean, and the unstable full 18-feature local ridge. | `baselines_full.csv`, `baselines_full_detail.csv` |
| `station_table.py` | One row per station (all 27): dates, coverage, role, and the **computed** reason each excluded station is excluded. | `station_table.csv` |
| `gbm_transfer.py` | Would gradient boosting transfer better than ridge? (No — ridge wins at every target, and it fails on the tail the same way.) | `gbm_transfer.csv` |
| `bootstrap_ci.py` | Dependence-aware uncertainty: a 14-day calendar-block bootstrap that shares blocks across stations, since they sit in one airshed. | `bootstrap_ci.csv` |

```bash
python scripts/station_table.py        # a fast one to start with
make supplementary                     # or run all five
```

A couple of these turned up things worth flagging, recorded in the CSVs
themselves: the station table shows that `RICH_SOURCES` is a *curated* long-baseline
set rather than a simple row-count threshold, and the block bootstrap shows the
extreme-day penalty, while directionally robust, sits right at the edge of 95 %
significance once regional dependence is accounted for.
