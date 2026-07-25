#!/usr/bin/env python3
"""Report the two local baselines that the methods define but no table reports.

  local_mean         predict the mean of the target's K-day adaptation window for
                     every test day (an adaptation-window mean baseline)
  local_ridge_unreg  the full 18-feature local ridge at alpha=1.0, fit on the
                     ~16-sequence K=30 adaptation window (the model the paper
                     calls unstable)

Both already exist inside `stage_baselines` and are scored on the common index,
but they never made it into a result table -- which reads as selective. This
re-runs that stage and emits median RMSE + paired win-rate vs persistence at K=30
and K=90 for an appendix table.

    python scripts/local_baselines.py

Writes outputs/baselines_full.csv (summary) and outputs/baselines_full_detail.csv
(per target). Needs the station workbook (see data/README.md).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from aq import config as cfg
from aq import stages
from aq.stats import compare

OUT = cfg.OUTPUT_DIR
BASELINES = {
    "local_mean": "adaptation-window mean",
    "local_ridge_unreg": "full 18-feature local ridge (alpha=1.0)",
}


def main():
    # Run the committed stage into a throwaway dir so we never touch the
    # published baselines.csv, then summarise the two under-reported columns.
    frame = stages.stage_baselines(Path(tempfile.mkdtemp(prefix="aq_baselines_")))

    summary = []
    print("=" * 84)
    print("LOCAL BASELINES -- median RMSE and paired win-rate vs persistence")
    print("=" * 84)
    for K in cfg.K_LIST:
        sub = frame[frame.K == K]
        note = "; station_20 dropped (0 adaptation sequences)" if K == 30 else ""
        print(f"\n--- K={K}  (n={len(sub)} targets{note}) ---")
        print(f"  persistence     median RMSE {sub.persistence.median():8.3f}")
        print(f"  pooled_transfer median RMSE {sub.pooled_transfer.median():8.3f}")
        for col, desc in BASELINES.items():
            c = compare(sub.persistence, sub[col])  # b = baseline; b_wins = baseline<pers
            p = f"{c.p_value:.4f}" if np.isfinite(c.p_value) else "-"
            print(f"  {col:<20} median RMSE {sub[col].median():8.3f}   "
                  f"beats persistence {c.b_wins}/{c.n} (p={p})   [{desc}]")
            summary.append(dict(
                K=K, baseline=col, description=desc,
                n_targets=int(c.n), median_rmse=round(float(sub[col].median()), 4),
                persistence_median=round(float(sub.persistence.median()), 4),
                pooled_transfer_median=round(float(sub.pooled_transfer.median()), 4),
                wins_vs_persistence=int(c.b_wins),
                winrate_vs_persistence=round(c.b_wins / c.n, 4),
                p_value_wilcoxon=(round(float(c.p_value), 6) if np.isfinite(c.p_value) else np.nan),
                pct_worse_than_persistence=round(
                    100.0 * (sub[col].median() / sub.persistence.median() - 1.0), 1)))

    pd.DataFrame(summary).to_csv(OUT / "baselines_full.csv", index=False)
    detail = ["target", "name", "K", "n_test", "n_adapt", "persistence",
              "pooled_transfer", "local_mean", "local_ridge_unreg"]
    frame[detail].sort_values(["K", "target"]).to_csv(
        OUT / "baselines_full_detail.csv", index=False)
    print(f"\nsaved -> {OUT / 'baselines_full.csv'}, {OUT / 'baselines_full_detail.csv'}")


if __name__ == "__main__":
    main()
