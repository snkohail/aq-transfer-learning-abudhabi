#!/usr/bin/env python3
"""Gradient-boosting pooled-transfer baseline, for comparison against ridge.

A HistGradientBoostingRegressor trained on the SAME pooled source design as the
ridge transfer model, scored on the SAME common index, at K=30 and K=90. It
answers the obvious "would a stronger nonlinear model transfer better?" question
with a number. (It does not: ridge beats it at every target, and it fails on the
tail the same way ridge does -- consistent with the capacity--transfer tension.)

    python scripts/gbm_transfer.py

Writes outputs/gbm_transfer.csv (per-target RMSE and MAE, all slices). Needs the
station workbook (see data/README.md).
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from aq import config as cfg
from aq.data import load_daily, neighbour_lag_frames, station_metadata, wide_matrix
from aq.evaluate import Prediction, align, build_common_index, score
from aq.features import build_design
from aq.models import RidgeBackbone
from aq.splits import select_targets, train_end_for
from aq.stages import _pooled_source_design
from aq.stats import compare

OUT = cfg.OUTPUT_DIR
GBM_KW = dict(max_iter=200, learning_rate=0.05, random_state=42)


def main():
    daily = load_daily()
    meta = station_metadata(daily)
    wide1, wide2 = neighbour_lag_frames(wide_matrix(daily))

    rows = []
    for sheet in select_targets(daily, meta):
        row = meta[meta.sheet == sheet].iloc[0]
        train_end = train_end_for(row["first"])
        design = build_design(daily[daily.sheet == sheet][["date", "pm25"]])
        threshold = float(np.nanpercentile(design.y, cfg.EXTREME_PCT)) if len(design.y) else np.nan
        Xtr, ytr = _pooled_source_design(daily, meta, train_end, row["first"], 0, wide1, wide2)
        if Xtr is None:
            continue
        ridge = RidgeBackbone().fit_sources(Xtr, ytr)
        gbm = HistGradientBoostingRegressor(**GBM_KW).fit(Xtr, ytr)
        for K in cfg.K_LIST:
            index = build_common_index(design, not_before=row["first"] + timedelta(days=K))
            if len(index) == 0:
                continue
            X_test, y_test = align(design, index)
            preds = {
                "gbm": Prediction(gbm.predict(X_test), index),
                "ridge": Prediction(ridge.predict(X_test), index),
                "persistence": Prediction(X_test[:, cfg.SEQ_L - 1], index),
            }
            scored = score(y_test, preds, index, relative_threshold=threshold)
            rows.append(dict(target=sheet, name=row["name"], K=K, n_test=len(index),
                             extrel_threshold=threshold, **scored))

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "gbm_transfer.csv", index=False)

    print("=" * 80)
    print(f"GBM POOLED TRANSFER vs RIDGE  (HistGBR {GBM_KW})")
    print("=" * 80)
    for K in cfg.K_LIST:
        sub = frame[frame.K == K]
        c = compare(sub.rmse_all_ridge, sub.rmse_all_gbm)     # a=ridge b=gbm
        cp = compare(sub.rmse_all_persistence, sub.rmse_all_gbm)
        ridge_beats = int((sub.rmse_all_ridge < sub.rmse_all_gbm).sum())
        print(f"\n--- K={K}  (n={len(sub)} targets) ---")
        print(f"  RMSE median : gbm {sub.rmse_all_gbm.median():.3f}  ridge "
              f"{sub.rmse_all_ridge.median():.3f}  persistence {sub.rmse_all_persistence.median():.3f}")
        print(f"  MAE  median : gbm {sub.mae_all_gbm.median():.3f}  ridge "
              f"{sub.mae_all_ridge.median():.3f}  persistence {sub.mae_all_persistence.median():.3f}")
        print(f"  ridge beats gbm {ridge_beats}/{c.n}  (gbm gain {c.improvement_pct:+.1f}%, "
              f"Wilcoxon p={c.p_value:.4f});  gbm beats persistence {cp.b_wins}/{cp.n}")
    print(f"\nsaved -> {OUT / 'gbm_transfer.csv'}")


if __name__ == "__main__":
    main()
