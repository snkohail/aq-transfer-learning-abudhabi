#!/usr/bin/env python3
"""No-interpolation sensitivity analysis (protects the leak-free claim).

Gap interpolation fills interior runs of <=3 days before the source/target
split, so in principle a bidirectionally-interpolated value near a forecast
origin could borrow from a later observation. The magnitude is tiny, but because
"leak-free" is the paper's headline we show it empirically rather than argue it.

We reload the whole corpus with interpolation switched OFF (raw non-null values
only, every gap left NaN). The 14-day window-validity rule then naturally drops
any window that would have leaned on a filled day. We re-run the two headline
analyses -- overall pooled transfer vs persistence, and the extreme-day
decomposition -- at K=30 and K=90 under both conditions, routing every
comparison through build_common_index() + score(), and check that all qualitative
conclusions survive.

    python scripts/leakfree_sensitivity.py

Writes outputs/no_interp_sensitivity.csv. Needs the station workbook (see
data/README.md).
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from aq import config as cfg
from aq.data import load_daily, neighbour_lag_frames, station_metadata, wide_matrix
from aq.evaluate import Prediction, align, build_common_index, score
from aq.features import build_design
from aq.models import RidgeBackbone
from aq.splits import select_targets, train_end_for
from aq.stages import _pooled_source_design
from aq.stats import compare

OUT = cfg.OUTPUT_DIR
SLICES = ("all", "normal", "ext", "extrel", "onset")


def analyse(interpolate: bool):
    """Per-(target, K) overall + extreme decomposition for one corpus variant."""
    cond = "interp" if interpolate else "no_interp"
    daily = load_daily(interpolate=interpolate)
    meta = station_metadata(daily)
    wide1, wide2 = neighbour_lag_frames(wide_matrix(daily))
    targets = select_targets(daily, meta)

    corpus_nonnull = int(daily["pm25"].notna().sum())
    rows, design_windows = [], {}
    for sheet in targets:
        row = meta[meta.sheet == sheet].iloc[0]
        train_end = train_end_for(row["first"])
        design = build_design(daily[daily.sheet == sheet][["date", "pm25"]])
        design_windows[sheet] = len(design)
        threshold = (float(np.nanpercentile(design.y, cfg.EXTREME_PCT))
                     if len(design.y) else float("nan"))
        Xtr, ytr = _pooled_source_design(daily, meta, train_end, row["first"], 0, wide1, wide2)
        model = RidgeBackbone().fit_sources(Xtr, ytr) if Xtr is not None else None
        for K in cfg.K_LIST:
            index = build_common_index(design, not_before=row["first"] + timedelta(days=K))
            if len(index) == 0 or model is None:
                continue
            X_test, y_test = align(design, index)
            preds = {
                "model": Prediction(model.predict(X_test), index),         # pooled transfer
                "persistence": Prediction(X_test[:, cfg.SEQ_L - 1], index),
            }
            scored = score(y_test, preds, index, relative_threshold=threshold)
            rows.append(dict(condition=cond, target=sheet, name=row["name"], K=K,
                             n_test=len(index), extrel_threshold=threshold, **scored))
    info = dict(condition=cond, corpus_nonnull=corpus_nonnull, n_targets=len(targets),
                targets=targets, design_windows=design_windows)
    return pd.DataFrame(rows), info


def slice_compare(frame, K, slice_name):
    sub = frame[frame.K == K]
    return compare(sub[f"rmse_{slice_name}_persistence"], sub[f"rmse_{slice_name}_model"],
                   label=slice_name)


def fmt(c):
    p = f"{c.p_value:.4f}" if np.isfinite(c.p_value) else "  -  "
    return (f"n={c.n:<2} pers={c.median_a:7.3f} transfer={c.median_b:7.3f} "
            f"gain={c.improvement_pct:+6.1f}% dPair={c.median_diff:+7.3f} "
            f"wins={c.b_wins}/{c.n} p={p}")


def main():
    frame_i, info_i = analyse(interpolate=True)
    frame_n, info_n = analyse(interpolate=False)
    pd.concat([frame_i, frame_n], ignore_index=True).to_csv(
        OUT / "no_interp_sensitivity.csv", index=False)

    print("=" * 82)
    print("DATA LOSS -- interpolated (committed) vs no-interpolation")
    print("=" * 82)
    lost = info_i["corpus_nonnull"] - info_n["corpus_nonnull"]
    print(f"  corpus non-null station-days : interp {info_i['corpus_nonnull']}  "
          f"no-interp {info_n['corpus_nonnull']}  lost {lost} "
          f"({100 * lost / info_n['corpus_nonnull']:.2f}%)")
    dropped = sorted(set(info_i["targets"]) - set(info_n["targets"]))
    print(f"  eligible targets             : interp {info_i['n_targets']}  "
          f"no-interp {info_n['n_targets']}  (dropped: {dropped or 'none'})")
    for K in cfg.K_LIST:
        ri = int(frame_i[frame_i.K == K].n_test.sum())
        rn = int(frame_n[frame_n.K == K].n_test.sum())
        print(f"  retained test windows K={K:<2}   : interp {ri}  no-interp {rn}  lost {ri - rn}")

    for title, slices in [("OVERALL (all-days)", ["all"]),
                          ("EXTREME DECOMPOSITION", list(SLICES))]:
        print("\n" + "=" * 82)
        print(title)
        print("=" * 82)
        for K in cfg.K_LIST:
            print(f"--- K={K} ---")
            for s in slices:
                print(f"  {s:<7} interp    : {fmt(slice_compare(frame_i, K, s))}")
                print(f"  {s:<7} no-interp : {fmt(slice_compare(frame_n, K, s))}")

    print("\n" + "=" * 82)
    print("VERDICT")
    print("=" * 82)
    for K in cfg.K_LIST:
        ca = slice_compare(frame_n, K, "all")
        ce, cr = slice_compare(frame_n, K, "ext"), slice_compare(frame_n, K, "extrel")
        print(f"K={K}: transfer beats persistence overall {ca.b_wins}/{ca.n} "
              f"(p={ca.p_value:.4f}); loses on the tail (abs {ce.improvement_pct:+.1f}%, "
              f"extrel {cr.improvement_pct:+.1f}%) -> conclusions "
              f"{'HOLD' if ca.b_wins == ca.n and ce.improvement_pct < 0 else 'CHECK'}")
    print(f"\nsaved -> {OUT / 'no_interp_sensitivity.csv'}")


if __name__ == "__main__":
    main()
