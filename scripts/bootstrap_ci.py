#!/usr/bin/env python3
"""Calendar-block bootstrap for dependence-aware uncertainty (K=90).

The paper's Wilcoxon tests treat the 7-12 stations as independent, but they share
one airshed and the same dust episodes, so those p-values can be over-confident.
This resamples contiguous 14-day CALENDAR blocks with replacement and applies the
SAME sampled dates to EVERY station (shared blocks preserve the regional
dependence -- stations are never resampled independently). For the extreme slice
it also resamples contiguous extreme EPISODES, the physical unit of dependence.

Statistic per resample: for each station, RMSE_transfer - RMSE_persistence on the
resampled days (paired, per-station), then the median across stations. We report
the 95% percentile CI of that median for the overall, normal, and absolute-extreme
(>=75) slices.

Sign convention: D = RMSE_transfer - RMSE_persistence.
  overall & normal -> transfer better -> D<0 -> CI should exclude 0 (better side)
  extreme          -> transfer worse  -> D>0 -> CI should exclude 0 (worse side)

    python scripts/bootstrap_ci.py

Writes outputs/bootstrap_ci.csv. Needs the station workbook (see data/README.md).
"""
from __future__ import annotations

import sys
from datetime import timedelta
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from aq import config as cfg
from aq.data import load_daily, neighbour_lag_frames, station_metadata, wide_matrix
from aq.evaluate import align, build_common_index
from aq.features import build_design
from aq.models import RidgeBackbone
from aq.splits import select_targets, train_end_for
from aq.stages import _pooled_source_design

OUT = cfg.OUTPUT_DIR
K = 90
BLOCK = 14
N_BOOT = 2000
EXT = cfg.EXTREME_ABS
MIN_EXT_DAYS = cfg.MIN_SLICE_DAYS
MIN_KEEP_EXT = 3
EPISODE_GAP = 2


def per_target_errors():
    """Per-target K=90 common-index days with per-day squared errors of transfer
    and persistence -- the identical path used by stage_extreme."""
    daily = load_daily()
    meta = station_metadata(daily)
    wide1, wide2 = neighbour_lag_frames(wide_matrix(daily))
    out = []
    for sheet in select_targets(daily, meta):
        row = meta[meta.sheet == sheet].iloc[0]
        design = build_design(daily[daily.sheet == sheet][["date", "pm25"]])
        index = build_common_index(design, not_before=row["first"] + timedelta(days=K))
        if len(index) == 0:
            continue
        Xtr, ytr = _pooled_source_design(
            daily, meta, train_end_for(row["first"]), row["first"], 0, wide1, wide2)
        if Xtr is None:
            continue
        X_test, y_test = align(design, index)
        pred_tr = RidgeBackbone().fit_sources(Xtr, ytr).predict(X_test)
        pred_pe = X_test[:, cfg.SEQ_L - 1]
        out.append(dict(target=sheet, dates=pd.DatetimeIndex(index), y=np.asarray(y_test, float),
                        se_tr=(y_test - pred_tr) ** 2, se_pe=(y_test - pred_pe) ** 2))
    return out


def build_matrices(tdata):
    """Align every target onto the shared union calendar; precompute per-slice
    (T x N) squared-error and membership matrices."""
    all_dates = pd.DatetimeIndex(sorted(set().union(*[t["dates"] for t in tdata])))
    pos = {d: i for i, d in enumerate(all_dates)}
    N, T = len(all_dates), len(tdata)
    slices = ("all", "normal", "ext")
    INSLICE = {s: np.zeros((T, N)) for s in slices}
    SE_TR = {s: np.zeros((T, N)) for s in slices}
    SE_PE = {s: np.zeros((T, N)) for s in slices}
    ext_active = np.zeros(N, dtype=bool)
    for ti, t in enumerate(tdata):
        idx = np.array([pos[d] for d in t["dates"]])
        is_ext = t["y"] >= EXT
        member = {"all": np.ones(len(idx), bool), "normal": ~is_ext, "ext": is_ext}
        for s in slices:
            m = member[s]
            INSLICE[s][ti, idx[m]] = 1.0
            SE_TR[s][ti, idx[m]] = t["se_tr"][m]
            SE_PE[s][ti, idx[m]] = t["se_pe"][m]
        ext_active[idx[is_ext]] = True
    return all_dates, N, T, INSLICE, SE_TR, SE_PE, ext_active


def d_agg(w, INSLICE, SE_TR, SE_PE, key, min_days, min_keep, agg="median"):
    """Aggregate across stations of (RMSE_transfer - RMSE_persistence) under day
    weights w, dropping stations with fewer than min_days weighted days."""
    cnt = INSLICE[key] @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.sqrt((SE_TR[key] @ w) / cnt) - np.sqrt((SE_PE[key] @ w) / cnt)
    keep = cnt >= min_days
    if keep.sum() < min_keep:
        return np.nan
    return float(np.median(d[keep]) if agg == "median" else np.mean(d[keep]))


def find_episodes(ext_active, gap=EPISODE_GAP):
    """Contiguous extreme episodes over the union calendar, merging runs of
    extreme-active days separated by <= gap inactive days."""
    positions = np.where(ext_active)[0]
    if len(positions) == 0:
        return []
    episodes, start, prev = [], positions[0], positions[0]
    for p in positions[1:]:
        if p - prev <= gap + 1:
            prev = p
        else:
            episodes.append(np.arange(start, prev + 1))
            start = prev = p
    episodes.append(np.arange(start, prev + 1))
    return episodes


def summarise(samples, point):
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    lo95, hi95 = np.percentile(s, [2.5, 97.5])
    lo90, hi90 = np.percentile(s, [5.0, 95.0])
    support = float(np.mean(s > 0) if point > 0 else np.mean(s < 0))
    return dict(lo95=float(lo95), hi95=float(hi95), lo90=float(lo90), hi90=float(hi90),
                support=support, nv=len(s))


def main():
    rng = np.random.default_rng(cfg.SEED)
    tdata = per_target_errors()
    all_dates, N, T, INSLICE, SE_TR, SE_PE, ext_active = build_matrices(tdata)
    print("=" * 84)
    print(f"CALENDAR-BLOCK BOOTSTRAP  K={K}  stations={T}  calendar N={N} days "
          f"({all_dates.min().date()} -> {all_dates.max().date()})")
    print(f"blocks={BLOCK}d  resamples={N_BOOT}  seed={cfg.SEED}  "
          f"D = RMSE_transfer - RMSE_persistence")
    print("=" * 84)

    cfgs = {"all": (20, T // 2 + 1), "normal": (20, T // 2 + 1), "ext": (MIN_EXT_DAYS, MIN_KEEP_EXT)}
    ones = np.ones(N)
    point = {s: d_agg(ones, INSLICE, SE_TR, SE_PE, s, *cfgs[s]) for s in cfgs}
    point_ext_mean = d_agg(ones, INSLICE, SE_TR, SE_PE, "ext", MIN_EXT_DAYS, MIN_KEEP_EXT, "mean")
    n_stat = {s: int((INSLICE[s] @ ones >= cfgs[s][0]).sum()) for s in cfgs}

    n_blocks, starts_hi, offs = ceil(N / BLOCK), N - BLOCK + 1, np.arange(BLOCK)
    boot = {s: [] for s in cfgs}
    boot_ext_mean = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, starts_hi, size=n_blocks)
        w = np.bincount((starts[:, None] + offs[None, :]).ravel()[:N], minlength=N).astype(float)
        for s in cfgs:
            boot[s].append(d_agg(w, INSLICE, SE_TR, SE_PE, s, *cfgs[s]))
        boot_ext_mean.append(d_agg(w, INSLICE, SE_TR, SE_PE, "ext", MIN_EXT_DAYS, MIN_KEEP_EXT, "mean"))

    episodes = find_episodes(ext_active)
    E = len(episodes)
    boot_ep, boot_ep_mean = [], []
    for _ in range(N_BOOT if E >= 2 else 0):
        w = np.bincount(np.concatenate([episodes[c] for c in rng.integers(0, E, size=E)]),
                        minlength=N).astype(float)
        boot_ep.append(d_agg(w, INSLICE, SE_TR, SE_PE, "ext", MIN_EXT_DAYS, MIN_KEEP_EXT))
        boot_ep_mean.append(d_agg(w, INSLICE, SE_TR, SE_PE, "ext", MIN_EXT_DAYS, MIN_KEEP_EXT, "mean"))

    rows = []

    def record(slice_name, method, pt, samples, stat="median_station_D", extra=""):
        b = summarise(samples, pt)
        excl = (b["lo95"] > 0) or (b["hi95"] < 0)
        side = "worse" if pt > 0 else "better"
        rows.append(dict(K=K, slice=slice_name, method=method, statistic=stat,
                         D_definition="RMSE_transfer_minus_RMSE_persistence",
                         n_stations=n_stat[slice_name if slice_name in n_stat else "ext"],
                         point_estimate=round(pt, 4),
                         ci95_lo=round(b["lo95"], 4), ci95_hi=round(b["hi95"], 4),
                         ci90_lo=round(b["lo90"], 4), ci90_hi=round(b["hi90"], 4),
                         excludes_zero_95=bool(excl), one_sided_support=round(b["support"], 4),
                         effect_side=side, n_resamples=N_BOOT, n_valid_resamples=b["nv"], notes=extra))
        print(f"\n{slice_name:<7} [{method:<18}] D={pt:+.3f}  95% CI [{b['lo95']:+.3f}, "
              f"{b['hi95']:+.3f}]  excludes 0: {str(excl).upper()}  ({side})")
        print(f"{'':>9} 90% CI [{b['lo90']:+.3f}, {b['hi90']:+.3f}]  "
              f"one-sided support {100 * b['support']:.1f}%  valid {b['nv']}/{N_BOOT}")

    record("all", "date_block_14d", point["all"], boot["all"])
    record("normal", "date_block_14d", point["normal"], boot["normal"])
    record("ext", "date_block_14d", point["ext"], boot["ext"],
           extra="absolute >=75 ug/m3; sparse extreme days -> lumpy interval")
    if boot_ep:
        record("ext", "episode_block", point["ext"], boot_ep,
               extra=f"{E} extreme episodes; episodes resampled as the unit")
    record("ext", "date_block_14d_mean", point_ext_mean, boot_ext_mean,
           stat="mean_station_D", extra="mean-across-stations robustness")
    if boot_ep_mean:
        record("ext", "episode_block_mean", point_ext_mean, boot_ep_mean,
               stat="mean_station_D", extra="mean-across-stations robustness")

    pd.DataFrame(rows).to_csv(OUT / "bootstrap_ci.csv", index=False)
    print(f"\nsaved -> {OUT / 'bootstrap_ci.csv'}")


if __name__ == "__main__":
    main()
