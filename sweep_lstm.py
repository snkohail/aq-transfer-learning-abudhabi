#!/usr/bin/env python3
"""Leak-free LSTM hyperparameter sweep.

Justifies the LSTM architecture behind the headline "ridge beats LSTM" claim,
and answers: does that claim survive the BEST configuration? Sweeps
hidden {32, 64, 128} x dropout {0.1, 0.2, 0.3} = 9 configs on the DIRECT
transfer LSTM -- the architecture used by aq.stages.stage_lstm, whose numbers
appear in FINDINGS.md 4.1.

## Config selection is leak-free (this is the point)

The winning config is chosen using a validation split carved from SOURCE data
ONLY. No target adaptation window and no target test day influences the choice:

  1. Pool every rich-source sequence over its full history (targets never enter).
  2. Sort chronologically; hold out the most recent 15% as selection-val, a
     forward-in-time split that mirrors deployment.
  3. Train each of the 9 configs on the first 85% (its own internal 10% early-
     stopping split is carved from within that 85%, so selection-val stays
     untouched).
  4. Winner = the single global config with the lowest selection-val RMSE.

The full 9-config grid is ALSO evaluated on the target test sets, but only for
transparency -- that grid does not choose the winner. Reporting test numbers is
honest; selecting on them would be the leak.

## Production comparison

The winning config is then run through the leak-free authentic protocol (sources
end the day before each target's deployment; scored on the common index built by
aq.evaluate) and compared to ridge and persistence at K=30 and K=90.

Output: outputs/lstm_sweep.csv, outputs/lstm_sweep_selection.csv.
Runtime: ~25-35 min on one CPU. Run in the background.
"""

from __future__ import annotations

import itertools
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from aq import config as cfg
from aq.data import load_daily, station_metadata
from aq.evaluate import Prediction, align, build_common_index, rmse, score
from aq.features import build_sequences
from aq.models import LSTMBackbone, RidgeBackbone
from aq.splits import select_targets, source_frames, train_end_for
from aq.stats import compare, summarise

HIDDEN = (32, 64, 128)
DROPOUT = (0.1, 0.2, 0.3)
CONFIGS = list(itertools.product(HIDDEN, DROPOUT))
SELECTION_VAL_FRACTION = 0.15

# The config shown in the paper's main table: a mid-range representative, chosen
# to be neither the source-val selector's pick nor the best-on-test config, so
# the LSTM is presented at neither its worst nor a cherry-picked best.
REPRESENTATIVE = (64, 0.2)


def log(*a):
    print(*a, flush=True)


def _scale_3d(scaler, X):
    shape = X.shape
    return scaler.transform(X.reshape(-1, shape[2])).reshape(shape)


def pooled_source_sequences(daily, meta, train_end, target_first):
    """Every eligible source's sequences for one target, dates aligned."""
    frames = source_frames(daily, train_end, target_first)
    parts = {s: build_sequences(f) for s, f in frames.items()}
    parts = {s: d for s, d in parts.items() if len(d) > cfg.MIN_SOURCE_ROWS}
    if len(parts) < cfg.MIN_SOURCES:
        return None, None
    X = np.concatenate([d.X for d in parts.values()])
    y = np.concatenate([d.y for d in parts.values()])
    return X, y


# ---------------------------------------------------------------------------
# leak-free config selection: source data only
# ---------------------------------------------------------------------------
def select_config(daily, meta):
    """Pick one global (hidden, dropout) using a source-only chronological split.

    Touches no target. Returns (winner, DataFrame of per-config val RMSE).

    Protocol constraint (matches splits.source_frames): the selection pool is
    restricted to source data STRICTLY BEFORE the earliest target's first
    observation. At real deployment no later data existed, so a config chosen on
    it could not have been selected in practice. Without this cut the validation
    slice ran into 2024, concurrent with every target's test window -- not
    leakage of target data, but a violation of the same temporal rule the rest
    of the pipeline enforces.
    """
    targets = select_targets(daily, meta)
    earliest_first = meta[meta.sheet.isin(targets)]["first"].min()
    cutoff = train_end_for(earliest_first)  # day before the earliest deployment

    frames = {
        s: build_sequences(
            daily[(daily.sheet == s) & (daily.date <= cutoff)][["date", "pm25"]]
        )
        for s in cfg.RICH_SOURCES
    }
    frames = {s: d for s, d in frames.items() if len(d) > cfg.MIN_SOURCE_ROWS}

    # concatenate with dates, then order chronologically so the held-out slice
    # is the most RECENT admissible source data (forward validation, pre-deployment).
    X = np.concatenate([d.X for d in frames.values()])
    y = np.concatenate([d.y for d in frames.values()])
    dates = np.concatenate([np.asarray(d.index.values) for d in frames.values()])
    order = np.argsort(dates)
    X, y = X[order], y[order]

    if len(X) > cfg.LSTM_MAX_SOURCE_SEQ:
        # evenly spaced thinning across the whole ordered range (np.linspace),
        # preserving chronological coverage while capping the count.
        keep = np.linspace(0, len(X) - 1, cfg.LSTM_MAX_SOURCE_SEQ).astype(int)
        X, y = X[keep], y[keep]

    cut = int((1.0 - SELECTION_VAL_FRACTION) * len(X))
    X_fit, y_fit = X[:cut], y[:cut]
    X_val, y_val = X[cut:], y[cut:]
    log(f"selection pool cut at {cutoff.date()} (before earliest target "
        f"{earliest_first.date()})")

    scaler = StandardScaler().fit(X_fit.reshape(-1, X_fit.shape[2]))
    X_fit_s, X_val_s = _scale_3d(scaler, X_fit), _scale_3d(scaler, X_val)

    log("=" * 78)
    log("CONFIG SELECTION — source-only chronological validation "
        f"(fit={len(X_fit)}, val={len(X_val)})")
    log("=" * 78)
    rows = []
    for hidden, dropout in CONFIGS:
        model = LSTMBackbone(hidden=hidden, dropout=dropout).fit_sources(X_fit_s, y_fit)
        val_rmse = rmse(y_val, model.predict(X_val_s))
        rows.append(dict(hidden=hidden, dropout=dropout,
                         config=f"h{hidden}_d{dropout}", val_rmse=val_rmse))
        log(f"  h{hidden:<3} d{dropout}  source-val RMSE = {val_rmse:.4f}")
    table = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
    winner = table.iloc[0]
    log(f"\nWINNER (source-val, leak-free): {winner.config}  "
        f"val RMSE {winner.val_rmse:.4f}")
    return (int(winner.hidden), float(winner.dropout)), table


# ---------------------------------------------------------------------------
# transparency grid: every config on the target test sets
# ---------------------------------------------------------------------------
def evaluate_grid(daily, meta, started):
    targets = select_targets(daily, meta)

    # per-target: target sequences, ridge (config-independent), common index per K
    prep = {}
    ridge_cache = {}
    for sheet in targets:
        row = meta[meta.sheet == sheet].iloc[0]
        train_end = train_end_for(row["first"])
        target = build_sequences(daily[daily.sheet == sheet][["date", "pm25"]])
        if len(target) == 0:
            continue
        Xsrc, ysrc = pooled_source_sequences(daily, meta, train_end, row["first"])
        if Xsrc is None:
            continue
        scaler = StandardScaler().fit(Xsrc.reshape(-1, Xsrc.shape[2]))
        key = str(train_end.date())
        if key not in ridge_cache:
            ridge_cache[key] = RidgeBackbone().fit_sources(Xsrc, ysrc)
        ridge = ridge_cache[key]

        per_k = {}
        for k_adapt in cfg.K_LIST:
            test_start = row["first"] + timedelta(days=k_adapt)
            index = build_common_index(target, not_before=test_start)
            if len(index) == 0:
                continue
            X_test, y_test = align(target, index)
            per_k[k_adapt] = dict(
                index=index, y_test=y_test,
                persistence=X_test[:, -1, 0],
                ridge=ridge.predict(X_test),
                X_test_scaled=_scale_3d(scaler, X_test),
            )
        if per_k:
            prep[sheet] = dict(name=row["name"], train_end=train_end,
                               Xsrc_scaled=_scale_3d(scaler, Xsrc), ysrc=ysrc,
                               per_k=per_k)

    rows = []
    lstm_cache = {}
    for ci, (hidden, dropout) in enumerate(CONFIGS, 1):
        tag = f"h{hidden}_d{dropout}"
        for sheet, info in prep.items():
            key = (hidden, dropout, str(info["train_end"].date()))
            if key not in lstm_cache:
                log(f"[grid {ci}/{len(CONFIGS)} {tag}] train direct LSTM "
                    f"train_end={key[2]}  ({time.time() - started:.0f}s)")
                lstm_cache[key] = LSTMBackbone(
                    hidden=hidden, dropout=dropout
                ).fit_sources(info["Xsrc_scaled"], info["ysrc"])
            model = lstm_cache[key]
            for k_adapt, slot in info["per_k"].items():
                index = slot["index"]
                preds = {
                    "persistence": Prediction(slot["persistence"], index),
                    "ridge": Prediction(slot["ridge"], index),
                    "lstm": Prediction(model.predict(slot["X_test_scaled"]), index),
                }
                scored = score(slot["y_test"], preds, index)
                rows.append(dict(
                    target=sheet, name=info["name"], hidden=hidden, dropout=dropout,
                    config=tag, K=k_adapt, n_test=len(index),
                    persistence=scored["rmse_all_persistence"],
                    ridge=scored["rmse_all_ridge"],
                    lstm=scored["rmse_all_lstm"],
                ))
    return pd.DataFrame(rows)


def _prep_targets(daily, meta):
    """Per-target sequences, ridge, common index per K, and scaled source data.

    Seed-independent, so it is computed once and shared across seeds. Identical
    to the prep inside evaluate_grid (kept separate to avoid disturbing that
    committed function)."""
    prep, ridge_cache = {}, {}
    for sheet in select_targets(daily, meta):
        row = meta[meta.sheet == sheet].iloc[0]
        train_end = train_end_for(row["first"])
        target = build_sequences(daily[daily.sheet == sheet][["date", "pm25"]])
        if len(target) == 0:
            continue
        Xsrc, ysrc = pooled_source_sequences(daily, meta, train_end, row["first"])
        if Xsrc is None:
            continue
        scaler = StandardScaler().fit(Xsrc.reshape(-1, Xsrc.shape[2]))
        key = str(train_end.date())
        if key not in ridge_cache:
            ridge_cache[key] = RidgeBackbone().fit_sources(Xsrc, ysrc)
        per_k = {}
        for k_adapt in cfg.K_LIST:
            test_start = row["first"] + timedelta(days=k_adapt)
            index = build_common_index(target, not_before=test_start)
            if len(index) == 0:
                continue
            X_test, y_test = align(target, index)
            per_k[k_adapt] = dict(index=index, y_test=y_test,
                                  persistence=X_test[:, -1, 0],
                                  ridge=ridge_cache[key].predict(X_test),
                                  X_test_scaled=_scale_3d(scaler, X_test))
        if per_k:
            prep[sheet] = dict(name=row["name"], train_end=train_end,
                               Xsrc_scaled=_scale_3d(scaler, Xsrc), ysrc=ysrc, per_k=per_k)
    return prep


def evaluate_grid_seeds(daily, meta, started, seeds=cfg.SEEDS):
    """The 9-config grid retrained under each seed, to estimate training noise."""
    prep = _prep_targets(daily, meta)
    rows, lstm_cache = [], {}
    for seed in seeds:
        for ci, (hidden, dropout) in enumerate(CONFIGS, 1):
            tag = f"h{hidden}_d{dropout}"
            for sheet, info in prep.items():
                key = (hidden, dropout, seed, str(info["train_end"].date()))
                if key not in lstm_cache:
                    log(f"[seed {seed} cfg {ci}/{len(CONFIGS)} {tag}] "
                        f"train_end={key[3]}  ({time.time() - started:.0f}s)")
                    lstm_cache[key] = LSTMBackbone(
                        hidden=hidden, dropout=dropout, seed=seed
                    ).fit_sources(info["Xsrc_scaled"], info["ysrc"])
                model = lstm_cache[key]
                for k_adapt, slot in info["per_k"].items():
                    index = slot["index"]
                    preds = {
                        "persistence": Prediction(slot["persistence"], index),
                        "ridge": Prediction(slot["ridge"], index),
                        "lstm": Prediction(model.predict(slot["X_test_scaled"]), index),
                    }
                    scored = score(slot["y_test"], preds, index)
                    rows.append(dict(
                        target=sheet, name=info["name"], hidden=hidden, dropout=dropout,
                        config=tag, seed=seed, K=k_adapt, n_test=len(index),
                        persistence=scored["rmse_all_persistence"],
                        ridge=scored["rmse_all_ridge"],
                        lstm=scored["rmse_all_lstm"]))
    return pd.DataFrame(rows)


def report_seeds(seeds_frame):
    """Capacity finding, now with training noise estimated across seeds."""
    from scipy.stats import kruskal
    g = seeds_frame
    log("\n" + "=" * 78)
    log(f"SEED SWEEP — {g.seed.nunique()} seeds x {g.config.nunique()} configs; "
        "capacity vs training noise")
    log("=" * 78)
    for K in sorted(g.K.unique()):
        s = g[g.K == K]
        # per (config, seed): median RMSE across targets
        cs = s.groupby(["config", "hidden", "dropout", "seed"]).lstm.median().reset_index()
        # per config: median across seeds + seed spread (max-min)
        per_cfg = cs.groupby(["config", "hidden", "dropout"]).lstm.agg(
            median="median", spread=lambda v: v.max() - v.min()).reset_index()
        log(f"\n--- K={K} ---")
        log(f"{'config':<12}{'median':>9}{'seed spread':>13}")
        for _, r in per_cfg.sort_values("median").iterrows():
            log(f"{r.config:<12}{r['median']:>9.3f}{r.spread:>13.3f}")

        within = float(per_cfg.spread.median())            # pure training noise
        level_med = per_cfg.groupby("hidden")["median"].median()
        between = float(level_med.max() - level_med.min())  # capacity signal
        ratio = between / within if within else float("inf")
        ordered = list(level_med.sort_index().values)
        monotonic = ordered[0] < ordered[1] < ordered[2]
        # trend test across the 3 hidden LEVELS (config-seed medians as units)
        groups = [cs[cs.hidden == h].lstm.values for h in (32, 64, 128)]
        kw = kruskal(*groups)
        log(f"  hidden-level medians: h32={level_med[32]:.3f} h64={level_med[64]:.3f} "
            f"h128={level_med[128]:.3f}  (monotonic h32<h64<h128: {monotonic})")
        log(f"  between-level range {between:.3f}  vs  within-config seed spread "
            f"{within:.3f}  -> ratio {ratio:.2f}x")
        log(f"  Kruskal-Wallis across 3 hidden groups: H={kw.statistic:.2f} p={kw.pvalue:.4f}")
        # ridge below every config-seed?
        ridge_med = s.groupby("seed").ridge.median().median()
        worst_lstm = cs.lstm.min()  # best (lowest) LSTM config-seed median
        log(f"  ridge median {ridge_med:.3f}  vs best LSTM config-seed median "
            f"{worst_lstm:.3f}  -> ridge below all: {ridge_med < worst_lstm}")


def capacity_analysis(grid, selection):
    """Why source-val ranks configs backwards: model capacity.

    Reports, per K: Spearman of hidden size and of dropout against target-test
    median RMSE; and Spearman of source-val rank against target-test rank. The
    capacity effect (hidden -> worse transfer) is the mechanism; the source-val
    inversion is its consequence, since larger models fit source data better.

    NOTE: these Spearman coefficients are computed on 9 SINGLE-RUN, non-independent
    points and overstate the evidence. The properly-powered version -- training
    noise estimated across 3 seeds, a Kruskal-Wallis trend test across the 3
    hidden levels -- is `report_seeds` (run `python sweep_lstm.py --seeds`). Cite
    that, not the numbers below, for the capacity finding.
    """
    sel = selection.set_index("config").val_rmse
    log("\n" + "=" * 78)
    log("CAPACITY ANALYSIS — what drives the config ranking")
    log("=" * 78)
    for k_adapt in sorted(grid.K.unique()):
        med = (grid[grid.K == k_adapt]
               .groupby(["config", "hidden", "dropout"]).lstm.median().reset_index())
        rh, ph = spearmanr(med.hidden, med.lstm)
        rd, pd_ = spearmanr(med.dropout, med.lstm)
        df = med.set_index("config")
        common = sel.index.intersection(df.index)
        rv, pv = spearmanr(sel.loc[common], df.loc[common, "lstm"])
        order = med.sort_values("lstm").reset_index(drop=True)
        h32 = [i + 1 for i, r in order.iterrows() if r.hidden == 32]
        h128 = [i + 1 for i, r in order.iterrows() if r.hidden == 128]
        log(f"\n--- K={k_adapt} ---")
        log(f"  hidden  vs target RMSE : Spearman {rh:+.3f}  p={ph:.4f}   (capacity effect)")
        log(f"  dropout vs target RMSE : Spearman {rd:+.3f}  p={pd_:.4f}   (null)")
        log(f"  source-val vs target   : Spearman {rv:+.3f}  p={pv:.4f}   (consequence)")
        log(f"  h32 ranks (best=1): {h32}   h128 ranks: {h128}")


def report_grid(grid, selection):
    """Full item-1 report from the saved grid + selection tables (no retraining)."""
    win = selection.sort_values("val_rmse").iloc[0].config
    log("=" * 78)
    log("TRANSPARENCY GRID — every config on the target test sets")
    log("(reporting only; the source-val winner was NOT chosen on this)")
    log("=" * 78)
    references = {
        f"h{REPRESENTATIVE[0]}_d{REPRESENTATIVE[1]}": "representative",
        win: "source-val pick",
    }
    for k_adapt in sorted(grid.K.unique()):
        sub = grid[grid.K == k_adapt]
        ridge_med = sub.ridge.median()
        rows = []
        for tag, g in sub.groupby("config"):
            c = compare(g.lstm, g.ridge)  # a=lstm, b=ridge -> b_wins = ridge wins
            rows.append((g.lstm.median(), tag, c.b_wins, c.n, c.p_value))
        rows.sort()
        oracle = rows[0][1]
        log(f"\n--- K={k_adapt}  (ridge median {ridge_med:.3f}) ---")
        log(f"{'config':<12}{'LSTM med':>10}{'ridge wins':>12}{'p(ridge<lstm)':>15}   note")
        for med, tag, rw, rn, rp in rows:
            note = references.get(tag, "oracle (best-on-test)" if tag == oracle else "")
            log(f"{tag:<12}{med:>10.3f}{f'{rw}/{rn}':>12}{rp:>15.4f}   {note}")
        med_of_meds = float(np.median([r[0] for r in rows]))
        log(f"  best-of-9 upward bias (median config - oracle) = "
            f"{med_of_meds - rows[0][0]:.3f} RMSE")

    # three reference configs vs ridge, side by side
    log("\n" + "=" * 78)
    log("THREE REFERENCE CONFIGS vs ridge (report all; adopt none as 'the LSTM')")
    log("=" * 78)
    labels = {f"h{REPRESENTATIVE[0]}_d{REPRESENTATIVE[1]}": "representative",
              win: "source-val pick"}
    for k_adapt in sorted(grid.K.unique()):
        sub = grid[grid.K == k_adapt]
        oracle = sub.groupby("config").lstm.median().idxmin()
        labels_k = dict(labels); labels_k.setdefault(oracle, "oracle (best-on-test)")
        log(f"\n--- K={k_adapt} ---")
        for tag, lab in labels_k.items():
            b = sub[sub.config == tag]
            rl = compare(b.lstm, b.ridge)
            log(f"  {lab:<22} {tag:<11} LSTM {b.lstm.median():.3f}  "
                f"ridge {b.ridge.median():.3f}  ridge wins {rl.b_wins}/{rl.n}  p={rl.p_value:.4f}")

    capacity_analysis(grid, selection)


def main():
    started = time.time()
    daily = load_daily()
    meta = station_metadata(daily)
    log(f"configs={len(CONFIGS)}  (hidden {HIDDEN} x dropout {DROPOUT})\n")

    _, selection = select_config(daily, meta)
    selection.to_csv(cfg.OUTPUT_DIR / "lstm_sweep_selection.csv", index=False)

    grid = evaluate_grid(daily, meta, started)
    grid.to_csv(cfg.OUTPUT_DIR / "lstm_sweep.csv", index=False)

    report_grid(grid, selection)
    log(f"\ntotal {time.time() - started:.0f}s")


def main_seeds():
    """Retrain the 9-config grid under each seed; write lstm_sweep_seeds.csv."""
    started = time.time()
    daily = load_daily()
    meta = station_metadata(daily)
    log(f"seeds={cfg.SEEDS}  configs={len(CONFIGS)}  "
        f"(~{len(cfg.SEEDS) * len(CONFIGS)} config-runs)\n")
    seeds_frame = evaluate_grid_seeds(daily, meta, started)
    seeds_frame.to_csv(cfg.OUTPUT_DIR / "lstm_sweep_seeds.csv", index=False)
    report_seeds(seeds_frame)
    log(f"\ntotal {time.time() - started:.0f}s")


def analyze():
    """Reproduce the full report from committed CSVs, without retraining."""
    grid = pd.read_csv(cfg.OUTPUT_DIR / "lstm_sweep.csv")
    selection = pd.read_csv(cfg.OUTPUT_DIR / "lstm_sweep_selection.csv")
    report_grid(grid, selection)


def analyze_seeds():
    """Reproduce the seed-sweep report from the committed CSV, without retraining."""
    report_seeds(pd.read_csv(cfg.OUTPUT_DIR / "lstm_sweep_seeds.csv"))


if __name__ == "__main__":
    import sys

    if "--seeds" in sys.argv:
        main_seeds()
    elif "--analyze-seeds" in sys.argv:
        analyze_seeds()
    elif "--analyze" in sys.argv:
        analyze()
    else:
        main()
