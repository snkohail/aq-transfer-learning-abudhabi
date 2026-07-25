"""Reproduce the headline numbers in FINDINGS.md Section 4.

Tolerances are set from measurement, not from taste:

  * The ridge / persistence stages are deterministic. Re-running the ORIGINAL
    scripts on a different platform (Linux/py3.12 -> macOS/py3.11, numpy 1.x ->
    2.4, pandas 2.x -> 3.0) moved every number by at most 3.1e-14 relative.
    REGRESSION_RTOL = 1e-12 is ~30x that. Bitwise equality would fail here for
    a reason that has nothing to do with correctness.

  * The LSTM stage trains with torch and does not reproduce bitwise across
    torch versions. Measured drift on this machine was 8.5e-4 relative, which
    left every decision unchanged. So its comparisons are gated on the
    DECISION (win-rate and significance), with the observed values reported.

If a number here fails, that is a finding. Report it -- do not widen the
tolerance until it passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aq import config as cfg
from aq.stats import compare

RTOL = cfg.REGRESSION_RTOL


@pytest.fixture(scope="module")
def extreme(tmp_path_factory):
    from aq import stages

    return stages.stage_extreme(tmp_path_factory.mktemp("extreme"))


@pytest.fixture(scope="module")
def ablation(tmp_path_factory):
    from aq import stages

    return stages.stage_ablation(tmp_path_factory.mktemp("ablation"))


@pytest.fixture(scope="module")
def baselines(tmp_path_factory):
    from aq import stages

    return stages.stage_baselines(tmp_path_factory.mktemp("baselines"))


@pytest.fixture(scope="module")
def detection(tmp_path_factory):
    from aq import stages

    out = tmp_path_factory.mktemp("detection")
    stages.stage_extreme(out)
    return pd.read_csv(out / "detection.csv")


@pytest.fixture(scope="module")
def ridge_alpha(tmp_path_factory):
    from aq import stages

    return stages.stage_ridge_alpha(tmp_path_factory.mktemp("ridge_alpha"))


# ---------------------------------------------------------------------------
# Ridge-penalty sensitivity: the ridge result is not an artifact of a lucky
# untuned alpha. Closes the tuning asymmetry with the 9-config LSTM sweep.
# ---------------------------------------------------------------------------
def test_ridge_alpha_is_flat_and_alpha1_near_optimal(ridge_alpha):
    from aq.stages import RIDGE_ALPHAS

    for K in (30, 90):
        sub = ridge_alpha[ridge_alpha.K == K]
        meds = {a: sub[f"ridge_a{a}"].median() for a in RIDGE_ALPHAS}
        best = min(meds.values())
        # alpha=1.0 (production) is within 0.01 RMSE of the best alpha
        assert meds[1.0] - best < 0.01, f"K={K}: alpha=1 not near-optimal: {meds}"
        # the [0.1, 10] plateau is essentially flat (< 0.05 RMSE spread)
        assert max(meds[0.1], meds[1.0], meds[10.0]) - min(
            meds[0.1], meds[1.0], meds[10.0]) < 0.05


def test_ridge_beats_persistence_at_every_alpha(ridge_alpha):
    from aq.stages import RIDGE_ALPHAS

    for K in (30, 90):
        sub = ridge_alpha[ridge_alpha.K == K]
        for a in RIDGE_ALPHAS:
            c = compare(sub.persistence, sub[f"ridge_a{a}"])  # b = ridge
            assert c.b_wins == 12, f"K={K}, alpha={a}: ridge beats persistence {c.b_wins}/12"
            assert c.p_value < 0.001


def test_ridge_beats_representative_lstm_across_alpha(ridge_alpha):
    """Ridge beats the representative LSTM at every alpha up to 100 (the plateau),
    so the ridge>LSTM result does not depend on the untuned penalty."""
    for K in (30, 90):
        sub = ridge_alpha[ridge_alpha.K == K]
        for a in (0.1, 1.0, 10.0, 100.0):
            c = compare(sub.lstm_repr, sub[f"ridge_a{a}"])  # b = ridge
            assert c.b_wins >= 10 and c.p_value < 0.05, f"K={K}, alpha={a}: {c.b_wins}/12 p={c.p_value}"


# ---------------------------------------------------------------------------
# Target-only baselines (item 2). Medians below are the verified audit figures;
# StandardScaler is fit on the ~16-sequence adaptation window only.
# ---------------------------------------------------------------------------
def test_baseline_ladder_k30(baselines):
    """K=30, 11 targets (station_20 has 0 adaptation sequences and is dropped)."""
    sub = baselines[baselines.K == 30]
    assert len(sub) == 11
    assert "station_20" not in set(sub.target)
    med = {m: sub[m].median() for m in
           ["local_ridge_unreg", "local_ridgecv", "local_mean", "local_ridge_hi",
            "local_lag1", "persistence", "pooled_transfer"]}
    assert med["local_ridge_unreg"] == pytest.approx(130.64, abs=0.05)  # alpha=1 collapses
    assert med["local_ridgecv"] == pytest.approx(53.93, abs=0.05)       # CV on ~16 pts fails
    assert med["local_mean"] == pytest.approx(20.19, abs=0.05)
    assert med["local_ridge_hi"] == pytest.approx(19.30, abs=0.05)
    assert med["local_lag1"] == pytest.approx(16.88, abs=0.05)          # best fair local
    assert med["persistence"] == pytest.approx(12.51, abs=0.05)
    assert med["pooled_transfer"] == pytest.approx(11.82, abs=0.05)
    # the ladder: transfer < persistence < best fair local
    assert med["pooled_transfer"] < med["persistence"] < med["local_lag1"]


def test_baseline_ladder_k90(baselines):
    """K=90, all 12 targets; local lag-1 nearly ties persistence, transfer still wins."""
    sub = baselines[baselines.K == 90]
    assert len(sub) == 12
    assert sub.local_lag1.median() == pytest.approx(12.60, abs=0.05)
    assert sub.persistence.median() == pytest.approx(12.33, abs=0.05)
    assert sub.pooled_transfer.median() == pytest.approx(11.59, abs=0.05)


def test_transfer_beats_best_local(baselines):
    """The finding: even the best fair local model loses to transfer at both K.

    Uses PAIRED statistics (median paired difference, win-rate, Wilcoxon), not a
    difference of column medians -- see the K=90 local-vs-persistence case where
    those two disagree in sign.
    """
    for K in (30, 90):
        sub = baselines[baselines.K == K]
        c = compare(sub.local_lag1, sub.pooled_transfer)  # b = transfer
        assert c.median_diff > 0  # transfer lower (better) on the paired median
        assert c.b_wins > c.n / 2  # transfer wins the majority of pairs
        assert c.p_value < 0.05, f"K={K}: transfer should beat best-local, p={c.p_value}"


def test_local_vs_persistence_direction(baselines):
    """Regression guard for the win-count error: at K=90 local BEATS persistence
    on the majority of paired comparisons (9/12) even though its column median is
    higher, because its few losses are large. Difference-of-medians and paired
    win-rate disagree in sign here; the paired statistic is authoritative."""
    sub90 = baselines[baselines.K == 90]
    c90 = compare(sub90.persistence, sub90.local_lag1)  # b = local
    assert c90.b_wins == 9  # local wins 9 of 12 pairs
    assert c90.median_diff > 0  # local lower (better) on the paired median
    assert c90.improvement_pct < 0  # ...yet WORSE by difference of column medians
    assert c90.p_value == pytest.approx(0.4238, abs=5e-4)  # not significant either way

    sub30 = baselines[baselines.K == 30]
    c30 = compare(sub30.persistence, sub30.local_lag1)  # b = local
    assert c30.b_wins == 1  # at K=30 local wins only 1 of 11
    assert c30.p_value == pytest.approx(0.0029, abs=5e-4)


# ---------------------------------------------------------------------------
# Section 2 -- data facts
# ---------------------------------------------------------------------------
def test_corpus_facts():
    from aq.data import load_daily, station_metadata

    daily = load_daily()
    observed = daily["pm25"].dropna()
    assert daily.sheet.nunique() == 27
    # After reindex + interpolation of interior gaps whose RUN LENGTH is <=3 days.
    # Was 26,191 while the loader used `.interpolate(limit=3)`, which also filled
    # the first 3 days of longer gaps: 306 fabricated values across 102 over-long
    # runs. 25,885 is the count under the documented per-run rule.
    assert len(observed) == 25885
    assert observed.max() == pytest.approx(491.75, rel=RTOL)
    assert observed.min() == pytest.approx(0.62, rel=RTOL)
    assert observed.median() == pytest.approx(34.47, abs=0.02)
    assert observed.mean() == pytest.approx(39.45, abs=0.02)
    assert (observed >= cfg.EXTREME_ABS).mean() == pytest.approx(0.0634, abs=0.0005)

    meta = station_metadata(daily)
    assert len(meta) == 27


def test_authentic_target_set():
    """The exact 12 targets from FINDINGS.md Section 2."""
    from aq.data import load_daily, station_metadata
    from aq.splits import select_targets

    daily = load_daily()
    targets = set(select_targets(daily, station_metadata(daily)))
    assert targets == {
        "station_12", "station_13", "station_14", "station_15",
        "station_16", "station_17", "station_18", "station_19",
        "station_20", "station_21", "station_22", "station_27",
    }
    # excluded: too few records to yield a test set
    assert "station_23" not in targets
    assert "station_24" not in targets


# ---------------------------------------------------------------------------
# Section 4.3 -- extreme-event decomposition
# ---------------------------------------------------------------------------
# Values below are POST gap-interpolation-fix. FINDINGS.md Section 4.3 records the
# pre-fix numbers, which were computed on a corpus containing 306 values fabricated
# inside over-long gaps. The deltas are small and no conclusion changed:
#   all     +5.8% -> +5.8%    normal +10.5% -> +10.4%
#   ext    -20.4% -> -20.6%   extrel -15.4% -> -15.6%   onset -7.6% -> -7.8%
# Every win-rate and every p-value is identical. FINDINGS.md has NOT been edited.
@pytest.mark.parametrize(
    "slice_name,n,persistence,model,gain,p_value",
    [
        ("all", 12, 12.405, 11.685, 5.8, 0.0005),
        ("normal", 12, 11.821, 10.589, 10.4, 0.0005),
        ("ext", 9, 26.854, 32.374, -20.6, 0.0039),
        ("extrel", 12, 19.139, 22.128, -15.6, 0.0010),
        ("onset", 7, 33.740, 36.376, -7.8, 0.0156),
    ],
)
def test_extreme_slices(extreme, slice_name, n, persistence, model, gain, p_value):
    result = compare(
        extreme[f"rmse_{slice_name}_persistence"],
        extreme[f"rmse_{slice_name}_model"],
        label=slice_name,
    )
    assert result.n == n
    assert result.median_a == pytest.approx(persistence, rel=1e-4)
    assert result.median_b == pytest.approx(model, rel=1e-4)
    assert result.improvement_pct == pytest.approx(gain, abs=0.05)
    assert result.p_value == pytest.approx(p_value, abs=5e-4)


def test_extrel_model_wins_exactly_one(extreme):
    """Item 6d: on the top-10% slice the transfer model beats persistence at
    exactly 1 of 12 targets (station_20, Al Muzoon: 19.506 vs 19.833). Guard so
    a future edit cannot silently turn this into 0/12."""
    c = compare(extreme.rmse_extrel_persistence, extreme.rmse_extrel_model)
    assert c.b_wins == 1
    winner = extreme.loc[extreme.rmse_extrel_model < extreme.rmse_extrel_persistence, "target"]
    assert set(winner) == {"station_20"}


def test_event_warning_skill(detection):
    """Item 3: next-day exceedance of the per-station top-10% threshold. The
    documented claim -- ridge misses ~72% of events -- restored from code."""
    ridge = detection[detection.model == "ridge"]
    pers = detection[detection.model == "persistence"]
    assert ridge.recall.median() == pytest.approx(0.28, abs=0.02)
    assert ridge.f1.median() == pytest.approx(0.34, abs=0.02)
    assert pers.recall.median() == pytest.approx(0.46, abs=0.02)
    assert pers.f1.median() == pytest.approx(0.46, abs=0.02)
    # the point: the trained model warns far less often than naive persistence
    assert ridge.recall.median() < pers.recall.median()


def test_the_gain_lives_entirely_on_ordinary_days(extreme):
    """The paper's central claim, as a test."""
    overall = compare(extreme.rmse_all_persistence, extreme.rmse_all_model)
    normal = compare(extreme.rmse_normal_persistence, extreme.rmse_normal_model)
    extremes = compare(extreme.rmse_ext_persistence, extreme.rmse_ext_model)

    assert overall.improvement_pct > 0 and overall.b_wins == overall.n
    assert normal.improvement_pct > overall.improvement_pct
    assert extremes.improvement_pct < -15
    assert extremes.b_wins == 0, "the model should lose on every measurable extreme slice"


# ---------------------------------------------------------------------------
# Section 5.5 -- spatial features on a common index
# ---------------------------------------------------------------------------
def test_common_index_holds_for_every_k(ablation):
    """If this fails the ablation is invalid, whatever the RMSEs say."""
    for target, group in ablation.groupby("target"):
        assert group.n_common.nunique() == 1, target
        assert group.rmse_all_persistence.nunique() == 1, target


# POST gap-fix. The ablation lost 6 of its 140 median common test days, so it is
# noisier than the pre-fix version recorded in FINDINGS.md Section 5.5:
#   overall  k=1 -1.4%/p=0.0098 -> -0.8%/p=0.0195    k=5 -2.3%/p=0.0645 -> -1.2%/p=0.4922
#   extreme  k=5 +1.3%/p=0.1602 -> +2.8%/p=0.0371    k=8 +2.3%/p=0.4316 -> +3.5%/p=0.3750
# k=5 extreme now has a raw p below 0.05. It does NOT survive Holm correction across
# the 10 tests in this grid (adjusted p=0.334), and k=8 has a LARGER effect with a
# WORSE p-value -- the signature of noise, not a dose-response. Treated as null; see
# test_the_refuted_35_percent_claim_does_not_reappear for the standing guard.
@pytest.mark.parametrize(
    "k,overall_gain,overall_p,extreme_gain",
    [
        (1, -0.8, 0.0195, -0.9),
        (2, -0.7, 0.0488, 0.2),
        (3, -0.7, 0.3750, 1.7),
        (5, -1.2, 0.4922, 2.8),
        (8, -1.0, 0.4922, 3.5),
    ],
)
def test_spatial_features_do_not_help(ablation, k, overall_gain, overall_p, extreme_gain):
    baseline = ablation[ablation.k == 0].set_index("target")
    arm = ablation[ablation.k == k].set_index("target")
    shared = baseline.index.intersection(arm.index)
    assert len(shared) == 10

    overall = compare(baseline.loc[shared, "rmse_all_model"], arm.loc[shared, "rmse_all_model"])
    assert overall.improvement_pct == pytest.approx(overall_gain, abs=0.05)
    assert overall.p_value == pytest.approx(overall_p, abs=5e-4)

    extremes = compare(
        baseline.loc[shared, "rmse_extrel_model"], arm.loc[shared, "rmse_extrel_model"]
    )
    assert extremes.improvement_pct == pytest.approx(extreme_gain, abs=0.05)


def test_capacity_effect_survives_seed_replication():
    """The capacity finding, validated against the committed 3-seed sweep
    (outputs/lstm_sweep_seeds.csv, 90-min run -- not regenerated here).

    Replaces the overstated single-run Spearman on 9 non-independent points with
    the properly-powered analysis: a between-hidden-level range several times the
    within-config training noise, a monotonic h32<h64<h128 ordering, a
    Kruskal-Wallis trend test across the 3 levels, and ridge below every
    config-seed median."""
    from scipy.stats import kruskal

    path = cfg.OUTPUT_DIR / "lstm_sweep_seeds.csv"
    if not path.exists():
        pytest.skip("seed sweep CSV absent; run `python sweep_lstm.py --seeds`")
    d = pd.read_csv(path)
    assert len(d) == 3 * 9 * 12 * 2 and sorted(d.seed.unique()) == [42, 43, 44]

    for K in (30, 90):
        s = d[d.K == K]
        cs = s.groupby(["config", "hidden", "seed"]).lstm.median().reset_index()
        per = cs.groupby(["config", "hidden"]).lstm.agg(
            median="median", spread=lambda v: v.max() - v.min()).reset_index()
        within = per.spread.median()
        lvl = per.groupby("hidden")["median"].median()
        between = lvl.max() - lvl.min()
        H, p = kruskal(*[cs[cs.hidden == h].lstm.values for h in (32, 64, 128)])

        assert lvl[32] < lvl[64] < lvl[128], f"K={K}: ordering broke: {dict(lvl)}"
        assert between / within > 2.0, f"K={K}: signal/noise only {between/within:.2f}x"
        assert p < 0.01, f"K={K}: hidden-level trend not significant (KW p={p:.4f})"
        # ridge sits below every config-seed LSTM median
        ridge_med = s.groupby("seed").ridge.median().median()
        assert ridge_med < cs.lstm.min(), f"K={K}: ridge not below all config-seeds"


def test_no_spatial_result_survives_multiplicity_correction(ablation):
    """The k-grid runs 10 tests; a raw p<0.05 among them means little on its own.

    After the gap-interpolation fix, k=5 on the extreme slice has a raw p of
    0.0371. This test records that it does NOT survive Holm correction across
    the grid, so nobody later reads that single cell as evidence that spatial
    neighbour features help on extreme days.

    If this test ever fails, something genuinely survived correction and is
    worth investigating properly -- do not delete the test, read the numbers.
    """
    baseline = ablation[ablation.k == 0].set_index("target")
    p_values = []
    for k in (1, 2, 3, 5, 8):
        arm = ablation[ablation.k == k].set_index("target")
        shared = baseline.index.intersection(arm.index)
        for column in ("rmse_all_model", "rmse_extrel_model"):
            p_values.append(
                (f"k={k}:{column}", compare(baseline.loc[shared, column], arm.loc[shared, column]).p_value)
            )

    ordered = sorted(p_values, key=lambda item: item[1])
    m = len(ordered)
    running = 0.0
    survivors = []
    for rank, (label, p) in enumerate(ordered):
        adjusted = min(1.0, max(running, (m - rank) * p))
        running = adjusted
        if adjusted < 0.05:
            survivors.append((label, p, adjusted))

    assert not survivors, (
        f"a spatial comparison survived Holm correction: {survivors}. "
        f"That would be a real finding -- investigate before believing it."
    )


def test_the_refuted_35_percent_claim_does_not_reappear(ablation):
    """FINDINGS 5.5. Any k showing a large extreme-event gain means the common
    index has been broken again."""
    baseline = ablation[ablation.k == 0].set_index("target")
    for k in (1, 2, 3, 5, 8):
        arm = ablation[ablation.k == k].set_index("target")
        shared = baseline.index.intersection(arm.index)
        gain = compare(
            baseline.loc[shared, "rmse_extrel_model"], arm.loc[shared, "rmse_extrel_model"]
        ).improvement_pct
        assert gain < 10.0, f"k={k} shows {gain:.1f}% on extremes -- check the index"


# ---------------------------------------------------------------------------
# Section 4.1 -- ridge vs LSTM (stochastic; gated on the decision)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_ridge_beats_lstm_and_both_beat_persistence(tmp_path):
    pytest.importorskip("torch")
    from aq import stages

    frame = stages.stage_lstm(tmp_path)
    for k_adapt in cfg.K_LIST:
        subset = frame[frame.K == k_adapt]
        assert len(subset) == 12

        ridge = compare(subset.rmse_all_persistence, subset.rmse_all_ridge)
        assert ridge.b_wins == 12, f"K={k_adapt}: ridge should beat persistence 12/12"
        assert ridge.p_value < 0.001

        versus = compare(subset.rmse_all_lstm, subset.rmse_all_ridge)
        assert versus.b_wins >= cfg.LSTM_MIN_RIDGE_WINS, (
            f"K={k_adapt}: ridge beat the LSTM at only {versus.b_wins}/12 "
            f"(observed p={versus.p_value:.4f}). FINDINGS reports 10/12."
        )
        assert versus.p_value < cfg.LSTM_MAX_P
