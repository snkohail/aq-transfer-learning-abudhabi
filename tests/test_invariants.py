"""The invariants that stop this project's known failure modes recurring.

Each test maps to a bug that actually invalidated a result. If one of these
starts failing, a real guarantee has been lost -- do not weaken the test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aq import config as cfg
from aq.evaluate import (
    IndexMismatchError,
    Prediction,
    align,
    build_common_index,
    score,
    slice_masks,
)
from aq.features import Design, build_design
from aq.models import SeasonalRouter, season_of
from aq.splits import LeakageError, build_manifest, require_adaptation_k, source_frames


def _design(dates, values) -> Design:
    index = pd.DatetimeIndex(dates)
    n = len(index)
    X = np.zeros((n, cfg.SEQ_L + 4))
    X[:, cfg.SEQ_L - 1] = values  # lag-1 / persistence column
    return Design(X=X, y=np.asarray(values, float), index=index, spatial=False, n_neighbours=0)


# ---------------------------------------------------------------------------
# FINDINGS 6.3 -- test-set alignment
# ---------------------------------------------------------------------------
def test_score_raises_on_different_indices():
    """The guard rail. Three results in this project died for want of it."""
    index = pd.date_range("2023-03-01", periods=10, freq="D")
    other = pd.date_range("2023-03-02", periods=10, freq="D")
    y = np.arange(10, dtype=float)

    predictions = {
        "good": Prediction(np.zeros(10), index),
        "misaligned": Prediction(np.zeros(10), other),
    }
    with pytest.raises(IndexMismatchError, match="different days"):
        score(y, predictions, index)


def test_score_raises_on_different_lengths():
    """The external ablation scored persistence on 507 points and models on 508."""
    index = pd.date_range("2023-03-01", periods=10, freq="D")
    y = np.arange(10, dtype=float)
    predictions = {"short": Prediction(np.zeros(9), index[:9])}
    with pytest.raises(IndexMismatchError, match="identical index"):
        score(y, predictions, index)


def test_common_index_is_the_intersection():
    """Section 5.5: 140 spatial days vs 331 temporal days must reconcile to 140."""
    wide_design = _design(pd.date_range("2023-01-01", periods=331, freq="D"), np.ones(331))
    narrow_dates = pd.date_range("2023-01-01", periods=331, freq="D")[:140]
    narrow_design = _design(narrow_dates, np.ones(140))

    index = build_common_index(wide_design, narrow_design, min_days=1)
    assert len(index) == 140
    assert index.equals(pd.DatetimeIndex(narrow_dates))


def test_common_index_returns_empty_below_min_days():
    design = _design(pd.date_range("2023-01-01", periods=20, freq="D"), np.ones(20))
    assert len(build_common_index(design, min_days=cfg.MIN_TEST)) == 0


def test_align_raises_when_design_cannot_cover_the_index():
    design = _design(pd.date_range("2023-01-01", periods=10, freq="D"), np.ones(10))
    index = pd.date_range("2023-01-05", periods=10, freq="D")  # runs past the design
    with pytest.raises(IndexMismatchError, match="missing"):
        align(design, index)


def test_persistence_comes_from_the_same_rows_as_the_model():
    """Reading the baseline off the design makes a row mismatch impossible."""
    values = np.array([10.0, 20.0, 30.0, 40.0])
    design = _design(pd.date_range("2023-01-01", periods=4, freq="D"), values)
    np.testing.assert_array_equal(design.persistence, values)


# ---------------------------------------------------------------------------
# gap interpolation -- the rule is per-RUN, not per-value
# ---------------------------------------------------------------------------
# These test CORRECTNESS, not agreement with the original scripts. The two are
# different properties: the six original loaders agreed with each other and were
# all wrong here, because `.interpolate(limit=n)` caps consecutive fills WITHIN
# a run instead of skipping over-long runs. test_equivalence.py covers agreement;
# this covers the semantics.
def _series_with_gaps(gap_lengths, *, lead=0, trail=0, spacing=6):
    """Build a station series with interior gaps of the given lengths."""
    from aq import config as cfg  # noqa: F401  (kept local for clarity)

    n = lead + trail + spacing * (len(gap_lengths) + 1) + sum(gap_lengths)
    index = pd.date_range("2021-01-01", periods=n, freq="D")
    values = np.arange(n, dtype=float) * 1.5 + 10.0
    spans = []
    cursor = lead + spacing
    for length in gap_lengths:
        values[cursor : cursor + length] = np.nan
        spans.append((cursor, length))
        cursor += length + spacing
    if lead:
        values[:lead] = np.nan
    if trail:
        values[-trail:] = np.nan
    return pd.Series(values, index=index), spans


@pytest.mark.parametrize("gap", [1, 2, 3])
def test_short_interior_gaps_are_fully_filled(gap):
    from aq.data import _interpolate_short_gaps

    series, spans = _series_with_gaps([gap])
    out = _interpolate_short_gaps(series, cfg.MAX_GAP)
    start, length = spans[0]
    assert int(out.iloc[start : start + length].notna().sum()) == length


@pytest.mark.parametrize("gap", [4, 5, 7, 30])
def test_over_long_interior_gaps_are_left_entirely_nan(gap):
    """A 4-day gap must NOT get its first 3 days filled. This was the bug."""
    from aq.data import _interpolate_short_gaps

    series, spans = _series_with_gaps([gap])
    out = _interpolate_short_gaps(series, cfg.MAX_GAP)
    start, length = spans[0]
    filled = int(out.iloc[start : start + length].notna().sum())
    assert filled == 0, (
        f"a {length}-day gap had {filled} day(s) synthesised; the rule applies to "
        f"the whole run, so it must be left entirely NaN"
    )


def test_mixed_gaps_in_one_series():
    """The case from the bug report: gaps of 1, 2, 3, 4 and 7 days together."""
    from aq.data import _interpolate_short_gaps

    lengths = [1, 2, 3, 4, 7]
    series, spans = _series_with_gaps(lengths)
    out = _interpolate_short_gaps(series, cfg.MAX_GAP)
    for (start, length) in spans:
        filled = int(out.iloc[start : start + length].notna().sum())
        expected = length if length <= cfg.MAX_GAP else 0
        assert filled == expected, f"{length}-day gap: filled {filled}, expected {expected}"


@pytest.mark.parametrize("lead,trail", [(1, 0), (0, 1), (3, 3), (5, 2)])
def test_gaps_touching_an_endpoint_are_never_filled(lead, trail):
    """Nothing brackets an edge gap, so there is nothing to interpolate between."""
    from aq.data import _interpolate_short_gaps

    series, _ = _series_with_gaps([2], lead=lead, trail=trail)
    out = _interpolate_short_gaps(series, cfg.MAX_GAP)
    if lead:
        assert out.iloc[:lead].isna().all()
    if trail:
        assert out.iloc[-trail:].isna().all()


def test_interpolation_never_invents_a_value_outside_the_observed_range():
    from aq.data import _interpolate_short_gaps

    series, spans = _series_with_gaps([2])
    out = _interpolate_short_gaps(series, cfg.MAX_GAP)
    observed = series.dropna()
    assert out.dropna().min() >= observed.min() - 1e-9
    assert out.dropna().max() <= observed.max() + 1e-9


def test_nan_runs_identifies_interior_correctly():
    from aq.data import nan_runs

    mask = np.array([True, False, True, True, False, False, True])
    assert nan_runs(mask) == [(0, 1, False), (2, 4, True), (6, 7, False)]


# ---------------------------------------------------------------------------
# FINDINGS 3 -- protocol
# ---------------------------------------------------------------------------
def test_source_frames_reject_a_leaking_cutoff():
    daily = pd.DataFrame(
        {
            "sheet": ["station_8"] * 5,
            "date": pd.date_range("2022-12-01", periods=5, freq="D"),
            "pm25": np.arange(5, dtype=float),
        }
    )
    first = pd.Timestamp("2023-01-01")
    with pytest.raises(LeakageError, match="strictly before"):
        source_frames(daily, train_end=first, target_first=first)
    with pytest.raises(LeakageError):
        source_frames(daily, train_end=first + pd.Timedelta(days=1), target_first=first)


def test_manifest_train_end_precedes_every_target_first_observation():
    from aq.data import load_daily, station_metadata

    daily = load_daily()
    meta = station_metadata(daily)
    manifest = build_manifest(daily, meta)
    assert len(manifest) > 0
    for _, row in manifest.iterrows():
        assert pd.Timestamp(row.train_end) < pd.Timestamp(row.adapt_start)


@pytest.mark.parametrize("k", [0, 1, 7, 13, 14, 29])
def test_adaptation_rejects_k_below_the_lag_window(k):
    """K < 14 gives ZERO adaptation sequences; K = 14 also gives zero."""
    with pytest.raises(ValueError, match="zero adaptation sequences"):
        require_adaptation_k(k)


@pytest.mark.parametrize("k", [30, 90])
def test_adaptation_accepts_valid_k(k):
    require_adaptation_k(k)


# ---------------------------------------------------------------------------
# FINDINGS 6.1 -- LSTM target standardisation
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_lstm_predicts_in_raw_scale_and_does_not_collapse_to_the_mean():
    """The 6.1 bug made the net converge on a large bias and underfit badly.

    A model fit on targets with mean ~39 must predict on that scale AND must
    track the signal, not emit a constant near the mean.
    """
    torch = pytest.importorskip("torch")
    from aq.models import LSTMBackbone

    rng = np.random.default_rng(cfg.SEED)
    n = 600
    trend = np.linspace(0, 4 * np.pi, n)
    level = 39.0 + 25.0 * np.sin(trend)  # mean ~39, like real PM2.5
    X = np.zeros((n, cfg.SEQ_L, 5), dtype=float)
    for i in range(n):
        X[i, :, 0] = level[i] + rng.normal(0, 0.5, cfg.SEQ_L)
    y = level + rng.normal(0, 0.5, n)

    model = LSTMBackbone().fit_sources(X, y, epochs=25, patience=25)
    predicted = model.predict(X)

    assert model.y_sd > 1.0, "targets were not standardised"
    assert 20.0 < predicted.mean() < 60.0, "predictions are not in raw PM2.5 scale"
    assert predicted.std() > 0.25 * y.std(), "predictions collapsed toward the mean"


# ---------------------------------------------------------------------------
# FINDINGS 6.2 -- seasonal routing
# ---------------------------------------------------------------------------
def test_season_labels():
    assert season_of(pd.to_datetime(["2023-01-15"]))[0] == "cool"
    assert season_of(pd.to_datetime(["2023-04-30"]))[0] == "cool"
    assert season_of(pd.to_datetime(["2023-07-15"]))[0] == "hot"
    assert season_of(pd.to_datetime(["2023-10-01"]))[0] == "cool"


def test_seasonal_router_routes_by_day_not_by_output_length():
    """The original returned whichever model produced more predictions."""

    class Stub:
        def __init__(self, value):
            self.value = value

        def fit_sources(self, X, y, sample_weight=None):
            return self

        def predict(self, X):
            return np.full(len(X), self.value)

    router = SeasonalRouter(factory=lambda: Stub(0.0))
    router.models = {"cool": Stub(-1.0), "hot": Stub(1.0)}

    dates = pd.to_datetime(
        ["2023-01-10", "2023-07-10", "2023-02-10", "2023-08-10", "2023-11-10"]
    )
    out = router.predict(np.zeros((len(dates), 3)), dates)
    np.testing.assert_array_equal(out, [-1.0, 1.0, -1.0, 1.0, -1.0])


def test_seasonal_router_falls_back_when_one_season_is_missing():
    class Stub:
        def predict(self, X):
            return np.full(len(X), 7.0)

    router = SeasonalRouter(factory=lambda: None)
    router.models = {"cool": Stub()}
    dates = pd.to_datetime(["2023-01-10", "2023-07-10"])
    np.testing.assert_array_equal(router.predict(np.zeros((2, 3)), dates), [7.0, 7.0])


# ---------------------------------------------------------------------------
# slice reporting
# ---------------------------------------------------------------------------
def test_onset_is_extreme_today_but_not_yesterday():
    y = np.array([10.0, 80.0, 90.0, 10.0, 100.0])
    masks = slice_masks(y)
    np.testing.assert_array_equal(masks["ext"], [False, True, True, False, True])
    np.testing.assert_array_equal(masks["onset"], [False, True, False, False, True])


def test_small_slices_report_nan_rather_than_a_number():
    """The >=5-day rule is why the >=75 slice covers 9 of 12 targets."""
    index = pd.date_range("2023-01-01", periods=20, freq="D")
    y = np.full(20, 10.0)
    y[:2] = 200.0  # only two extreme days -> below MIN_SLICE_DAYS
    predictions = {"m": Prediction(np.full(20, 12.0), index)}
    out = score(y, predictions, index)
    assert out["n_ext"] == 2
    assert np.isnan(out["rmse_ext_m"])
    assert np.isfinite(out["rmse_all_m"])


def test_no_module_hardcodes_an_absolute_path():
    """config.py owns every path; that is what makes the repo portable."""
    import pathlib

    package = pathlib.Path(__file__).resolve().parent.parent / "aq"
    offenders = []
    for path in package.glob("*.py"):
        if path.name == "config.py":
            continue
        text = path.read_text()
        for marker in ("/mnt/", "/home/claude", "C:\\", "/Users/"):
            if marker in text:
                offenders.append(f"{path.name}: {marker}")
    assert not offenders, f"absolute paths leaked outside config.py: {offenders}"
