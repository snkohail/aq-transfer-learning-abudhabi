"""The single design-matrix builder.

Replaces four separate implementations of the same tabular design
(stage4.build_design, stage5.build_design, stage5b.design, and the `lag_xy`
pair in stage2b/verify_neighbors) plus stage1's `make_lag_xy`.

## Feature layout

Temporal (always present), 18 columns:

    [0:14]   target PM2.5 at t-14 .. t-1, oldest first
             -> column 13 is lag-1, which is also the persistence prediction
    [14:18]  calendar: sin/cos day-of-week, sin/cos day-of-year

Spatial (only when `neighbours` is given), 2k further columns:

    [18:18+k]      each neighbour's value at t-1, nearest neighbour first
    [18+k:18+2k]   each neighbour's (t-1) - (t-2) delta, the "front" signal

All features are strictly past-only with respect to the day being predicted.

## Why this returns its index, and why you must not slice it yourself

Requesting neighbours drops every day on which any neighbour reading is
missing (see `_neighbour_block`). So the temporal and spatial variants of the
same station legitimately produce DIFFERENT numbers of rows -- roughly 331
versus 140 in this dataset.

That asymmetry is not a bug and cannot be designed away: a spatial model
genuinely cannot predict a day whose neighbour inputs do not exist. The bug
that cost this project its headline result (FINDINGS.md Section 5.5) was
scoring the two variants on their own row sets and comparing the answers.

Note that both variants already came from ONE shared function with a `spatial`
flag, and that did not prevent it. Sharing the builder is not the safeguard.
The safeguard is that `Design.index` is the only description of which days a
model can speak to, and `aq.evaluate` is the only module allowed to turn
indices into an evaluation mask. Never write `X[index >= some_date]` in a
stage; call `aq.evaluate.build_common_index()` instead.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import config as cfg

__all__ = ["Design", "build_design", "build_sequences", "N_TEMPORAL_FEATURES"]

N_TEMPORAL_FEATURES = cfg.SEQ_L + 4
PERSISTENCE_COL = cfg.SEQ_L - 1  # lag-1 lives here; used as the persistence baseline


class Design(NamedTuple):
    """A design matrix together with the days it can actually speak to.

    Attributes:
        X: (n, n_features) float array
        y: (n,) float array, the value on the predicted day
        index: DatetimeIndex of length n, the predicted days
        spatial: whether neighbour features are present
        n_neighbours: how many neighbours contributed
    """

    X: np.ndarray
    y: np.ndarray
    index: pd.DatetimeIndex
    spatial: bool
    n_neighbours: int

    def __len__(self) -> int:
        return len(self.y)

    @property
    def persistence(self) -> np.ndarray:
        """The lag-1 baseline for these rows, read from the design itself.

        Taking it from the design rather than recomputing it guarantees the
        baseline is scored on exactly the rows the model is scored on.
        """
        return self.X[:, PERSISTENCE_COL]


def _empty(n_features: int, spatial: bool, n_neighbours: int) -> Design:
    return Design(
        X=np.empty((0, n_features)),
        y=np.array([]),
        index=pd.DatetimeIndex([]),
        spatial=spatial,
        n_neighbours=n_neighbours,
    )


def _calendar(days: pd.DatetimeIndex) -> np.ndarray:
    return np.column_stack(
        [
            np.sin(2 * np.pi * days.dayofweek / 7),
            np.cos(2 * np.pi * days.dayofweek / 7),
            np.sin(2 * np.pi * days.dayofyear / 366),
            np.cos(2 * np.pi * days.dayofyear / 366),
        ]
    )


def _neighbour_block(
    days: pd.DatetimeIndex,
    neighbours: Sequence[tuple[str, float]],
    wide1: pd.DataFrame,
    wide2: pd.DataFrame,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neighbour values at t-1 and their t-1 minus t-2 deltas.

    Returns (values, deltas, valid). `valid` is False on any day where a
    neighbour's t-1 reading is missing -- THIS is where the two variants of a
    station diverge in row count. A missing t-2 only zeroes the delta; it does
    not drop the row, because the level is the informative part.
    """
    columns = [name for name, _ in neighbours]
    values = wide1.reindex(days)[columns].to_numpy(dtype=float)
    previous = wide2.reindex(days)[columns].to_numpy(dtype=float)

    deltas = values - previous
    deltas = np.where(np.isnan(deltas), 0.0, deltas)

    if weights is not None:
        values = values * weights
        deltas = deltas * weights

    valid = ~np.isnan(values).any(axis=1)
    return values, deltas, valid


def build_design(
    series: pd.DataFrame,
    *,
    neighbours: Sequence[tuple[str, float]] | None = None,
    wide1: pd.DataFrame | None = None,
    wide2: pd.DataFrame | None = None,
    decay_tau: float | None = None,
) -> Design:
    """Build the design matrix for one station.

    Args:
        series: DataFrame with `date` and `pm25` columns, one station, daily grid.
        neighbours: [(sheet, distance_km)] nearest-first, or None/[] for the
            temporal-only variant.
        wide1, wide2: the (t-1, t-2) station matrices from
            `aq.data.neighbour_lag_frames`. Required when `neighbours` is given.
        decay_tau: if set, weight each neighbour block by exp(-distance / tau).
            Note this is mathematically a no-op at k=1, where a single weight
            normalises away (FINDINGS.md Section 5.4).

    Returns:
        A `Design`. Read `.index` to learn which days it covers; do not slice
        it by date yourself.
    """
    neighbours = list(neighbours or [])
    spatial = bool(neighbours)
    n_features = N_TEMPORAL_FEATURES + 2 * len(neighbours)

    if spatial and (wide1 is None or wide2 is None):
        raise ValueError("wide1 and wide2 are required when neighbours are requested")

    frame = series.sort_values("date").reset_index(drop=True)
    pm = frame["pm25"].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(frame["date"])

    if len(frame) <= cfg.SEQ_L:
        return _empty(n_features, spatial, len(neighbours))

    windows = sliding_window_view(pm, cfg.SEQ_L)[: len(frame) - cfg.SEQ_L]
    y = pm[cfg.SEQ_L :]
    days = dates[cfg.SEQ_L :]

    valid = (~np.isnan(windows).any(axis=1)) & (~np.isnan(y))
    blocks = [windows, _calendar(days)]

    if spatial:
        weights = None
        if decay_tau is not None:
            distances = np.array([dist for _, dist in neighbours], dtype=float)
            weights = np.exp(-distances / decay_tau)
        values, deltas, neighbour_valid = _neighbour_block(
            days, neighbours, wide1, wide2, weights
        )
        valid &= neighbour_valid
        blocks += [values, deltas]

    X = np.hstack(blocks)
    return Design(
        X=X[valid],
        y=y[valid],
        index=days[valid],
        spatial=spatial,
        n_neighbours=len(neighbours),
    )


def build_sequences(series: pd.DataFrame) -> Design:
    """3-D sequences for the recurrent backbone: (n, SEQ_L, 5).

    Per timestep the channels are [pm25, dow_sin, dow_cos, doy_sin, doy_cos].
    Shares the validity rule with `build_design`, so a day usable by the ridge
    model is usable by the LSTM and vice versa.

    `X` is 3-D here, so `Design.persistence` does not apply; the lag-1 value is
    `X[:, -1, 0]`.
    """
    frame = series.sort_values("date").reset_index(drop=True)
    pm = frame["pm25"].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(frame["date"])

    if len(frame) <= cfg.SEQ_L:
        return Design(
            X=np.empty((0, cfg.SEQ_L, 5)),
            y=np.array([]),
            index=pd.DatetimeIndex([]),
            spatial=False,
            n_neighbours=0,
        )

    channels = np.column_stack([pm, _calendar(dates)])
    windows = sliding_window_view(channels, (cfg.SEQ_L, 5)).squeeze(axis=1)
    windows = windows[: len(frame) - cfg.SEQ_L]
    y = pm[cfg.SEQ_L :]
    days = dates[cfg.SEQ_L :]

    valid = (~np.isnan(windows[:, :, 0]).any(axis=1)) & (~np.isnan(y))
    return Design(
        X=windows[valid],
        y=y[valid],
        index=days[valid],
        spatial=False,
        n_neighbours=0,
    )
