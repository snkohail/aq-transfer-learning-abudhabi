"""Evaluation, and the guard rail that makes comparisons trustworthy.

Three separate results in this project were invalidated by scoring two methods
on different sets of days (FINDINGS.md Section 6.3):

  * stage4 scored its spatial model on ~140 days and its temporal model on
    ~331, producing a false +35.6% extreme-event improvement;
  * an external ablation scored persistence on 507 points and models on 508;
  * stage5's neighbour ablation let the test set shrink as k grew.

Every one of those would have been caught here. The rule this module enforces:

    Build ONE evaluation index first, from the intersection of what every
    method can actually predict. Then score every method on exactly it.

`score()` raises on mismatch rather than warning, because a warning in a long
stage log is a finding nobody reads.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd

from . import config as cfg
from .features import Design

__all__ = [
    "Prediction",
    "IndexMismatchError",
    "build_common_index",
    "align",
    "slice_masks",
    "score",
    "rmse",
    "mae",
    "medae",
    "r2",
]


class IndexMismatchError(ValueError):
    """Raised when a method's predictions do not cover the evaluation index."""


class Prediction(NamedTuple):
    """Predictions bound to the days they were made for."""

    values: np.ndarray
    index: pd.DatetimeIndex

    def __len__(self) -> int:
        return len(self.values)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def rmse(actual, predicted) -> float:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual, predicted) -> float:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs(actual - predicted)))


def medae(actual, predicted) -> float:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.median(np.abs(actual - predicted)))


def r2(actual, predicted) -> float:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


_METRICS = {"rmse": rmse, "mae": mae, "medae": medae, "r2": r2}


# --------------------------------------------------------------------------
# the common index
# --------------------------------------------------------------------------
def build_common_index(
    *designs: Design,
    not_before: pd.Timestamp | None = None,
    min_days: int = cfg.MIN_TEST,
) -> pd.DatetimeIndex:
    """Intersect what every method can predict, then apply the test-start cut.

    This is the ONLY sanctioned way to produce an evaluation mask. Stages must
    not derive their own from a single design's `.index` -- that is exactly how
    the spatial-vs-temporal comparison went wrong.

    Args:
        *designs: every design that will take part in the comparison. Pass all
            of them, including the one you think is the "baseline".
        not_before: drop days before this (the test-window start).
        min_days: return an empty index if fewer days survive, so callers can
            skip the target rather than report an underpowered comparison.

    Returns:
        A sorted DatetimeIndex, possibly empty.
    """
    if not designs:
        raise ValueError("build_common_index needs at least one design")

    common: pd.DatetimeIndex | None = None
    for design in designs:
        index = pd.DatetimeIndex(design.index)
        common = index if common is None else common.intersection(index)

    assert common is not None
    if not_before is not None:
        common = common[common >= not_before]

    common = common.sort_values()
    if len(common) < min_days:
        return pd.DatetimeIndex([])
    return common


def align(design: Design, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Select the rows of `design` matching `index`, in `index` order.

    Raises IndexMismatchError if the design cannot cover the index -- an
    explicit failure beats silently scoring on whatever overlapped.
    """
    positions = pd.Index(design.index).get_indexer(pd.DatetimeIndex(index))
    if (positions < 0).any():
        missing = int((positions < 0).sum())
        raise IndexMismatchError(
            f"design is missing {missing} of {len(index)} required evaluation days"
        )
    return design.X[positions], design.y[positions]


# --------------------------------------------------------------------------
# slices
# --------------------------------------------------------------------------
def slice_masks(
    y_true: np.ndarray, relative_threshold: float | None = None
) -> dict[str, np.ndarray]:
    """The four evaluation slices from FINDINGS.md Section 4.3.

      all      every day
      normal   below the absolute extreme threshold
      ext      at or above EXTREME_ABS (75 ug/m3)
      extrel   at or above the per-station relative threshold (top 10%)
      onset    extreme today, not extreme yesterday -- where persistence must fail

    `relative_threshold` should be computed from the target's FULL observed
    series, not from the test window, so it describes the station rather than
    the sample.
    """
    y_true = np.asarray(y_true, float)
    is_extreme = y_true >= cfg.EXTREME_ABS
    previous = np.concatenate([[False], is_extreme[:-1]])

    masks = {
        "all": np.ones_like(is_extreme, dtype=bool),
        "normal": ~is_extreme,
        "ext": is_extreme,
        "onset": is_extreme & (~previous),
    }
    if relative_threshold is not None and np.isfinite(relative_threshold):
        masks["extrel"] = y_true >= relative_threshold
    return masks


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score(
    y_true: np.ndarray,
    predictions: Mapping[str, Prediction],
    index: pd.DatetimeIndex,
    *,
    metrics: Sequence[str] = ("rmse", "mae"),
    relative_threshold: float | None = None,
    min_slice_days: int = cfg.MIN_SLICE_DAYS,
) -> dict[str, float]:
    """Score every method on exactly `index`, or raise.

    MAE is reported alongside RMSE for every slice by default; the two are
    computed on the identical masked rows, so `mae_*` never changes any `rmse_*`
    value. Pass `metrics=("rmse",)` to recover the RMSE-only output.

    Args:
        y_true: observed values, aligned to `index`.
        predictions: {method_name: Prediction}. Every prediction must cover
            `index` exactly -- same length, same days, same order.
        index: the common evaluation index from `build_common_index`.
        metrics: which of rmse / mae / medae / r2 to compute.
        relative_threshold: enables the `extrel` slice.
        min_slice_days: slices with fewer qualifying days report NaN rather
            than a number computed from a handful of points. This is why the
            >=75 slice covers 9 of 12 targets and onset covers 7 -- a reporting
            rule, not an absence of events.

    Returns:
        Flat dict keyed `{metric}_{slice}_{method}`, plus `n_{slice}` counts.

    Raises:
        IndexMismatchError: if any method's index differs from `index`.
    """
    index = pd.DatetimeIndex(index)
    y_true = np.asarray(y_true, float)

    if len(y_true) != len(index):
        raise IndexMismatchError(
            f"y_true has {len(y_true)} rows but the evaluation index has {len(index)}"
        )

    for name, prediction in predictions.items():
        pred_index = pd.DatetimeIndex(prediction.index)
        if len(prediction.values) != len(index):
            raise IndexMismatchError(
                f"method {name!r} produced {len(prediction.values)} predictions "
                f"for an evaluation index of {len(index)} days. Every method must "
                f"be scored on the identical index -- see FINDINGS.md Section 6.3."
            )
        if not pred_index.equals(index):
            n_diff = int((pred_index != index).sum())
            raise IndexMismatchError(
                f"method {name!r} is indexed on different days than the evaluation "
                f"index ({n_diff} differ). Build one index with "
                f"build_common_index() and score every method on it."
            )

    masks = slice_masks(y_true, relative_threshold)
    out: dict[str, float] = {}

    for slice_name, mask in masks.items():
        count = int(mask.sum())
        out[f"n_{slice_name}"] = count
        usable = count >= min_slice_days or slice_name == "all"
        for metric_name in metrics:
            fn = _METRICS[metric_name]
            for method, prediction in predictions.items():
                key = f"{metric_name}_{slice_name}_{method}"
                out[key] = (
                    fn(y_true[mask], np.asarray(prediction.values, float)[mask])
                    if usable and count > 0
                    else float("nan")
                )
    return out
