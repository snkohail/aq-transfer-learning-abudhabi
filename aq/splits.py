"""Split construction, with the leak-free protocol enforced in code.

The protocol (FINDINGS.md Section 3), for each target station j:

    sources     all rich stations, data strictly <= (first observation of j - 1 day)
    adaptation  j's first K days
    test        strictly after the adaptation window, at least MIN_TEST usable days

The target-selection loop below appeared, copy-pasted, in five of the seven
original scripts. It is one function now.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from . import config as cfg

__all__ = [
    "LeakageError",
    "train_end_for",
    "select_targets",
    "source_frames",
    "build_manifest",
]


class LeakageError(AssertionError):
    """Raised when a split would let source data reach a target's own history."""


def train_end_for(target_first: pd.Timestamp) -> pd.Timestamp:
    """Last day of source data admissible for a target deployed on `target_first`.

    Strictly the day before deployment. Anything later means the sources have
    seen days the target itself lived through.
    """
    return pd.Timestamp(target_first) - timedelta(days=1)


def _assert_no_leakage(train_end: pd.Timestamp, target_first: pd.Timestamp) -> None:
    if train_end >= pd.Timestamp(target_first):
        raise LeakageError(
            f"source cutoff {train_end.date()} reaches the target's first "
            f"observation {pd.Timestamp(target_first).date()}; sources must end "
            f"strictly before deployment"
        )


def select_targets(
    daily: pd.DataFrame,
    meta: pd.DataFrame,
    k_adapt: int = cfg.K_ADAPT_DEFAULT,
) -> list[str]:
    """Stations that are genuinely new AND have enough usable test days.

    A station qualifies when it is first observed on or after TARGET_START and,
    after reserving `k_adapt` adaptation days and the SEQ_L lag window, at least
    MIN_TEST predictable days remain.
    """
    candidates = meta[meta["first"] >= pd.Timestamp(cfg.TARGET_START)]
    targets: list[str] = []
    for _, row in candidates.iterrows():
        observed = daily[(daily.sheet == row.sheet) & daily.pm25.notna()]["date"]
        usable = (observed > row["first"] + timedelta(days=k_adapt)).sum() - cfg.SEQ_L
        if usable >= cfg.MIN_TEST:
            targets.append(row.sheet)
    return targets


def source_frames(
    daily: pd.DataFrame,
    train_end: pd.Timestamp,
    target_first: pd.Timestamp,
    sources=cfg.RICH_SOURCES,
) -> dict[str, pd.DataFrame]:
    """Per-source `date`/`pm25` frames truncated at `train_end`.

    Asserts the cutoff before returning, so a leaking split fails loudly at the
    point it is constructed rather than silently improving a score later.
    """
    _assert_no_leakage(train_end, target_first)
    frames: dict[str, pd.DataFrame] = {}
    for sheet in sources:
        sub = daily[(daily.sheet == sheet) & (daily.date <= train_end)][
            ["date", "pm25"]
        ]
        if len(sub):
            frames[sheet] = sub
    return frames


def build_manifest(
    daily: pd.DataFrame,
    meta: pd.DataFrame,
    k_values=cfg.K_LIST,
    k_adapt: int = cfg.K_ADAPT_DEFAULT,
) -> pd.DataFrame:
    """One row per (target, K): the split boundaries, for the record.

    Everything downstream can read this instead of recomputing the split, which
    is what stops two stages from disagreeing about where a test window starts.
    """
    targets = select_targets(daily, meta, k_adapt=k_adapt)
    rows = []
    for sheet in targets:
        row = meta[meta.sheet == sheet].iloc[0]
        first = row["first"]
        train_end = train_end_for(first)
        _assert_no_leakage(train_end, first)
        for k in k_values:
            test_start = first + timedelta(days=k)
            observed = daily[(daily.sheet == sheet) & daily.pm25.notna()]["date"]
            rows.append(
                {
                    "target": sheet,
                    "target_name": row["name"],
                    "K": k,
                    "train_end": train_end.date().isoformat(),
                    "adapt_start": first.date().isoformat(),
                    "adapt_end": (test_start - timedelta(days=1)).date().isoformat(),
                    "test_start": test_start.date().isoformat(),
                    "test_end": row["last"].date().isoformat(),
                    "n_observed_after_test_start": int((observed >= test_start).sum()),
                }
            )
    return pd.DataFrame(rows)


def require_adaptation_k(k: int) -> None:
    """Refuse a K too small to yield any adaptation sequences.

    With a SEQ_L-day lag window the first predictable day of a station's life
    is day SEQ_L + 1, so K < SEQ_L produces ZERO adaptation sequences and
    K = SEQ_L produces zero as well. Fine-tuning and calibration therefore
    require K >= MIN_K_FOR_ADAPTATION.
    """
    if k < cfg.MIN_K_FOR_ADAPTATION:
        raise ValueError(
            f"K={k} is too small to adapt on: the {cfg.SEQ_L}-day lag window "
            f"means the first predictable day is day {cfg.SEQ_L + 1}, so K<{cfg.SEQ_L} "
            f"yields zero adaptation sequences and K={cfg.SEQ_L} yields zero. "
            f"Use K >= {cfg.MIN_K_FOR_ADAPTATION} (FINDINGS.md Section 3)."
        )
