"""Paired statistics that always report n, win-rate and a per-slice p-value.

Reporting one p-value for a multi-slice table is how FINDINGS.md Section 5.5
originally read as "null" when the overall slice was in fact significant
(k=1 hurts overall RMSE at p=0.0098 while its extreme-slice effect is null at
p=0.43). `compare` therefore returns one record per comparison, and
`compare_slices` returns one record per slice.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

__all__ = ["Comparison", "compare", "compare_slices", "summarise"]


class Comparison(NamedTuple):
    """One paired comparison. Lower is better for both arms (these are errors).

    IMPORTANT: `improvement_pct` is a ratio of the two COLUMN medians -- a
    difference of medians. `median_diff` is the median of the PAIRED differences.
    They usually agree, but a skewed loss distribution can pull them apart: at
    K=90 the target-only local model has a higher column median than persistence
    (favouring persistence) yet wins the majority of paired comparisons and has a
    negative median paired difference (favouring local). For paired data the
    paired statistics -- `median_diff`, `b_wins`, `p_value` -- are the correct
    ones. Read `improvement_pct` only when the win-rate is decisive.
    """

    label: str
    n: int
    median_a: float
    median_b: float
    improvement_pct: float  # difference of column medians; positive nominally favours b
    median_diff: float  # median of paired (a - b); positive means b lower => b better
    b_wins: int
    p_value: float

    def as_dict(self) -> dict:
        return self._asdict()


def compare(a, b, label: str = "") -> Comparison:
    """Wilcoxon signed-rank on paired errors, dropping pairs with a NaN.

    Args:
        a: baseline errors (e.g. persistence RMSE per target)
        b: candidate errors
        label: what to call this comparison in the output

    Returns:
        A Comparison. `p_value` is NaN when fewer than three usable pairs
        remain or the test degenerates -- never silently zero.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must match: {a.shape} vs {b.shape}")

    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    n = len(a)
    if n == 0:
        return Comparison(label, 0, np.nan, np.nan, np.nan, np.nan, 0, np.nan)

    median_a, median_b = float(np.median(a)), float(np.median(b))
    improvement = 100.0 * (1.0 - median_b / median_a) if median_a else np.nan

    p_value = np.nan
    if n >= 3:
        try:
            p_value = float(wilcoxon(a, b).pvalue)
        except ValueError:
            p_value = np.nan

    return Comparison(
        label=label,
        n=n,
        median_a=median_a,
        median_b=median_b,
        improvement_pct=improvement,
        median_diff=float(np.median(a - b)),
        b_wins=int((b < a).sum()),
        p_value=p_value,
    )


def compare_slices(
    frame: pd.DataFrame,
    baseline_prefix: str,
    candidate_prefix: str,
    slices: Sequence[str] = ("all", "normal", "ext", "extrel", "onset"),
    metric: str = "rmse",
) -> pd.DataFrame:
    """One row per slice, each with its own n, win-rate and p-value.

    Expects columns named `{metric}_{slice}_{prefix}`, which is what
    `aq.evaluate.score` produces.
    """
    rows = []
    for slice_name in slices:
        col_a = f"{metric}_{slice_name}_{baseline_prefix}"
        col_b = f"{metric}_{slice_name}_{candidate_prefix}"
        if col_a not in frame or col_b not in frame:
            continue
        rows.append(compare(frame[col_a], frame[col_b], label=slice_name).as_dict())
    return pd.DataFrame(rows)


def summarise(comparisons: Mapping[str, Comparison] | Sequence[Comparison]) -> str:
    """Fixed-width text table, for stage logs."""
    items = (
        list(comparisons.values())
        if isinstance(comparisons, Mapping)
        else list(comparisons)
    )
    width = max([12] + [len(item.label) for item in items]) + 2
    # Show BOTH the difference-of-medians (gain%) and the paired median diff
    # (dPair). When they disagree in sign the distribution is skewed and the
    # paired columns (dPair, wins, p) are the ones to trust.
    header = (
        f"{'comparison':<{width}}{'n':>4}{'baseline':>11}{'model':>10}"
        f"{'gain%':>8}{'dPair':>9}{'wins':>8}{'p':>10}"
    )
    lines = [header, "-" * len(header)]
    for item in items:
        wins = f"{item.b_wins}/{item.n}" if item.n else "-"
        p = f"{item.p_value:.4f}" if np.isfinite(item.p_value) else "-"
        lines.append(
            f"{item.label:<{width}}{item.n:>4}{item.median_a:>11.3f}{item.median_b:>10.3f}"
            f"{item.improvement_pct:>7.1f}%{item.median_diff:>+9.3f}{wins:>8}{p:>10}"
        )
    return "\n".join(lines)
