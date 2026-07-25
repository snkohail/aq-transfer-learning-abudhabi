#!/usr/bin/env python3
"""Per-station documentation table for all 27 stations.

One row per station: id, name, lat/lon, first and last valid observation, count
of valid daily observations, missingness over the station's active span, and its
role (source / target / excluded). Every excluded station's reason is COMPUTED
from the data and asserted -- not assumed -- so the table states the criterion a
station actually fails.

    python scripts/station_table.py

Writes outputs/station_table.csv. Needs the station workbook (see data/README.md).

A note the table makes explicit: `RICH_SOURCES` is a curated long-baseline set
(every source has >=1339 valid days before the source cutoff), not a simple
">=100 rows" threshold. So three of the six excluded stations are dropped for not
being curated sources despite having far more than 100 valid days -- the
MIN_SOURCE_ROWS=100 constant governs per-target pooling, not source selection.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from aq import config as cfg
from aq.data import load_daily, station_metadata
from aq.features import build_design
from aq.splits import select_targets, train_end_for

OUT = cfg.OUTPUT_DIR
TS = pd.Timestamp(cfg.TARGET_START)


def main():
    daily = load_daily()                       # committed corpus (drives the roles)
    raw = load_daily(interpolate=False)         # genuine, un-interpolated counts
    meta = station_metadata(daily)
    targets = set(select_targets(daily, meta))
    sources = set(cfg.RICH_SOURCES)
    earliest_first = meta[meta.sheet.isin(targets)]["first"].min()
    src_cutoff = train_end_for(earliest_first)  # day before the earliest deployment

    rows, diag = [], {}
    for _, m in meta.iterrows():
        sheet = m.sheet
        first, last = m["first"], m["last"]
        n_valid = int(len(raw[(raw.sheet == sheet) & raw.pm25.notna()]))
        span = (last - first).days + 1
        obs_i = daily[(daily.sheet == sheet) & daily.pm25.notna()]["date"]
        usable_test = int((obs_i > first + timedelta(days=cfg.K_ADAPT_DEFAULT)).sum()) - cfg.SEQ_L
        pre = daily[(daily.sheet == sheet) & (daily.date <= src_cutoff)]
        pre_valid = int(pre.pm25.notna().sum())
        pre_design = len(build_design(pre[["date", "pm25"]]))
        role = "source" if sheet in sources else "target" if sheet in targets else "excluded"
        diag[sheet] = dict(first=first, last=last, pre2023=first < TS, usable_test=usable_test,
                           pre_valid=pre_valid, pre_design=pre_design)
        rows.append(dict(
            station_id=sheet, name=m["name"], lat=round(float(m.lat), 5), lon=round(float(m.lon), 5),
            first_obs=first.date().isoformat(), last_obs=last.date().isoformat(),
            n_valid_obs=n_valid, active_span_days=span,
            missingness_pct=round(100.0 * (1 - n_valid / span), 1), role=role,
            usable_test_days=usable_test, pre_cutoff_valid_days=pre_valid,
            pre_cutoff_design_rows=pre_design, exclusion_rule="", exclusion_reason=""))

    df = pd.DataFrame(rows)
    src_floor = min(diag[s]["pre_valid"] for s in sources)

    def reason(sheet):
        d = diag[sheet]
        if not d["pre2023"]:
            assert d["usable_test"] < cfg.MIN_TEST, sheet
            return ("too_short_target",
                    f"post-{TS.year} (first {d['first'].date()}): {d['usable_test']} usable test "
                    f"days after K={cfg.K_ADAPT_DEFAULT}+{cfg.SEQ_L}-day lag < MIN_TEST={cfg.MIN_TEST}")
        if d["pre_design"] <= cfg.MIN_SOURCE_ROWS:
            return ("too_sparse_source",
                    f"pre-{TS.year} (first {d['first'].date()}): {d['pre_design']} design rows / "
                    f"{d['pre_valid']} valid days before the {src_cutoff.date()} cutoff "
                    f"<= MIN_SOURCE_ROWS={cfg.MIN_SOURCE_ROWS}")
        assert d["pre_valid"] > cfg.MIN_SOURCE_ROWS, sheet
        why = (f"record ends {d['last'].date()} before any target deployed"
               if d["last"] < earliest_first else
               f"record begins {d['first'].date()} (short baseline)")
        return ("not_curated_rich_source",
                f"pre-{TS.year} but not in RICH_SOURCES; {why}; {d['pre_valid']} valid days before "
                f"cutoff (>> {cfg.MIN_SOURCE_ROWS}) yet far below the source floor of {src_floor} "
                f"-- does NOT fail the <{cfg.MIN_SOURCE_ROWS} rule")

    for i, r in df.iterrows():
        if r.role == "excluded":
            df.at[i, "exclusion_rule"], df.at[i, "exclusion_reason"] = reason(r.station_id)

    df = df.sort_values("station_id", key=lambda s: s.str.split("_").str[1].astype(int))
    df.to_csv(OUT / "station_table.csv", index=False)

    counts = df.role.value_counts().to_dict()
    assert (counts.get("source"), counts.get("target"), counts.get("excluded")) == (9, 12, 6), counts
    print(f"roles reconstruct to 9 source / 12 target / 6 excluded: OK  "
          f"(source valid-day floor {src_floor}, cutoff {src_cutoff.date()})\n")
    for _, r in df[df.role == "excluded"].iterrows():
        print(f"  {r.station_id:<11} [{r.exclusion_rule}] {r.exclusion_reason}")
    curated = list(df[df.exclusion_rule == "not_curated_rich_source"].station_id)
    print(f"\nNote: the '<{cfg.MIN_SOURCE_ROWS} valid days' rule explains only station_11, 23, 24. "
          f"{', '.join(curated)} have 207-730 pre-cutoff valid days yet are excluded because "
          f"RICH_SOURCES is a curated long-baseline set.")
    print(f"\nsaved -> {OUT / 'station_table.csv'}")


if __name__ == "__main__":
    main()
