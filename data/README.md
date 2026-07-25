# Data

The raw station workbook is **not** included in this repository. It holds
government air-quality monitoring records for the Emirate of Abu Dhabi, and
redistribution has to be cleared separately — so we keep it out of the public
tree rather than assume permission.

You do **not** need it to read the findings. Every result table the paper cites
is committed under [`../outputs/`](../outputs/), so you can inspect the numbers,
re-run the statistics on them, and follow the analysis without the raw data.

## If you have the workbook

Drop it here as `Abu_Dhabi_stations.xlsx`, or point the code at it:

```bash
export AQ_DATA_FILE=/path/to/Abu_Dhabi_stations.xlsx
```

Then `python run.py --stage all` (and the `scripts/` analyses) will run end to end.

## Expected format

One worksheet per station. The loader reads six columns, by position:

| # | column          | notes                                    |
|---|-----------------|------------------------------------------|
| 0 | `ts`            | timestamp; date is normalised, time dropped |
| 1 | `Station_name`  | free text                                |
| 2 | `city`          | unused by the pipeline                   |
| 3 | `Longitude`     | decimal degrees                          |
| 4 | `Latitude`      | decimal degrees                          |
| 5 | `PM2.5`         | µg/m³; non-numeric → missing             |

The workbook behind the paper has **27 sheets**, daily PM2.5 from
**2017-02-17 to 2024-11-07**, **25,577** non-null observations. Column positions
(not names) are what matter — they live in `aq/config.py` (`COL_*`) if your export
differs.

## What the loader does with it

`aq/data.py` reindexes each station onto a complete daily grid and linearly
interpolates interior gaps of **≤ 3 days** (`MAX_GAP`); longer gaps and anything
touching an endpoint stay missing. That fills 308 station-days (~1.2 %); the
`scripts/leakfree_sensitivity.py` analysis shows every conclusion survives with
interpolation switched off.
