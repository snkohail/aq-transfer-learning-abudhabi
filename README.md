# Transfer Learning for PM2.5 at Newly Deployed Air-Quality Stations

[![tests](https://github.com/snkohail/pm25-transfer-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/snkohail/pm25-transfer-learning/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

When a brand-new air-quality sensor comes online it has no history of its own, yet
it's expected to forecast tomorrow's pollution straight away. The usual fix is to
*transfer* a model trained on older, data-rich stations. This repository asks, as
honestly as we could manage, whether that actually works — for next-day PM2.5 at
**genuinely new** 2023+ deployments in Abu Dhabi, under a protocol that never lets
a model peek at the future.

**The headline is a negative result, and that's the point.** Pooled multi-source
transfer beats a plain persistence baseline at **all 12** newly deployed stations
(+5.8 %, *p* = 0.0005) with no local history required — but the gain comes
**entirely from ordinary days**. On the extreme-pollution days that public health
actually cares about, the same models are **15–20 % worse than persistence** and
miss **~72 % of exceedance events**. Nothing we tried fixed it: not an LSTM with
attention, not gradient boosting, not adaptation data, not similarity-based source
selection, not spatial neighbour features. The lesson is methodological — aggregate
RMSE quietly hides tail failure in air-quality transfer learning.

| Test slice | n targets | persistence | transfer (ridge) | change | *p* |
|---|---:|---:|---:|---:|---:|
| overall | 12 | 12.405 | 11.685 | **+5.8 %** | 0.0005 |
| ordinary days (< 75) | 12 | 11.821 | 10.589 | **+10.4 %** | 0.0005 |
| extreme (≥ 75 µg/m³) | 9 | 26.854 | 32.374 | **−20.6 %** | 0.0039 |
| top 10 % | 12 | 19.139 | 22.128 | **−15.6 %** | 0.0010 |
| onset days | 7 | 33.740 | 36.376 | **−7.8 %** | 0.0156 |

*(RMSE in µg/m³, K = 30 days of local history. "change" is positive when transfer
helps. Scores also carry MAE alongside RMSE — see [`outputs/`](outputs/).)*

## Just want the findings?

You don't have to run anything. The result tables the paper cites are committed
under [`outputs/`](outputs/) as plain CSVs, so you can open them in a spreadsheet
and check the numbers yourself. The narrative — including **everything that was
refuted** and the four bugs that nearly made it into the paper — is in
[`FINDINGS.md`](FINDINGS.md), and the full method, precise enough to reimplement
from the text, is in [`SPEC.md`](SPEC.md).

## Getting started

You need Python 3.11 or newer. The analysis runs on NumPy / pandas / SciPy /
scikit-learn; only the LSTM stage needs PyTorch, which is an optional extra.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[lstm,dev]"     # drop [lstm] to skip torch; or: pip install -r requirements.txt
pytest -m "not slow"             # ~60 fast checks, a few seconds
```

If you have [`make`](Makefile), `make setup` does the install and `make test`
runs the checks; `make help` lists the rest.

Reproducing the tables needs the station workbook, which **isn't** in the repo (see
[Data](#data) below). Once it's in place:

```bash
python run.py --stage all        # rebuild every result table into outputs/
make supplementary               # the extra robustness analyses in scripts/
```

## What you can run

Each stage is self-contained: it builds its models, scores every method on one
shared evaluation index, and writes a CSV to `outputs/`.

| `python run.py --stage …` | Produces | Runtime |
|---|---|---|
| `audit` | corpus facts, station sets, split manifest | ~5 s |
| `selection` | source selection vs uniform pooling (finds no benefit) | ~10 s |
| `baselines` | target-only local models vs transfer | ~5 s |
| `ridge_alpha` | ridge-penalty sensitivity of transfer | ~5 s |
| `ablation` | neighbour-feature ablation on a common index | ~15 s |
| `extreme` | slice decomposition + event-warning skill | ~5 s |
| `lstm` | LSTM+attention vs ridge vs persistence (needs torch) | ~4 min |
| `all` | everything above | ~5 min |

`python sweep_lstm.py` runs the leak-free 9-configuration LSTM sweep; `--seeds`
adds the 3-seed replication (~80 min); `--analyze` / `--analyze-seeds` reproduce
the reports from the committed CSVs without retraining.

The reviewer-response analyses (leak-free sensitivity, extra baselines, a station
table, a gradient-boosting baseline, and a dependence-aware bootstrap) live in
[`scripts/`](scripts/) — see [`scripts/README.md`](scripts/README.md).

## How the repo is laid out

```
aq/                the pipeline package
  config.py          all paths and constants — nothing else hardcodes either
  data.py            one Excel loader (daily grid, ≤3-day gap interpolation)
  splits.py          the leak-free protocol; asserts sources end before deployment
  features.py        one design-matrix builder (temporal + spatial variants)
  evaluate.py        build_common_index / align / score — the guard rail
  models.py          ridge + LSTM behind one interface; y-standardisation internal
  stats.py           paired Wilcoxon helpers (n, win-rate, paired effect per slice)
  stages.py          the analyses invoked by run.py
run.py               command-line entry point
sweep_lstm.py        LSTM hyperparameter + seed sweep
scripts/             supplementary reviewer-response analyses
tests/               invariants + regression (reproduce the headline numbers)
outputs/             result tables the paper cites — regenerate with run.py
data/                where the station workbook goes (not shipped; see data/README.md)
SPEC.md              methods, precise enough to reimplement from the text
FINDINGS.md          verified & refuted findings + the four bugs and their guards
```

## The one rule the code enforces

Three separate results in this project were once invalidated by the same mistake:
scoring two methods on different sets of days. So **no method derives its own
evaluation mask.** You build one common index across every arm of a comparison
first, then score every arm on exactly it:

```python
index  = build_common_index(*designs, not_before=test_start)   # intersect FIRST
X, y   = align(design, index)                                  # then align
scores = score(y_true, predictions, index)                     # raises on any mismatch
```

`evaluate.score()` **raises**, not warns, if any method is indexed on different
days. That guard, a leak assertion in `splits.py`, and the LSTM's internal target
standardisation are what keep the pipeline honest — each has a test in `tests/`.

## Reproducibility

The deterministic (ridge / persistence) stages reproduced across a different
platform and stack (Linux/Py 3.12 → macOS/Py 3.11; NumPy 1.x → 2.4; pandas 2.x →
3.0) to a maximum relative deviation of **3 × 10⁻¹⁴** — floating-point
reassociation, not a behavioural difference. The regression tests gate those
stages at `rtol = 1e-12`, and gate the stochastic LSTM comparison on the
*decision* (win-rate and significance) rather than individual decimals, because
bitwise equality isn't a fair cross-platform bar for a torch model. See
[`SPEC.md`](SPEC.md) §8.3.

## Data

`data/Abu_Dhabi_stations.xlsx` — 27 station sheets, daily PM2.5,
2017-02-17 → 2024-11-07, 25,577 non-null observations will be released once the paper is accepted for publication.
The committed tables in `outputs/` let you follow
and check every result without it. If you do have the workbook, drop it in `data/`
(or set `$AQ_DATA_FILE`); [`data/README.md`](data/README.md) has the expected
format. Tests and scripts that need it skip themselves cleanly when it's absent.

## Caveats to carry into any write-up

n = 12 target stations (underpowered below ~1 % effects — say "no evidence of
benefit", not "proof of no benefit"); slice-dependent n from a ≥5-qualifying-days
reporting rule; a single city, a single pollutant, a next-day horizon only; and no
meteorological covariates — wind is the obvious omission for dust events and a
plausible partial explanation for the extreme-day failure.

## Citing

If this is useful, please cite this repo. 

## Contributing

Issues and pull requests are welcome. If you touch a stage, keep the guard rail
intact and run `pytest` — the regression tests exist to catch exactly the kind of
silent number-drift that this project was built to avoid.

## License

[MIT](LICENSE).
