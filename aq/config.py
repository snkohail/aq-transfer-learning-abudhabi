"""Central configuration.

Every constant and every path lives here. No other module in `aq` may hardcode
an absolute path or a magic number -- that rule is what makes the pipeline
portable between machines, and it is checked by tests/test_invariants.py.

Paths can be overridden with environment variables so the same code runs from a
clone, a CI job, or a scratch directory without edits:

    AQ_DATA_FILE    full path to the station workbook
    AQ_OUTPUT_DIR   where freshly produced CSVs are written
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("AQ_DATA_DIR", REPO_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("AQ_OUTPUT_DIR", REPO_ROOT / "outputs"))

DATA_FILE = Path(os.environ.get("AQ_DATA_FILE", DATA_DIR / "Abu_Dhabi_stations.xlsx"))

# Column positions in each worksheet: ts, Station_name, city, Longitude, Latitude, PM2.5
COL_TS, COL_NAME, COL_CITY, COL_LON, COL_LAT, COL_PM = 0, 1, 2, 3, 4, 5

# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------
SEED = 42

# Seeds for estimating LSTM training stochasticity in the sweep. A single run
# per config cannot separate the capacity signal from training noise, so the
# seed sweep (sweep_lstm.py --seeds) retrains each config under each of these.
SEEDS = (42, 43, 44)

# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------
PM_CAP = 500.0  # clip implausible spikes (observed max 491.75)
MAX_GAP = 3  # interpolate interior gaps of at most this many days

# Default for load_daily(interpolate=...). Interpolation fills interior NaN runs
# of <= MAX_GAP days (308 station-days, ~1.2% of the corpus) BEFORE the
# source/target split. A no-interpolation run -- load_daily(interpolate=False),
# equivalently setting this True -- leaves every gap NaN so the 14-day
# window-validity rule drops any window that would have leaned on a filled day.
# It exists only for the leak-free sensitivity analysis; the committed tables use
# interpolation (this flag False), and every existing call keeps that default.
DISABLE_INTERPOLATION = False

# --------------------------------------------------------------------------
# protocol (FINDINGS.md Section 3)
# --------------------------------------------------------------------------
SEQ_L = 14  # lag window; the first predictable day of a station's life is day 15
MIN_TEST = 60  # a target needs at least this many usable test days
K_ADAPT_DEFAULT = 30  # adaptation window used for target selection
K_LIST = (30, 90)  # adaptation windows evaluated

# K < SEQ_L yields ZERO usable adaptation sequences, so any path that fits
# something on the adaptation window (fine-tuning, calibration) must refuse a
# smaller K. K=14 gives zero sequences; K=30 gives roughly 16.
MIN_K_FOR_ADAPTATION = 30

TARGET_START = "2023-01-01"  # a target is a station first observed on/after this

RICH_SOURCES = (
    "station_8",
    "station_6",
    "station_2",
    "station_4",
    "station_3",
    "station_7",
    "station_25",
    "station_1",
    "station_5",
)
MIN_SOURCES = 5  # refuse to pool fewer than this many source stations
MIN_SOURCE_ROWS = 100  # a source contributes only if it has more rows than this

# --------------------------------------------------------------------------
# evaluation (FINDINGS.md Section 4.3)
# --------------------------------------------------------------------------
EXTREME_ABS = 75.0  # absolute "unhealthy" threshold, ug/m3
EXTREME_PCT = 90  # per-station relative threshold, percentile
MIN_SLICE_DAYS = 5  # a slice RMSE is only reported with at least this many days

# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
RIDGE_ALPHA = 1.0
GEO_TAU = 25.0  # km, exponential decay for geographic similarity weighting

LSTM_HIDDEN = 64
LSTM_ATTENTION = 32
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 40
LSTM_PATIENCE = 6
LSTM_BATCH = 128
LSTM_LR = 1e-3
LSTM_MAX_SOURCE_SEQ = 20000  # subsample cap, keeps a full run feasible on 1 CPU
LSTM_VAL_FRACTION = 0.1

FT_HEAD_EPOCHS = 20
FT_HEAD_LR = 1e-3
FT_FULL_EPOCHS = 5
FT_FULL_LR = 1e-4

# --------------------------------------------------------------------------
# regression tolerance
# --------------------------------------------------------------------------
# The deterministic (ridge / persistence) stages reproduce across platforms to
# a relative error of ~3e-14, which is BLAS round-off rather than a behavioural
# difference. Measured on macOS/py3.11/numpy 2.4 against results produced on
# Linux/py3.12. Bitwise equality is NOT a valid gate; this is.
REGRESSION_RTOL = 1e-12

# The LSTM stage trains with torch and does not reproduce bitwise across torch
# versions or platforms. Gate its comparisons on the decision, not the decimals.
LSTM_MIN_RIDGE_WINS = 9  # of 12 targets
LSTM_MAX_P = 0.05
