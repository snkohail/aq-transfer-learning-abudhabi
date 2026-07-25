#!/usr/bin/env python3
"""Pipeline entry point.

    python run.py --stage audit        data facts, station sets, manifest
    python run.py --stage selection    source selection vs uniform pooling
    python run.py --stage baselines    target-only local models vs transfer
    python run.py --stage ridge_alpha  ridge penalty sensitivity of transfer
    python run.py --stage ablation     neighbour ablation on a common index
    python run.py --stage extreme      extreme-event decomposition + warning skill
    python run.py --stage lstm         LSTM+attention (needs torch, ~5 min)
    python run.py --stage all          everything

The analyses themselves live in aq/stages.py; this file only parses arguments
and dispatches. Result tables are written to `outputs/`; the committed copies
there are the tables the paper cites.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from aq import config as cfg
from aq.stages import STAGES, log


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args(argv)

    out_dir = cfg.OUTPUT_DIR if args.output_dir is None else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = list(STAGES) if args.stage == "all" else [args.stage]
    started = time.time()
    for name in chosen:
        STAGES[name](out_dir)
    log(f"\ndone in {time.time() - started:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
