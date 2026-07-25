"""Skip the tests that need the private station workbook when it isn't present.

A fresh clone (or CI) won't have data/Abu_Dhabi_stations.xlsx. Rather than fail
with a FileNotFoundError, skip the handful of tests that read it and still run the
synthetic unit tests, which cover the core guard-rails (common-index alignment,
gap-interpolation semantics, the leak assertion, slice reporting). With the
workbook present, this changes nothing.
"""
from __future__ import annotations

import pytest

from aq import config as cfg

# Everything in test_regression reproduces headline numbers from the real corpus;
# in test_invariants only the manifest test reads the workbook.
_NEEDS_DATA_MODULES = {"test_regression"}
_NEEDS_DATA_TESTS = {"test_manifest_train_end_precedes_every_target_first_observation"}


def pytest_collection_modifyitems(config, items):
    if cfg.DATA_FILE.exists():
        return
    skip = pytest.mark.skip(
        reason=f"station workbook not found at {cfg.DATA_FILE} — see data/README.md"
    )
    for item in items:
        if item.module.__name__ in _NEEDS_DATA_MODULES or item.name in _NEEDS_DATA_TESTS:
            item.add_marker(skip)
