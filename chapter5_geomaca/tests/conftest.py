"""Shared fixtures for the chapter5_geomaca test-suite.

The tests run on the CPU. Tests that need the checkpoint archive read its
location from ``CHAPTER5_CHECKPOINTS`` (the folder that contains
``cb_runs_v4/`` and ``cb_runs_v3/``) and are skipped without it; the
Level-Based Foraging smoke test needs ``gymnasium`` and ``lbforaging``.
"""

from __future__ import annotations

import os

# Hide the GPU before torch is imported anywhere in the test session.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def archive() -> Path:
    """Root of the checkpoint archive (or skip)."""
    root = os.environ.get("CHAPTER5_CHECKPOINTS")
    if not root or not (Path(root) / "cb_runs_v4").is_dir():
        pytest.skip("set CHAPTER5_CHECKPOINTS to the checkpoint archive to run this test")
    return Path(root)


@pytest.fixture(scope="session")
def records() -> Path:
    return PACKAGE_ROOT / "records"
