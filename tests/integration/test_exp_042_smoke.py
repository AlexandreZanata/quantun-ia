"""Smoke test — exp_042 sample-scale precision imports."""

from __future__ import annotations

from experiments.exp_042_sample_scale_precision.run import run_exp_042


def test_exp_042_import():
    assert callable(run_exp_042)
