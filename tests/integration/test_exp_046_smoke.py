"""Smoke test — exp_046 model scale curve imports."""

from __future__ import annotations

from experiments.exp_046_model_scale_curve.run import run_exp_046


def test_exp_046_import():
    assert callable(run_exp_046)
