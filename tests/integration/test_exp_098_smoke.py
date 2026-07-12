"""Smoke: exp_098 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_098_continual_crop_year.run import run_exp_098


def test_exp_098_import():
    assert callable(run_exp_098)
