"""Smoke: exp_087 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_087_fourier_reupload_climate.run import run_exp_087


def test_exp_087_import():
    assert callable(run_exp_087)
