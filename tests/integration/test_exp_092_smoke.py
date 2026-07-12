"""Smoke: exp_092 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_092_histgb_distill_nano_maize.run import run_exp_092


def test_exp_092_import():
    assert callable(run_exp_092)
