"""Smoke: exp_088 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_088_shadow_features_nano_maize.run import run_exp_088


def test_exp_088_import():
    assert callable(run_exp_088)
