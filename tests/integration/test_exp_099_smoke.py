"""Smoke: exp_099 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_099_ssl_climate_pretrain.run import run_exp_099


def test_exp_099_import():
    assert callable(run_exp_099)
