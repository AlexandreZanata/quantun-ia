"""Smoke: exp_085 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_085_sample_efficiency_agro.run import run_exp_085


def test_exp_085_import():
    assert callable(run_exp_085)
