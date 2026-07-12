"""Smoke: exp_096 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_096_gobug_streaming_nano.run import run_exp_096


def test_exp_096_import():
    assert callable(run_exp_096)
