"""Smoke: exp_090 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_090_multicrop_joint_nano.run import run_exp_090


def test_exp_090_import():
    assert callable(run_exp_090)
