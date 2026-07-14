"""Smoke: exp_112 runner importable (synthesis; no GPU training)."""

from experiments.exp_112_cycle3_image_grand_leaderboard.run import gate_passed, run_exp_112


def test_exp_112_import():
    assert callable(run_exp_112)
    assert callable(gate_passed)
