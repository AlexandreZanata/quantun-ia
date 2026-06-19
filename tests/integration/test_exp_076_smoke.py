"""Smoke test — exp_076 conventional NIHR baselines."""

from experiments.exp_076_conventional_nihr_baselines.run import gate_passed, run_exp_076


def test_exp_076_import():
    assert callable(run_exp_076)
    assert callable(gate_passed)
