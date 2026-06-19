"""Smoke test — exp_075 GV-ALR hybrid NIHR imports."""

from experiments.exp_075_adaptive_hybrid_nihr.run import gate_passed, run_exp_075


def test_exp_075_import():
    assert callable(run_exp_075)
    assert callable(gate_passed)
