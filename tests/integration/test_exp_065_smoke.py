"""Smoke test — exp_065 ACYD GV-ALR hybrid imports."""

from experiments.exp_065_gv_alr_hybrid_acyd.run import gate_passed, run_exp_065


def test_exp_065_import():
    assert callable(run_exp_065)
    assert callable(gate_passed)
