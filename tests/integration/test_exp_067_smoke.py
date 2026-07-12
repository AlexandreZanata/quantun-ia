"""Smoke test — exp_067 ACYD re-upload ladder imports."""

from experiments.exp_067_reupload_ladder_acyd.run import gate_passed, run_exp_067


def test_exp_067_import():
    assert callable(run_exp_067)
    assert callable(gate_passed)
