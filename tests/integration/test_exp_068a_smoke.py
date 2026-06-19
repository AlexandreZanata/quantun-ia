"""Smoke test — exp_068a angle encoding ACYD imports."""

from experiments.exp_068a_angle_encoding_acyd.run import gate_passed, run_exp_068a


def test_exp_068a_import():
    assert callable(run_exp_068a)
    assert callable(gate_passed)
