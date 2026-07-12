"""Smoke test — exp_066 ACYD noise-reg imports."""

from experiments.exp_066_noise_reg_acyd.run import gate_passed, run_exp_066


def test_exp_066_import():
    assert callable(run_exp_066)
    assert callable(gate_passed)
