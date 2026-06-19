"""Smoke test — exp_055 noise regularization imports."""

from experiments.exp_055_noise_reg_gobug.run import gate_passed, run_exp_055


def test_exp_055_import():
    assert callable(run_exp_055)
    assert callable(gate_passed)
