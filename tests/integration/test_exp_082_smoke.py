"""Smoke test — exp_082 ACYD calibration imports."""

from experiments.exp_082_calibration_acyd.run import gate_passed, run_exp_082


def test_exp_082_import():
    assert callable(run_exp_082)
    assert callable(gate_passed)
