"""Smoke: exp_106 runner importable (no publication training)."""

from experiments.exp_106_latent_residual_qnn.run import gate_passed, run_exp_106


def test_exp_106_import():
    assert callable(run_exp_106)
    assert callable(gate_passed)
