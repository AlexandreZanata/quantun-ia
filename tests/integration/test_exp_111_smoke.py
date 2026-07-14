"""Smoke: exp_111 runner importable (no publication training)."""

from experiments.exp_111_measurement_scheduled_cfg.run import gate_passed, run_exp_111


def test_exp_111_import():
    assert callable(run_exp_111)
    assert callable(gate_passed)
