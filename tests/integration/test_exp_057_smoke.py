"""Smoke test — exp_057 param-shift ablation imports."""

from experiments.exp_057_param_shift_ablation.run import gate_passed, run_exp_057


def test_exp_057_import():
    assert callable(run_exp_057)
    assert callable(gate_passed)
