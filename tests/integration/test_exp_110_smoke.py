"""Smoke: exp_110 runner importable (no publication training)."""

from experiments.exp_110_text_quantum_token_fusion.run import gate_passed, run_exp_110


def test_exp_110_import():
    assert callable(run_exp_110)
    assert callable(gate_passed)
