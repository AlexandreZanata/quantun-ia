"""Smoke: exp_107 runner importable (no publication training)."""

from experiments.exp_107_patch_amplitude_bottleneck.run import gate_passed, run_exp_107


def test_exp_107_import():
    assert callable(run_exp_107)
    assert callable(gate_passed)
