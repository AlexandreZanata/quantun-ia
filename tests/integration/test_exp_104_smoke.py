"""Smoke: exp_104 runner importable (no publication training)."""

from experiments.exp_104_distill_image_nano.run import gate_passed, run_exp_104


def test_exp_104_import():
    assert callable(run_exp_104)
    assert callable(gate_passed)
