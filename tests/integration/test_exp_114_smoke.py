"""Smoke: exp_114 runner importable (no publication training)."""

from experiments.exp_114_nano_unet_v2_efficient_i2i.run import gate_passed, run_exp_114


def test_exp_114_import():
    assert callable(run_exp_114)
    assert callable(gate_passed)
