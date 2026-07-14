"""Smoke: exp_102 runner importable (no publication training)."""

from experiments.exp_102_nano_unet_cifar_i2i.run import gate_passed, run_exp_102


def test_exp_102_import():
    assert callable(run_exp_102)
    assert callable(gate_passed)
