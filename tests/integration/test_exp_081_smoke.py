"""Smoke test — exp_081 ACYD maize LargeNanoMLP imports."""

from experiments.exp_081_large_nano_acyd_maize.run import run_exp_081


def test_exp_081_import():
    assert callable(run_exp_081)
