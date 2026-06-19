"""Smoke test — exp_062 hybrid ACYD imports."""

from experiments.exp_062_hybrid_nano_acyd_soy.run import gate_passed, run_exp_062


def test_exp_062_import():
    assert callable(run_exp_062)
    assert callable(gate_passed)
