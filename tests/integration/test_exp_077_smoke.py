"""Smoke test — exp_077 conventional GoBug baselines."""

from experiments.exp_077_conventional_gobug_baselines.run import gate_passed, run_exp_077


def test_exp_077_import():
    assert callable(run_exp_077)
    assert callable(gate_passed)
