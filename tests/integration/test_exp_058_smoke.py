"""Smoke test — exp_058 conventional HIGGS baselines."""

from experiments.exp_058_conventional_higgs_baselines.run import gate_passed, run_exp_058


def test_exp_058_import():
    assert callable(run_exp_058)
    assert callable(gate_passed)
