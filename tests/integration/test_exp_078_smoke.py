"""Smoke test — exp_078 agro validation imports."""

from experiments.exp_078_agro_clinical_cases.run import gate_passed, run_exp_078


def test_exp_078_import():
    assert callable(run_exp_078)
    assert callable(gate_passed)
