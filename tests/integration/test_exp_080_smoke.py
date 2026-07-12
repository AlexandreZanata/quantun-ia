"""Smoke test — exp_080 ACYD quantum champion fusion imports."""

from experiments.exp_080_quantum_champion_fusion_acyd.run import gate_passed, run_exp_080


def test_exp_080_import():
    assert callable(run_exp_080)
    assert callable(gate_passed)
