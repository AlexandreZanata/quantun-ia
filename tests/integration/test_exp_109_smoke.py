"""Smoke: exp_109 runner importable (no publication training)."""

from experiments.exp_109_circuit_cut_latent_6q.run import gate_passed, run_exp_109


def test_exp_109_import():
    assert callable(run_exp_109)
    assert callable(gate_passed)
