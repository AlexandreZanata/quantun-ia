"""Smoke test — exp_071 hybrid GoBug imports."""

from experiments.exp_071_hybrid_nano_gobug.run import gate_passed, run_exp_071


def test_exp_071_import():
    assert callable(run_exp_071)
    assert callable(gate_passed)
