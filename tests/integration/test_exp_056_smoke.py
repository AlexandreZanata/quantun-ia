"""Smoke test — exp_056 re-upload ladder imports."""

from experiments.exp_056_reupload_curriculum_ladder.run import gate_passed, run_exp_056


def test_exp_056_import():
    assert callable(run_exp_056)
    assert callable(gate_passed)
