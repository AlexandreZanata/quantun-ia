"""Smoke: exp_105 runner importable (no publication training)."""

from experiments.exp_105_image_difficulty_curriculum.run import gate_passed, run_exp_105


def test_exp_105_import():
    assert callable(run_exp_105)
    assert callable(gate_passed)
