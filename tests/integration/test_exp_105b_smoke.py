"""Smoke: exp_105b runner importable (no publication training)."""

from experiments.exp_105b_gv_alr_image_ddpm.run import gate_passed, run_exp_105b


def test_exp_105b_import():
    assert callable(run_exp_105b)
    assert callable(gate_passed)
