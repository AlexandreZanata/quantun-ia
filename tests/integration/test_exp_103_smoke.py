"""Smoke: exp_103 runner importable (no publication training)."""

from experiments.exp_103_tiny_dit_flickr_t2i.run import gate_passed, run_exp_103


def test_exp_103_import():
    assert callable(run_exp_103)
    assert callable(gate_passed)
