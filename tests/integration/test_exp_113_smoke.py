"""Smoke: exp_113 runner importable."""

from experiments.exp_113_open_image_corpus_expand.run import gate_passed, run_exp_113


def test_exp_113_import():
    assert callable(run_exp_113)
    assert callable(gate_passed)
