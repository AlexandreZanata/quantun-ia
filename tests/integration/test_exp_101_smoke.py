"""Smoke: exp_101 runner importable (no download requirement)."""

from experiments.exp_101_open_image_corpus_ingest.run import gate_passed, run_exp_101


def test_exp_101_import():
    assert callable(run_exp_101)
    assert callable(gate_passed)
