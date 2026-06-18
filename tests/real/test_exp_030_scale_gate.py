"""Real gate — exp_030 30-seed scale stability on RTX 4060."""

from __future__ import annotations

import pytest

from experiments.exp_030_publication_large.run import run_exp_030

pytestmark = pytest.mark.real


def test_exp_030_publication_large_scale_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_030 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_030(profile="publication_large", verbose=False, require_cuda=True)

    assert result.n_samples == 1000
    assert result.n_seeds == 30
    assert result.reference_seeds == 10
    assert result.delta_pp <= result.parity_max_delta_pp
