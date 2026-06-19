"""Real gate — exp_060 LargeNanoMLP beats logistic on ACYD val AUC (RTX 4060)."""

from __future__ import annotations

import pytest

from experiments.exp_060_large_nano_acyd_soy.run import run_exp_060

pytestmark = pytest.mark.real


def test_exp_060_large_nano_acyd_soy_publication_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_060 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_060(profile="publication", verbose=False, require_cuda=True)

    assert result.n_params >= 1_000_000
    assert result.n_train_rows == 50_107
    assert result.auc_advantage_pp >= result.min_auc_advantage_pp
