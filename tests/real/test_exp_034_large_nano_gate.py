"""Real gate — exp_034 LargeNanoMLP beats logistic on Synthea val AUC (RTX 4060)."""

from __future__ import annotations

import pytest

from experiments.exp_034_large_nano_synthea.run import run_exp_034

pytestmark = pytest.mark.real


def test_exp_034_large_nano_synthea_publication_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_034 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_034(profile="publication", verbose=False, require_cuda=True)

    assert result.n_params >= 1_000_000
    assert result.n_train_rows == 700_000
    assert result.auc_advantage_pp >= result.min_auc_advantage_pp
