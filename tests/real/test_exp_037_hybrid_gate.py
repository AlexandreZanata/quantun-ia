"""Real gate — exp_037 hybrid QNN head on RTX 4060."""

from __future__ import annotations

import pytest

from experiments.exp_037_hybrid_nano_higgs.run import run_exp_037

pytestmark = pytest.mark.real


def test_exp_037_hybrid_ci_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_037 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_037(profile="ci", verbose=False)

    assert result.n_train_rows == 8000
    assert result.n_val_rows == 1500
    assert result.n_backbone_params > 1_000_000
    assert result.n_trainable_params < 10_000
    assert 0.5 < result.classical_val_auc <= 1.0
    assert 0.5 < result.hybrid_val_auc <= 1.0
    assert result.vs_classical_pp >= result.min_vs_classical_pp
