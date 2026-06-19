"""Real gate — exp_054 GV-ALR on frozen hybrid QNN head (RTX 4060)."""

from __future__ import annotations

import pytest

from experiments.exp_054_adaptive_hybrid_higgs.run import gate_passed, run_exp_054

pytestmark = pytest.mark.real


def test_exp_054_adaptive_hybrid_ci_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_054 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_054(profile="ci", verbose=False, require_cuda=True)

    assert result.n_train_rows == 8000
    assert result.n_val_rows == 1500
    assert result.n_trainable_params < 10_000
    assert result.adaptive_epochs <= int(result.fixed_epochs * result.max_epoch_fraction)
    assert 0.5 < result.fixed_val_auc <= 1.0
    assert 0.5 < result.adaptive_val_auc <= 1.0
    assert gate_passed(result)
