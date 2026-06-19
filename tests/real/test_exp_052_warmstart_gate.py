"""Real gate — exp_052 quantum warm-start vs e2e hybrid on HIGGS (RTX 4060)."""

from __future__ import annotations

import pytest

from experiments.exp_052_quantum_warmstart_higgs.run import gate_passed, run_exp_052

pytestmark = pytest.mark.real


def test_exp_052_warmstart_ci_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_052 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_052(profile="ci", verbose=False, require_cuda=True)

    assert result.n_seeds == 1
    assert result.n_train_rows == 5000
    assert result.n_val_rows == 1500
    assert result.classical_epochs + result.quantum_epochs == result.total_epochs
    assert 0.5 < result.mean_e2e_auc <= 1.0
    assert 0.5 < result.mean_warmstart_auc <= 1.0
    assert gate_passed(result)
