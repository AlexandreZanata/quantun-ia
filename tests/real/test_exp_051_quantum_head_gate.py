"""Real gate — exp_051 hybrid QNN head on frozen NIHR backbone (RTX 4060)."""

from __future__ import annotations

import pytest

from experiments.exp_051_quantum_head_nihr.run import gate_passed, run_exp_051

pytestmark = pytest.mark.real


def test_exp_051_quantum_head_ci_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_051 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_051(profile="ci", verbose=False, require_cuda=True)

    assert result.n_train_rows == 5_000
    assert result.n_val_rows == 2_000
    assert result.n_backbone_params > 1_000_000
    assert result.n_trainable_params < 10_000
    assert 0.0 < result.classical_val_pr_auc <= 1.0
    assert 0.0 < result.hybrid_val_pr_auc <= 1.0
    assert gate_passed(result)
