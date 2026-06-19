"""Real gate — exp_055 depolarizing noise on GoBug hybrid (RTX 4060)."""

from __future__ import annotations

import pytest

from experiments.exp_055_noise_reg_gobug.run import gate_passed, run_exp_055

pytestmark = pytest.mark.real


def test_exp_055_noise_reg_ci_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_055 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_055(profile="ci", verbose=False, require_cuda=True)

    assert result.n_train_rows == 5000
    assert result.n_val_rows == 1500
    assert result.n_test_rows > 0
    assert result.n_params_noiseless > 0
    assert 0.0 < result.noiseless_test_pr_auc <= 1.0
    assert 0.0 < result.noisy_test_pr_auc <= 1.0
    assert gate_passed(result)
