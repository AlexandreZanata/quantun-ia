"""Real gate — exp_044 NIHR CV baseline on RTX 4060."""

from __future__ import annotations

import pytest

from experiments.exp_044_nihr_cv_baseline.run import gate_passed, run_exp_044

pytestmark = pytest.mark.real


def test_exp_044_nihr_ci_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_044 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_044(profile="ci", verbose=False, require_cuda=True)

    assert result.n_train_rows == 20_000
    assert result.n_val_rows == 5_000
    assert result.n_params > 0
    assert 0.5 < result.nano_val_auc < 1.0
    assert gate_passed(result)
