"""Real gate — exp_069 LargeNanoMLP NIHR CV on RTX 4060 (CI profile)."""

from __future__ import annotations

import pytest

from experiments.exp_069_large_nano_nihr.run import gate_passed, run_exp_069

pytestmark = pytest.mark.real


def test_exp_069_large_nano_nihr_ci_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_069 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_069(profile="ci", verbose=False, require_cuda=True)

    assert result.n_params >= 1_000_000
    assert result.n_train_rows == 5_000
    assert 0.0 < result.nano_val_pr_auc <= 1.0
    assert gate_passed(result)
