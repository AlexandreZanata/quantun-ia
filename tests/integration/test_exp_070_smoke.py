"""Smoke test — exp_070 imports and CI profile runs."""

import pytest


def test_exp_070_ci_smoke(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_070 smoke")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    from experiments.exp_070_large_nano_gobug.run import run_exp_070

    result = run_exp_070(profile="ci", verbose=False, require_cuda=True)
    assert result.n_train_rows > 0
    assert result.n_val_rows > 0
    assert result.n_params >= 1_000_000
