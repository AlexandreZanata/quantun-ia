"""Smoke test — exp_060 imports and CI profile runs."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.real


def test_exp_060_ci_smoke(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_060 smoke")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    from experiments.exp_060_large_nano_acyd_soy.run import run_exp_060

    result = run_exp_060(profile="ci", verbose=False, require_cuda=True)
    assert result.n_params >= 1_000_000
    assert result.n_train_rows == 5000
    assert result.n_val_rows == 1000
