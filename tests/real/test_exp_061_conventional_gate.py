"""Real gate — exp_061 conventional ACYD baseline comparison completes and logs."""

from __future__ import annotations

import pytest

from experiments.exp_061_conventional_acyd_baselines.run import run_exp_061

pytestmark = pytest.mark.real


def test_exp_061_conventional_acyd_baselines_publication(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_061 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_061(profile="publication", verbose=False, require_cuda=True)

    assert result.n_train_rows == 50_107
    assert result.n_val_rows == 5_830
    assert len(result.scores) == 5
    assert any(s.model_key == "large_nano_mlp" for s in result.scores)
