"""Real gate — continuous train champion promotion on RTX 4060."""

from __future__ import annotations

import pytest
import torch

from scripts.continuous_train import run_continuous_train


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for real gate — run on RTX 4060 workstation")
    if __import__("os").environ.get("QML_DEVICE", "cuda") == "cpu":
        pytest.skip("QML_DEVICE=cpu — set QML_DEVICE=cuda for real gate")


@pytest.fixture
def _cuda_env(monkeypatch):
    _require_cuda()
    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")


@pytest.mark.real
def test_continuous_train_one_cycle(tmp_path, monkeypatch, _cuda_env):
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_continuous_train(
        profile="ci",
        epochs=8,
        seed=42,
        challenger_exp_id="exp_027_real",
        champion_exp_id="quantum_nano_bc_app_real",
        champion_seed=42,
    )

    assert 0.5 <= result.challenger_accuracy <= 1.0
    assert result.delta_pp >= 0.0
    manifest = tmp_path / "artifacts" / "champion" / "manifest.json"
    assert manifest.is_file()
