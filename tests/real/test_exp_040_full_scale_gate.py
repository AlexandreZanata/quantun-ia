"""Real gate — exp_040 full-scale HIGGS ablation on RTX 4060 (805K × 3 seeds)."""

from __future__ import annotations

import pytest

from experiments.exp_036_method_ablation_higgs.run import METHODS
from experiments.exp_040_full_scale_ablation_higgs.run import run_exp_040

pytestmark = pytest.mark.real


def test_exp_040_full_scale_ablation_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_040 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_040(verbose=False)

    assert result.n_seeds == 3
    assert result.n_train_rows == 805_000
    assert result.n_val_rows == 172_500
    for method in METHODS:
        assert method in result.mean_auc_by_method
        assert 0.5 < result.mean_auc_by_method[method] < 1.0
        assert len(result.auc_by_method_seed[method]) == 3
