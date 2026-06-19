"""Real gate — exp_057 parameter-shift vs autograd (RTX 4060)."""

from __future__ import annotations

import pytest

from experiments.exp_057_param_shift_ablation.run import gate_passed, run_exp_057

pytestmark = pytest.mark.real


def test_exp_057_param_shift_ci_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_057 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_057(profile="ci", verbose=False, require_cuda=True)

    assert result.n_seeds == 1
    assert 0.0 <= result.mean_autograd_acc <= 1.0
    assert 0.0 <= result.mean_param_shift_acc <= 1.0
    assert result.autograd_grad_var >= 0.0
    assert result.param_shift_grad_var >= 0.0
    assert gate_passed(result)
