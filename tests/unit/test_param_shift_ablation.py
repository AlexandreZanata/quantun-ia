"""Unit tests for parameter-shift vs autograd ablation."""

import torch

from src.quantum.qnn_reupload import QuantumNetReupload
from src.training.param_shift_ablation import (
    ParamShiftAblationResult,
    gate_passed,
    measure_reupload_gradient_variance,
    train_reupload_grad_method,
)


def test_measure_reupload_gradient_variance_finite():
    var = measure_reupload_gradient_variance(
        n_qubits=4,
        n_layers=2,
        input_dim=8,
        diff_method="backprop",
        n_samples=2,
        seed=42,
    )
    assert var >= 0.0
    assert not torch.isnan(torch.tensor(var))


def test_train_reupload_grad_method_backprop_returns_accuracy():
    model = QuantumNetReupload(n_qubits=4, n_layers=2, input_dim=4, diff_method="backprop")
    x = torch.randn(16, 4)
    y = (x[:, 0] > 0).float()
    acc = train_reupload_grad_method(
        model,
        x,
        y,
        x,
        y,
        "exp_057",
        "backprop_smoke",
        epochs=3,
        lr=0.05,
        seed=42,
        profile="ci",
    )
    assert 0.0 <= acc <= 1.0


def test_gate_passed_when_within_thresholds():
    ok = ParamShiftAblationResult(
        n_seeds=1,
        mean_autograd_acc=0.9,
        mean_param_shift_acc=0.895,
        mean_holdout_pp=0.5,
        autograd_grad_var=0.02,
        param_shift_grad_var=0.008,
        variance_ratio=2.5,
        max_holdout_pp=1.0,
        min_variance_ratio=2.0,
        elapsed_s=1.0,
    )
    assert gate_passed(ok)
