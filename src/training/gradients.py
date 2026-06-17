"""Gradient diagnostics for barren plateau experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.statistics import seed_summary


def measure_gradient_variance(
    model_factory,
    n_qubits_list: list[int],
    n_samples: int = 20,
    batch_size: int = 10,
    input_dim: int = 2,
    *,
    use_parameter_shift: bool = False,
) -> dict[int, dict]:
    """Gradient variance per qubit count with bootstrap 95% CI across random inits."""
    if use_parameter_shift:
        batch_size = 1
    results: dict[int, dict] = {}
    for n_q in n_qubits_list:
        sample_vars: list[float] = []
        for _ in range(n_samples):
            model = model_factory(n_q)
            X_dummy = torch.randn(batch_size, input_dim)
            y_dummy = torch.randint(0, 2, (batch_size,)).float()

            model.training = True
            model.zero_grad()
            pred = model(X_dummy).reshape(-1)
            y_dummy = y_dummy.reshape(-1)
            loss = nn.functional.binary_cross_entropy(pred, y_dummy)
            loss.backward()

            grad_parts = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
            if not grad_parts:
                continue
            grad_flat = torch.cat(grad_parts)
            if grad_flat.numel() > 1:
                sample_vars.append(grad_flat.var().item())

        if sample_vars:
            stats = seed_summary(sample_vars)
            results[n_q] = {
                "mean": stats["mean"],
                "std": stats["std"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "n_samples": stats["n_seeds"],
            }
        else:
            results[n_q] = {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_samples": 0}
    return results
