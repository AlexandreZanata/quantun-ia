"""Gradient diagnostics for barren plateau experiments."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def measure_gradient_variance(
    model_factory,
    n_qubits_list: list[int],
    n_samples: int = 20,
    batch_size: int = 10,
    input_dim: int = 2,
) -> dict[int, float]:
    """Mean gradient variance across random initializations.

    Concatenates all parameter gradients into one vector before computing
    variance — avoids NaN from scalar per-parameter `.var()` calls.
    """
    variances: dict[int, float] = {}
    for n_q in n_qubits_list:
        sample_vars: list[float] = []
        for _ in range(n_samples):
            model = model_factory(n_q)
            X_dummy = torch.randn(batch_size, input_dim)
            y_dummy = torch.randint(0, 2, (batch_size,)).float()

            model.training = True
            model.zero_grad()
            pred = model(X_dummy)
            loss = nn.functional.binary_cross_entropy(pred, y_dummy)
            loss.backward()

            grad_parts = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
            if not grad_parts:
                continue
            grad_flat = torch.cat(grad_parts)
            if grad_flat.numel() > 1:
                sample_vars.append(grad_flat.var().item())

        variances[n_q] = float(np.mean(sample_vars)) if sample_vars else 0.0
    return variances
