"""Monte-Carlo dropout uncertainty for tabular nanomodel inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class McDropoutUncertainty:
    mean_probability: float
    std_probability: float
    n_samples: int
    method: str = "mc_dropout"


def enable_mc_dropout(model: nn.Module) -> None:
    """Keep Dropout modules in train mode while the rest stays in eval."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    *,
    n_samples: int = 20,
    seed: int = 42,
) -> McDropoutUncertainty:
    """Estimate predictive std via stochastic forward passes with dropout on."""
    if n_samples < 2:
        msg = "n_samples must be >= 2"
        raise ValueError(msg)
    device = next(model.parameters()).device
    x = x.to(device)
    enable_mc_dropout(model)
    rng = np.random.default_rng(seed)
    probs: list[float] = []
    with torch.no_grad():
        for i in range(n_samples):
            torch.manual_seed(int(rng.integers(0, 2**31 - 1)) + i)
            out = model(x)
            if out.ndim == 0:
                out = out.unsqueeze(0)
            probs.append(float(out.reshape(-1)[0].detach().cpu()))
    arr = np.asarray(probs, dtype=np.float64)
    return McDropoutUncertainty(
        mean_probability=float(arr.mean()),
        std_probability=float(arr.std(ddof=1)),
        n_samples=n_samples,
    )
