"""Seasonal cyclic feature extraction for ACYD agro-climate QNN heads."""

from __future__ import annotations

import math

import torch

# Mean columns of the in-season weather block (7 vars × mean stat first).
ACYD_SEASON_MEAN_INDICES = (9, 13, 17, 21, 25, 29, 33)
N_CYCLIC_FEATURES = 4


def extract_acyd_seasonal_cyclic(x: torch.Tensor) -> torch.Tensor:
    """
    Derive four phase angles from scaled ACYD tabular rows (37 features).

    Uses mean in-season weather aggregates to form DOY-like sin/cos harmonics
    suitable for PennyLane AngleEmbedding.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.shape[-1] < max(ACYD_SEASON_MEAN_INDICES) + 1:
        msg = f"expected ACYD feature dim >= {max(ACYD_SEASON_MEAN_INDICES) + 1}, got {x.shape[-1]}"
        raise ValueError(msg)

    season_means = x[:, list(ACYD_SEASON_MEAN_INDICES)]
    phase = season_means.mean(dim=1)
    phase_norm = torch.sigmoid(phase)
    two_pi = 2.0 * math.pi
    angles = torch.stack(
        [
            torch.sin(two_pi * phase_norm),
            torch.cos(two_pi * phase_norm),
            torch.sin(two_pi * 2.0 * phase_norm),
            torch.cos(two_pi * 2.0 * phase_norm),
        ],
        dim=1,
    )
    return angles * math.pi
