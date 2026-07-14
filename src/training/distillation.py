"""Knowledge-distillation helpers for conventional → nano soft-label training."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def soft_targets_from_proba(proba_2d: np.ndarray) -> np.ndarray:
    """Extract positive-class probabilities from predict_proba output."""
    arr = np.asarray(proba_2d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("expected predict_proba array with shape (n, 2+)")
    return arr[:, 1].astype(np.float64, copy=False)


def mix_hard_soft_targets(
    hard: np.ndarray,
    soft: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Mix soft teacher probs with hard labels: ``alpha * soft + (1 - alpha) * hard``.

    ``alpha=1`` is pure distillation; ``alpha=0`` is hard-label training.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    h = np.asarray(hard, dtype=np.float64).reshape(-1)
    s = np.asarray(soft, dtype=np.float64).reshape(-1)
    if h.shape != s.shape:
        raise ValueError(f"hard/soft shape mismatch: {h.shape} vs {s.shape}")
    mixed = alpha * s + (1.0 - alpha) * h
    return np.clip(mixed, 0.0, 1.0)


def denoise_distill_loss(
    student_eps: torch.Tensor,
    teacher_eps: torch.Tensor,
    true_noise: torch.Tensor,
    *,
    alpha: float = 0.7,
) -> torch.Tensor:
    """Mix soft teacher noise predictions with hard Gaussian noise targets.

    ``alpha=1`` is pure teacher imitation; ``alpha=0`` is standard DDPM MSE.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    soft = F.mse_loss(student_eps, teacher_eps)
    hard = F.mse_loss(student_eps, true_noise)
    return alpha * soft + (1.0 - alpha) * hard


def relative_fid_improvement(fid_baseline: float, fid_challenger: float) -> float:
    """Return ``1 - challenger/baseline`` (positive ⇒ challenger is better / lower FID)."""
    if fid_baseline <= 0:
        raise ValueError("fid_baseline must be > 0")
    return 1.0 - (fid_challenger / fid_baseline)
