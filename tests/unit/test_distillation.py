"""Unit tests for HistGB soft-label distillation helpers."""

from __future__ import annotations

import numpy as np
import torch

from src.training.distillation import (
    denoise_distill_loss,
    mix_hard_soft_targets,
    relative_fid_improvement,
    soft_targets_from_proba,
)


def test_soft_targets_from_proba_shape():
    proba = np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float64)
    soft = soft_targets_from_proba(proba)
    assert soft.shape == (2,)
    assert np.allclose(soft, [0.8, 0.1])


def test_mix_hard_soft_targets():
    hard = np.array([0.0, 1.0, 1.0])
    soft = np.array([0.2, 0.7, 0.9])
    mixed = mix_hard_soft_targets(hard, soft, alpha=0.7)
    assert mixed.shape == (3,)
    assert np.allclose(mixed, 0.7 * soft + 0.3 * hard)
    assert np.all((mixed >= 0.0) & (mixed <= 1.0))


def test_denoise_distill_loss_alpha_extremes():
    student = torch.zeros(2, 3, 4, 4)
    teacher = torch.ones(2, 3, 4, 4)
    noise = torch.zeros(2, 3, 4, 4)
    pure_soft = denoise_distill_loss(student, teacher, noise, alpha=1.0)
    pure_hard = denoise_distill_loss(student, teacher, noise, alpha=0.0)
    assert float(pure_hard) == 0.0
    assert float(pure_soft) > 0.0


def test_relative_fid_improvement():
    assert abs(relative_fid_improvement(100.0, 80.0) - 0.2) < 1e-9
