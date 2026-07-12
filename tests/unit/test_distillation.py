"""Unit tests for HistGB soft-label distillation helpers."""

from __future__ import annotations

import numpy as np

from src.training.distillation import mix_hard_soft_targets, soft_targets_from_proba


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
