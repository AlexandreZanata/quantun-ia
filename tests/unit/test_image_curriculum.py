"""Unit tests for image curriculum helpers."""

from __future__ import annotations

import numpy as np
import torch

from src.classical.nano_unet import NanoUNet
from src.training.image_curriculum import (
    cumulative_image_stages,
    sharpness_difficulty,
    sort_by_random,
    sort_by_sharpness,
    train_staged_ddpm_curriculum,
)
from src.training.image_ddpm import DDPMSchedule


def test_sharpness_orders_easy_to_hard():
    # Smooth vs noisy patch
    smooth = np.zeros((1, 3, 32, 32), dtype=np.float32)
    noisy = np.random.default_rng(0).standard_normal((1, 3, 32, 32)).astype(np.float32)
    imgs = np.concatenate([noisy, smooth], axis=0)
    ordered = sort_by_sharpness(imgs)
    scores = sharpness_difficulty(ordered)
    assert scores[0] <= scores[-1]


def test_cumulative_stages():
    imgs = np.zeros((16, 3, 8, 8), dtype=np.float32)
    stages = cumulative_image_stages(imgs, n_stages=4)
    assert [len(s) for s in stages] == [4, 8, 12, 16]


def test_sort_by_random_deterministic():
    imgs = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    a = sort_by_random(imgs, seed=7)
    b = sort_by_random(imgs, seed=7)
    assert np.allclose(a, b)


def test_train_staged_ddpm_smoke():
    device = torch.device("cpu")
    schedule = DDPMSchedule(timesteps=4, device=device)
    model = NanoUNet(in_channels=3, base_channels=8)
    imgs = np.random.default_rng(0).standard_normal((8, 3, 32, 32)).astype(np.float32)
    hist = train_staged_ddpm_curriculum(
        model,
        imgs,
        schedule,
        n_stages=2,
        epochs_per_stage=1,
        refine_epochs=1,
        batch_size=4,
        lr=1e-3,
        seed=0,
    )
    assert len(hist) == 3
