"""Smoke: distill DDPM path with tiny tensors."""

from __future__ import annotations

import torch

from src.classical.nano_unet import NanoUNet
from src.training.image_ddpm import DDPMSchedule, train_ddpm, train_ddpm_distill


def test_train_ddpm_distill_smoke():
    device = torch.device("cpu")
    schedule = DDPMSchedule(timesteps=4, device=device)
    teacher = NanoUNet(in_channels=3, base_channels=8)
    student = NanoUNet(in_channels=3, base_channels=8)
    x = torch.randn(8, 3, 32, 32)
    train_ddpm(teacher, x, schedule, epochs=1, batch_size=4, lr=1e-3, seed=0)
    hist = train_ddpm_distill(
        student,
        teacher,
        x,
        schedule,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        alpha=0.7,
        seed=1,
    )
    assert len(hist) == 1
    assert hist[0] >= 0.0
