"""Smoke: DDPM GV-ALR trainer on tiny tensors."""

from __future__ import annotations

import torch

from src.classical.nano_unet import NanoUNet
from src.training.adaptive_lr import AdaptiveLRConfig
from src.training.image_ddpm import DDPMSchedule, step_ddpm_gradient_variance, train_ddpm_gvalr


def test_step_ddpm_gradient_variance_positive():
    device = torch.device("cpu")
    schedule = DDPMSchedule(timesteps=4, device=device)
    model = NanoUNet(in_channels=3, base_channels=8)
    x = torch.randn(4, 3, 32, 32)
    var = step_ddpm_gradient_variance(model, schedule, x)
    assert var >= 0.0


def test_train_ddpm_gvalr_smoke():
    device = torch.device("cpu")
    schedule = DDPMSchedule(timesteps=4, device=device)
    model = NanoUNet(in_channels=3, base_channels=8)
    x = torch.randn(8, 3, 32, 32)
    cfg = AdaptiveLRConfig(base_lr=1e-3, warmup_epochs=0, var_target=0.01)
    hist = train_ddpm_gvalr(
        model,
        x,
        schedule,
        epochs=2,
        batch_size=4,
        adaptive_config=cfg,
        seed=0,
    )
    assert len(hist) == 2
