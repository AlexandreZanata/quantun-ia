"""Smoke: NanoUNet + DDPM wiring (tiny tensors, CPU-friendly)."""

from __future__ import annotations

import torch

from src.classical.nano_unet import NanoUNet
from src.training.image_ddpm import DDPMSchedule, q_sample, sample_ddpm, train_ddpm


def test_nano_unet_forward_shape():
    model = NanoUNet(in_channels=3, base_channels=8)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 10, (2,))
    out = model(x, t)
    assert out.shape == x.shape
    assert model.count_parameters() > 0


def test_ddpm_train_and_sample_smoke():
    device = torch.device("cpu")
    model = NanoUNet(in_channels=3, base_channels=8)
    schedule = DDPMSchedule(timesteps=4, device=device)
    x = torch.randn(8, 3, 32, 32)
    hist = train_ddpm(model, x, schedule, epochs=1, batch_size=4, lr=1e-3, seed=0)
    assert len(hist) == 1
    samples = sample_ddpm(model, schedule, n=2, shape=(3, 32, 32))
    assert samples.shape == (2, 3, 32, 32)


def test_q_sample_shapes():
    schedule = DDPMSchedule(timesteps=5, device=torch.device("cpu"))
    x0 = torch.randn(3, 3, 32, 32)
    t = torch.tensor([0, 2, 4])
    x_t, noise = q_sample(schedule, x0, t)
    assert x_t.shape == x0.shape
    assert noise.shape == x0.shape
