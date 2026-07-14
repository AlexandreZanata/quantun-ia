"""Unit tests for measurement-scheduled CFG helpers (H-Q3.6)."""

from __future__ import annotations

import torch
from torch import nn

from src.training.image_ddpm import DDPMSchedule
from src.training.measurement_cfg import (
    measurement_keep_prob,
    sample_ddpm_classical_cfg,
    sample_ddpm_measurement_schedule,
    train_ddpm_cfg,
)


class _ToyFusion(nn.Module):
    def __init__(self, in_dim: int = 8, out_dim: int = 4) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class _ToyDiT(nn.Module):
    def __init__(self, text_dim: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * 4 * 4 + text_dim + 1, 3 * 4 * 4)

    def forward(self, x: torch.Tensor, t: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.view(b, -1)
        te = t.float().view(b, 1) / 10.0
        h = torch.cat([flat, text, te], dim=-1)
        return self.proj(h).view_as(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def test_measurement_keep_prob_ends_higher_than_mid() -> None:
    t_mid = measurement_keep_prob(5, 11, floor=0.25)
    t_end = measurement_keep_prob(0, 11, floor=0.25)
    t_start = measurement_keep_prob(10, 11, floor=0.25)
    assert t_end > t_mid
    assert t_start > t_mid
    assert 0.25 <= t_mid <= 1.0


def test_train_and_sample_shapes() -> None:
    device = torch.device("cpu")
    schedule = DDPMSchedule(timesteps=4, device=device)
    fusion = _ToyFusion()
    model = _ToyDiT()
    x = torch.randn(8, 3, 4, 4)
    clip = torch.randn(8, 8)
    hist = train_ddpm_cfg(
        model,
        fusion,
        x,
        clip,
        schedule,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        p_uncond=0.2,
        seed=0,
    )
    assert len(hist) == 1
    s_cfg = sample_ddpm_classical_cfg(
        model, fusion, clip[:2], schedule, guidance_scale=1.5, shape=(3, 4, 4)
    )
    s_meas = sample_ddpm_measurement_schedule(
        model, fusion, clip[:2], schedule, guidance_scale=1.5, keep_floor=0.3, shape=(3, 4, 4)
    )
    assert s_cfg.shape == (2, 3, 4, 4)
    assert s_meas.shape == (2, 3, 4, 4)
