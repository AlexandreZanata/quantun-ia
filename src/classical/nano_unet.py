"""Compact U-Net denoiser for 32x32 RGB DDPM (Phase H NanoUNet floor)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class NanoUNet(nn.Module):
    """Noise-predicting U-Net for CIFAR-scale RGB (~1–5M params depending on base_ch)."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        ch = base_channels
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.in_conv = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.down1 = ResidualBlock(ch, ch, time_dim)
        self.down2 = ResidualBlock(ch, ch * 2, time_dim)
        self.pool = nn.AvgPool2d(2)
        self.mid1 = ResidualBlock(ch * 2, ch * 2, time_dim)
        self.mid2 = ResidualBlock(ch * 2, ch * 2, time_dim)
        self.up_conv = nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1)
        self.up1 = ResidualBlock(ch * 2, ch, time_dim)
        self.up2 = ResidualBlock(ch, ch, time_dim)
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h0 = self.in_conv(x)
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(self.pool(h1), t_emb)
        h = self.mid2(self.mid1(h2, t_emb), t_emb)
        h = self.up_conv(h)
        h = self.up1(torch.cat([h, h1], dim=1), t_emb)
        h = self.up2(h, t_emb)
        return self.out_conv(F.silu(self.out_norm(h)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train(self, X=None, y=None, **kwargs):  # noqa: N802 — torch Mode + optional API
        if y is None and (X is None or isinstance(X, bool)):
            mode = True if X is None else bool(X)
            return super().train(mode)
        raise TypeError("Use src.training.image_ddpm.train_ddpm for DDPM fitting")

    def predict(self, n: int, *, timesteps: int, device: torch.device | None = None) -> torch.Tensor:
        from src.training.image_ddpm import DDPMSchedule, sample_ddpm

        schedule = DDPMSchedule(timesteps=timesteps, device=device or next(self.parameters()).device)
        return sample_ddpm(self, schedule, n=n)

    def evaluate(self, x: torch.Tensor, *, timesteps: int) -> dict:
        from src.training.image_ddpm import DDPMSchedule, noise_prediction_mse

        device = x.device
        schedule = DDPMSchedule(timesteps=timesteps, device=device)
        mse = noise_prediction_mse(self, schedule, x)
        return {"denoise_mse": float(mse), "n": int(x.shape[0])}
