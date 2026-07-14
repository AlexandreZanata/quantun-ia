"""Measurement-scheduled guidance vs classical CFG for captioned DDPM (H-Q3.6)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.image_ddpm import DDPMSchedule, q_sample


def measurement_keep_prob(step: int, timesteps: int, *, floor: float = 0.25) -> float:
    """
    Annealed keep probability for text-channel measurements.

    Low mid-noise (more uncond-like), higher keep near t=0 and t=T-1.
    """
    if timesteps <= 1:
        return 1.0
    x = step / float(timesteps - 1)
    return float(floor + (1.0 - floor) * (2.0 * abs(x - 0.5)))


def _randn_like(ref: torch.Tensor) -> torch.Tensor:
    return torch.randn(ref.shape, device=ref.device, dtype=ref.dtype)


@torch.no_grad()
def sample_ddpm_classical_cfg(
    model: nn.Module,
    fusion: nn.Module,
    clip_emb: torch.Tensor,
    schedule: DDPMSchedule,
    *,
    guidance_scale: float = 2.0,
    shape: tuple[int, int, int] = (3, 32, 32),
) -> torch.Tensor:
    """Classifier-free guidance: eps = eps_u + w (eps_c − eps_u)."""
    model.eval()
    fusion.eval()
    device = schedule.device
    n = clip_emb.shape[0]
    text_c = fusion(clip_emb.to(device))
    text_u = torch.zeros_like(text_c)
    x = torch.randn(n, *shape, device=device)
    w = float(guidance_scale)
    for step in reversed(range(schedule.timesteps)):
        t = torch.full((n,), step, device=device, dtype=torch.long)
        eps_c = model(x, t, text_c)
        eps_u = model(x, t, text_u)
        eps = eps_u + w * (eps_c - eps_u)
        beta = schedule.betas[step]
        alpha = schedule.alphas[step]
        alpha_bar = schedule.alphas_cumprod[step]
        mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - alpha_bar)) * eps)
        if step > 0:
            x = mean + torch.sqrt(beta) * _randn_like(x)
        else:
            x = mean
    return x.clamp(-1.0, 1.0)


@torch.no_grad()
def sample_ddpm_measurement_schedule(
    model: nn.Module,
    fusion: nn.Module,
    clip_emb: torch.Tensor,
    schedule: DDPMSchedule,
    *,
    guidance_scale: float = 2.0,
    keep_floor: float = 0.25,
    shape: tuple[int, int, int] = (3, 32, 32),
) -> torch.Tensor:
    """
    Measurement-scheduled CFG substitute.

    At each step, Bernoulli-mask fusion channels with annealed keep_prob(t);
    guide as eps = eps_meas + w (eps_full − eps_meas).
    """
    model.eval()
    fusion.eval()
    device = schedule.device
    n = clip_emb.shape[0]
    text_full = fusion(clip_emb.to(device))
    x = torch.randn(n, *shape, device=device)
    w = float(guidance_scale)
    for step in reversed(range(schedule.timesteps)):
        t = torch.full((n,), step, device=device, dtype=torch.long)
        p = measurement_keep_prob(step, schedule.timesteps, floor=keep_floor)
        mask = torch.bernoulli(torch.full(text_full.shape, p, device=device))
        text_meas = text_full * mask
        eps_full = model(x, t, text_full)
        eps_meas = model(x, t, text_meas)
        eps = eps_meas + w * (eps_full - eps_meas)
        beta = schedule.betas[step]
        alpha = schedule.alphas[step]
        alpha_bar = schedule.alphas_cumprod[step]
        mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - alpha_bar)) * eps)
        if step > 0:
            x = mean + torch.sqrt(beta) * _randn_like(x)
        else:
            x = mean
    return x.clamp(-1.0, 1.0)


def train_ddpm_cfg(
    model: nn.Module,
    fusion: nn.Module,
    x_train: torch.Tensor,
    clip_emb: torch.Tensor,
    schedule: DDPMSchedule,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    p_uncond: float = 0.1,
    seed: int = 42,
    logger=None,
) -> list[float]:
    """Train with random null text (CFG prep)."""
    if x_train.shape[0] != clip_emb.shape[0]:
        raise ValueError("x_train/clip_emb length mismatch")
    torch.manual_seed(seed)
    device = schedule.device
    model.to(device)
    fusion.to(device)
    model.train()
    fusion.train()
    index = torch.arange(x_train.shape[0])
    loader = DataLoader(TensorDataset(index), batch_size=batch_size, shuffle=True)
    params = [p for p in model.parameters() if p.requires_grad] + [
        p for p in fusion.parameters() if p.requires_grad
    ]
    opt = torch.optim.AdamW(params, lr=lr)
    history: list[float] = []
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_batches = 0
        for (idxs,) in loader:
            batch = x_train[idxs].to(device)
            emb = clip_emb[idxs].to(device)
            text = fusion(emb)
            drop = torch.rand(text.shape[0], device=device) < p_uncond
            text = torch.where(drop.view(-1, 1), torch.zeros_like(text), text)
            t = torch.randint(0, schedule.timesteps, (batch.shape[0],), device=device)
            x_t, noise = q_sample(schedule, batch, t)
            pred = model(x_t, t, text)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        mean_loss = running / max(n_batches, 1)
        history.append(mean_loss)
        if logger is not None:
            logger.log(epoch, loss=mean_loss)
    return history
