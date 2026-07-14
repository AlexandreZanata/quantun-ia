"""1-D latent DDPM helpers for NanoVAE latents (Phase J)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.image_ddpm import DDPMSchedule


def q_sample_latent(
    schedule: DDPMSchedule,
    z0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(z0)
    s = schedule.sqrt_alphas_cumprod[t].view(-1, 1)
    s2 = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
    return s * z0 + s2 * noise, noise


@torch.no_grad()
def sample_latent_ddpm(
    model: nn.Module,
    schedule: DDPMSchedule,
    *,
    n: int,
    latent_dim: int,
) -> torch.Tensor:
    model.eval()
    device = schedule.device
    # Quantum path needs CPU — move schedule tensors to model device via z device
    z = torch.randn(n, latent_dim, device=device)
    for step in reversed(range(schedule.timesteps)):
        t = torch.full((n,), step, device=device, dtype=torch.long)
        eps = model(z, t)
        beta = schedule.betas[step]
        alpha = schedule.alphas[step]
        alpha_bar = schedule.alphas_cumprod[step]
        mean = (1.0 / torch.sqrt(alpha)) * (z - (beta / torch.sqrt(1.0 - alpha_bar)) * eps)
        if step > 0:
            z = mean + torch.sqrt(beta) * torch.randn_like(z)
        else:
            z = mean
    return z


def train_latent_ddpm(
    model: nn.Module,
    z_train: torch.Tensor,
    schedule: DDPMSchedule,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int = 42,
    logger=None,
) -> list[float]:
    torch.manual_seed(seed)
    device = schedule.device
    # Keep model where its params live; schedule on same device as batches for classical
    model.train()
    loader = DataLoader(TensorDataset(z_train), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    history: list[float] = []
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            t = torch.randint(0, schedule.timesteps, (batch.shape[0],), device=device)
            z_t, noise = q_sample_latent(schedule, batch, t)
            pred = model(z_t, t)
            loss = F.mse_loss(pred, noise.to(pred.device))
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
