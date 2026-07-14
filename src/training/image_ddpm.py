"""DDPM schedule, training, and sampling for NanoUNet (Phase H)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DDPMSchedule:
    timesteps: int
    device: torch.device
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self) -> None:
        betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


def q_sample(
    schedule: DDPMSchedule,
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(x0)
    s = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    s2 = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return s * x0 + s2 * noise, noise


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    schedule: DDPMSchedule,
    *,
    n: int,
    shape: tuple[int, int, int] = (3, 32, 32),
) -> torch.Tensor:
    model.eval()
    device = schedule.device
    x = torch.randn(n, *shape, device=device)
    for step in reversed(range(schedule.timesteps)):
        t = torch.full((n,), step, device=device, dtype=torch.long)
        eps = model(x, t)
        beta = schedule.betas[step]
        alpha = schedule.alphas[step]
        alpha_bar = schedule.alphas_cumprod[step]
        mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - alpha_bar)) * eps)
        if step > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean
    return x.clamp(-1.0, 1.0)


def noise_prediction_mse(
    model: nn.Module,
    schedule: DDPMSchedule,
    x0: torch.Tensor,
    *,
    batch_size: int = 128,
) -> float:
    model.eval()
    device = schedule.device
    total = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, x0.shape[0], batch_size):
            batch = x0[i : i + batch_size].to(device)
            t = torch.randint(0, schedule.timesteps, (batch.shape[0],), device=device)
            x_t, noise = q_sample(schedule, batch, t)
            pred = model(x_t, t)
            total += float(F.mse_loss(pred, noise, reduction="sum").item())
            n += batch.numel()
    return total / max(n, 1)


def train_ddpm(
    model: nn.Module,
    x_train: torch.Tensor,
    schedule: DDPMSchedule,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int = 42,
    log_every: int = 1,
    logger=None,
) -> list[float]:
    """Train noise predictor; returns per-epoch mean loss."""
    torch.manual_seed(seed)
    device = schedule.device
    model.to(device)
    model.train()
    loader = DataLoader(
        TensorDataset(x_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[float] = []
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            t = torch.randint(0, schedule.timesteps, (batch.shape[0],), device=device)
            x_t, noise = q_sample(schedule, batch, t)
            pred = model(x_t, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        mean_loss = running / max(n_batches, 1)
        history.append(mean_loss)
        if logger is not None and (epoch % log_every == 0 or epoch == epochs):
            logger.log(epoch, loss=mean_loss)
    return history
