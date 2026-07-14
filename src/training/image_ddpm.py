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


def train_ddpm_distill(
    student: nn.Module,
    teacher: nn.Module,
    x_train: torch.Tensor,
    schedule: DDPMSchedule,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    alpha: float = 0.7,
    seed: int = 42,
    log_every: int = 1,
    logger=None,
) -> list[float]:
    """Train student with soft teacher noise targets mixed with hard Gaussian noise."""
    from src.training.distillation import denoise_distill_loss

    torch.manual_seed(seed)
    device = schedule.device
    student.to(device)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.train()
    loader = DataLoader(
        TensorDataset(x_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    history: list[float] = []
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            t = torch.randint(0, schedule.timesteps, (batch.shape[0],), device=device)
            x_t, noise = q_sample(schedule, batch, t)
            with torch.no_grad():
                teacher_eps = teacher(x_t, t)
            student_eps = student(x_t, t)
            loss = denoise_distill_loss(student_eps, teacher_eps, noise, alpha=alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        mean_loss = running / max(n_batches, 1)
        history.append(mean_loss)
        if logger is not None and (epoch % log_every == 0 or epoch == epochs):
            logger.log(epoch, loss=mean_loss, distill_alpha=alpha)
    return history


def step_ddpm_gradient_variance(
    model: nn.Module,
    schedule: DDPMSchedule,
    x_batch: torch.Tensor,
) -> float:
    """Gradient variance of one DDPM noise-prediction step (GV-ALR diagnostic)."""
    device = schedule.device
    model.train()
    model.zero_grad(set_to_none=True)
    batch = x_batch.to(device)
    t = torch.randint(0, schedule.timesteps, (batch.shape[0],), device=device)
    x_t, noise = q_sample(schedule, batch, t)
    pred = model(x_t, t)
    loss = F.mse_loss(pred, noise)
    loss.backward()
    grad_parts = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
    if not grad_parts:
        return 0.0
    grad_flat = torch.cat(grad_parts)
    if grad_flat.numel() <= 1:
        return 0.0
    return float(grad_flat.var().item())


def train_ddpm_gvalr(
    model: nn.Module,
    x_train: torch.Tensor,
    schedule: DDPMSchedule,
    *,
    epochs: int,
    batch_size: int,
    adaptive_config,
    seed: int = 42,
    log_every: int = 1,
    logger=None,
) -> list[float]:
    """Train DDPM noise predictor with gradient-variance adaptive LR (GV-ALR)."""
    from src.training.adaptive_lr import AdaptiveLRConfig, compute_lr_scale

    cfg: AdaptiveLRConfig = adaptive_config
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
    current_lr = cfg.base_lr
    opt = torch.optim.AdamW(model.parameters(), lr=current_lr)
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

        grad_var = 0.0
        if epoch > cfg.warmup_epochs and epoch % cfg.adapt_every == 0:
            sample_batch = next(iter(loader))[0]
            grad_var = step_ddpm_gradient_variance(model, schedule, sample_batch)
            scale = compute_lr_scale(grad_var, cfg)
            current_lr = cfg.base_lr * scale
            for group in opt.param_groups:
                group["lr"] = current_lr

        mean_loss = running / max(n_batches, 1)
        history.append(mean_loss)
        if logger is not None and (epoch % log_every == 0 or epoch == epochs):
            logger.log(epoch, loss=mean_loss, learning_rate=current_lr, grad_variance=grad_var)
    return history


def train_ddpm_captioned(
    model: nn.Module,
    embedder: nn.Module,
    x_train: torch.Tensor,
    captions: list[str],
    schedule: DDPMSchedule,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int = 42,
    logger=None,
) -> list[float]:
    """Train noise predictor with caption embeddings from ``embedder``."""
    if x_train.shape[0] != len(captions):
        raise ValueError("x_train/captions length mismatch")
    torch.manual_seed(seed)
    device = schedule.device
    model.to(device)
    embedder.to(device)
    model.train()
    embedder.train()
    index = torch.arange(x_train.shape[0])
    loader = DataLoader(
        TensorDataset(index),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    params = [p for p in model.parameters() if p.requires_grad] + [
        p for p in embedder.parameters() if p.requires_grad
    ]
    opt = torch.optim.AdamW(params, lr=lr)
    history: list[float] = []
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_batches = 0
        for (idxs,) in loader:
            batch = x_train[idxs].to(device)
            caps = [captions[int(i)] for i in idxs.tolist()]
            text_emb = embedder(caps)
            t = torch.randint(0, schedule.timesteps, (batch.shape[0],), device=device)
            x_t, noise = q_sample(schedule, batch, t)
            pred = model(x_t, t, text_emb)
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


@torch.no_grad()
def sample_ddpm_captioned(
    model: nn.Module,
    embedder: nn.Module,
    captions: list[str],
    schedule: DDPMSchedule,
    *,
    shape: tuple[int, int, int] = (3, 32, 32),
) -> torch.Tensor:
    model.eval()
    embedder.eval()
    device = schedule.device
    n = len(captions)
    text_emb = embedder(captions).to(device)
    x = torch.randn(n, *shape, device=device)
    for step in reversed(range(schedule.timesteps)):
        t = torch.full((n,), step, device=device, dtype=torch.long)
        eps = model(x, t, text_emb)
        beta = schedule.betas[step]
        alpha = schedule.alphas[step]
        alpha_bar = schedule.alphas_cumprod[step]
        mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - alpha_bar)) * eps)
        if step > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean
    return x.clamp(-1.0, 1.0)
