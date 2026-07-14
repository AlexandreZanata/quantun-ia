"""Image difficulty curriculum for NanoUNet DDPM (Phase I / exp_105)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.image_ddpm import DDPMSchedule, q_sample


def sharpness_difficulty(images: np.ndarray | torch.Tensor) -> np.ndarray:
    """Higher score = harder (sharper). Laplacian variance on grayscale NCHW [-1,1]."""
    if isinstance(images, np.ndarray):
        x = torch.from_numpy(np.asarray(images, dtype=np.float32))
    else:
        x = images.detach().float().cpu()
    if x.ndim != 4:
        msg = f"expected NCHW images, got shape {tuple(x.shape)}"
        raise ValueError(msg)
    # grayscale
    if x.shape[1] == 1:
        gray = x[:, 0]
    else:
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    # 3x3 Laplacian
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(gray.unsqueeze(1), kernel, padding=1)
    var = lap.flatten(1).var(dim=1, unbiased=False)
    return var.numpy().astype(np.float64)


def sort_by_sharpness(images: np.ndarray) -> np.ndarray:
    """Sort images easy→hard (low sharpness first)."""
    scores = sharpness_difficulty(images)
    idx = np.argsort(scores, kind="mergesort")
    return images[idx]


def sort_by_random(images: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(images))
    return images[idx]


def cumulative_image_stages(images: np.ndarray, *, n_stages: int = 4) -> list[np.ndarray]:
    """Cumulative prefixes of an already-sorted (easy→hard or random) image set."""
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    stage_size = max(len(images) // n_stages, 1)
    stages: list[np.ndarray] = []
    for stage in range(1, n_stages + 1):
        end = stage * stage_size if stage < n_stages else len(images)
        stages.append(images[:end])
    return stages


def train_staged_ddpm_curriculum(
    model: nn.Module,
    images_ordered: np.ndarray,
    schedule: DDPMSchedule,
    *,
    n_stages: int = 4,
    epochs_per_stage: int = 2,
    refine_epochs: int = 4,
    batch_size: int = 128,
    lr: float = 2e-4,
    seed: int = 42,
    logger=None,
) -> list[float]:
    """Cumulative stage DDPM training with shared Adam + final full-data refine."""
    torch.manual_seed(seed)
    device = schedule.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    stages = cumulative_image_stages(images_ordered, n_stages=n_stages)
    history: list[float] = []
    epoch = 0
    for stage_idx, stage_np in enumerate(stages, start=1):
        x = torch.from_numpy(np.asarray(stage_np, dtype=np.float32))
        loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)
        model.train()
        for _ in range(epochs_per_stage):
            epoch += 1
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
            if logger is not None:
                logger.log(epoch, loss=mean_loss, stage=stage_idx)

    # refine on full ordered set
    x_full = torch.from_numpy(np.asarray(images_ordered, dtype=np.float32))
    loader = DataLoader(TensorDataset(x_full), batch_size=batch_size, shuffle=True)
    for _ in range(refine_epochs):
        epoch += 1
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
        if logger is not None:
            logger.log(epoch, loss=mean_loss, stage=0)  # 0 = refine
    return history
