"""Masked weather-feature SSL for ACYD tabular climate (Phase D / exp_099)."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.device import resolve_device
from src.training.metrics import ExperimentLogger
from src.training.reproducibility import set_global_seed
from src.training.trainer import count_parameters

# ACYD 37-d: 0–2 geo/area, 3–8 soil, 9–36 weather aggregates (7 vars × 4 stats).
ACYD_WEATHER_START = 9
ACYD_WEATHER_END = 37


def weather_feature_indices(input_dim: int = 37) -> np.ndarray:
    end = min(ACYD_WEATHER_END, int(input_dim))
    start = min(ACYD_WEATHER_START, end)
    return np.arange(start, end, dtype=np.int64)


def mask_weather_features(
    x: torch.Tensor,
    *,
    mask_ratio: float = 0.3,
    weather_idx: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (masked_x, boolean mask) with weather columns zeroed at ``mask_ratio``."""
    if not 0.0 < mask_ratio < 1.0:
        msg = f"mask_ratio must be in (0,1), got {mask_ratio}"
        raise ValueError(msg)
    idx = weather_idx if weather_idx is not None else weather_feature_indices(x.shape[1])
    if len(idx) == 0:
        msg = "no weather features available to mask"
        raise ValueError(msg)

    n_weather = len(idx)
    weather = torch.rand(x.shape[0], n_weather, device=x.device) < mask_ratio
    empty = ~weather.any(dim=1)
    if empty.any():
        choices = torch.randint(0, n_weather, (int(empty.sum().item()),), device=x.device)
        weather[empty, choices] = True
    full_mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
    full_mask[:, torch.as_tensor(idx, device=x.device, dtype=torch.long)] = weather
    masked = x.masked_fill(full_mask, 0.0)
    return masked, full_mask


class ResidualNanoSSL(nn.Module):
    """ResidualNano encoder + linear reconstruction head for masked climate SSL."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.n_blocks = n_blocks
        template = ResidualNanoMLP(
            input_dim,
            hidden=hidden,
            n_blocks=n_blocks,
            bottleneck=64,
            dropout=dropout,
        )
        self.stem = template.stem
        self.blocks = template.blocks
        self.decoder = nn.Linear(hidden, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


def copy_encoder_to_residual_nano(ssl: ResidualNanoSSL, supervised: ResidualNanoMLP) -> int:
    """Copy stem+blocks weights from SSL encoder into a ResidualNanoMLP."""
    supervised.stem.load_state_dict(ssl.stem.state_dict())
    supervised.blocks.load_state_dict(ssl.blocks.state_dict())
    n = sum(p.numel() for p in ssl.stem.parameters()) + sum(
        p.numel() for p in ssl.blocks.parameters()
    )
    return int(n)


def pretrain_masked_climate(
    model: ResidualNanoSSL,
    x_train: np.ndarray,
    *,
    exp_id: str,
    model_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    mask_ratio: float,
    seed: int,
    profile: str | None = None,
) -> dict:
    """Masked weather reconstruction pretrain; returns final train MSE."""
    set_global_seed(seed)
    dev = resolve_device(model=model)
    model = model.to(dev)
    weather_idx = weather_feature_indices(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="none")
    log = ExperimentLogger(exp_id, model_name, seed=seed, profile=profile)
    log._tracker.log_params(
        {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "mask_ratio": mask_ratio,
            "seed": seed,
            "profile": profile,
            "device": str(dev),
            "n_weather_features": int(len(weather_idx)),
        }
    )

    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    t0 = time.time()
    last_mse = float("nan")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for (x_batch,) in loader:
            x_batch = x_batch.to(dev)
            masked, mask = mask_weather_features(
                x_batch,
                mask_ratio=mask_ratio,
                weather_idx=weather_idx,
            )
            pred = model(masked)
            per_elem = criterion(pred, x_batch)
            denom = mask.sum().clamp_min(1.0)
            loss = (per_elem * mask.float()).sum() / denom
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        last_mse = total_loss / max(n_batches, 1)
        log.log(epoch, loss=last_mse, accuracy=0.0, mse=last_mse)

    elapsed = time.time() - t0
    log.finish(
        elapsed,
        test_accuracy=0.0,
        test_loss=last_mse,
        eval_set="ssl_pretrain",
        n_params=count_parameters(model),
        final_mse=last_mse,
    )
    return {
        "mse": float(last_mse),
        "elapsed_s": float(elapsed),
        "n_weather": int(len(weather_idx)),
    }


def train_supervised_residual(
    model: ResidualNanoMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    exp_id: str,
    model_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
    profile: str,
) -> float:
    train_model_batched(
        model,
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        exp_id,
        model_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        X_val=torch.tensor(x_val, dtype=torch.float32),
        y_val=torch.tensor(y_val, dtype=torch.float32),
        seed=seed,
        profile=profile,
        save_checkpoints=False,
    )
    return float(
        evaluate_with_auc(
            model,
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )["roc_auc"]
    )
