"""Ship NanoUNet CIFAR I2I serve bundle (Phase K / K-T1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.classical.nano_unet import NanoUNet
from src.data.open_images import load_cifar10_nchw
from src.training.config import load_experiment_config
from src.training.device import resolve_device
from src.training.image_ddpm import DDPMSchedule, train_ddpm
from src.training.structured_log import init_correlation_id, log_event

SERVE_KEY = "nano_unet_cifar"
DEFAULT_SERVE_DIR = Path("dist/serve_models") / SERVE_KEY


@dataclass(frozen=True)
class NanoUNetShipResult:
    serve_dir: Path
    n_params: int
    epochs: int
    timesteps: int
    base_channels: int
    device: str
    profile: str
    elapsed_s: float


def nano_unet_serve_dir(root: Path | None = None) -> Path:
    base = root or Path(".")
    return base / "dist" / "serve_models" / SERVE_KEY


def is_nano_unet_shipped(root: Path | None = None) -> bool:
    serve = nano_unet_serve_dir(root)
    return (serve / "model.pt").is_file() and (serve / "meta.json").is_file()


def ship_nano_unet_cifar(
    *,
    profile: str = "ci",
    root: Path | None = None,
    verbose: bool = True,
) -> NanoUNetShipResult:
    """Train NanoUNet (exp_102 config) and publish weights for /predict/image."""
    import time

    init_correlation_id()
    root = root or Path(".")
    cfg = load_experiment_config("exp_102_nano_unet_cifar_i2i", profile=profile)
    seed = int(cfg.get("seed", 42))
    epochs = int(cfg.get("epochs", 2))
    batch_size = int(cfg.get("batch_size", 8))
    lr = float(cfg.get("lr", 2e-4))
    timesteps = int(cfg.get("timesteps", 20))
    base_channels = int(cfg.get("base_channels", 32))
    n_train = int(cfg.get("n_train", 32))

    device = resolve_device()
    torch.manual_seed(seed)
    x_train_np, _ = load_cifar10_nchw(split="train", n_take=n_train, seed=seed)
    x_train = torch.from_numpy(x_train_np)
    model = NanoUNet(in_channels=3, base_channels=base_channels)
    schedule = DDPMSchedule(timesteps=timesteps, device=device)
    t0 = time.perf_counter()
    if verbose:
        print(
            f"shipping nano_unet_cifar profile={profile} device={device} "
            f"n_train={n_train} epochs={epochs}",
            flush=True,
        )
    train_ddpm(
        model,
        x_train,
        schedule,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        logger=None,
    )
    serve = nano_unet_serve_dir(root)
    serve.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "base_channels": base_channels,
        "timesteps": timesteps,
        "img_size": 32,
        "in_channels": 3,
    }
    torch.save(payload, serve / "model.pt")
    meta = {
        "registry_key": SERVE_KEY,
        "exp_id": "exp_102",
        "profile": profile,
        "seed": seed,
        "n_params": model.count_parameters(),
        "base_channels": base_channels,
        "timesteps": timesteps,
        "img_size": 32,
        "shipped_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "hardware_note": "trained for serve on local workstation GPU when available",
    }
    (serve / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - t0
    log_event(
        "info",
        "nano_unet_cifar ship complete",
        registry_key=SERVE_KEY,
        serve_dir=str(serve),
        elapsed_s=round(elapsed, 3),
    )
    result = NanoUNetShipResult(
        serve_dir=serve,
        n_params=model.count_parameters(),
        epochs=epochs,
        timesteps=timesteps,
        base_channels=base_channels,
        device=str(device),
        profile=profile,
        elapsed_s=elapsed,
    )
    if verbose:
        print(f"wrote {serve / 'model.pt'} params={result.n_params:,} elapsed={elapsed:.1f}s", flush=True)
    return result
