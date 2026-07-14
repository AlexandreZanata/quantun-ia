"""Image Nano Lab predict — sample from shipped NanoUNet (Phase K / K-T2)."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from src.application.image_nano_ship import is_nano_unet_shipped, nano_unet_serve_dir
from src.classical.nano_unet import NanoUNet
from src.shared.result import Result, fail, ok
from src.training.device import resolve_device
from src.training.image_ddpm import DDPMSchedule, sample_ddpm


class ImagePredictError:
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ImageSampleResult:
    n: int
    img_size: int
    timesteps: int
    png_base64: list[str]
    model_key: str
    device: str


def load_image_nano_model_card(root: Path | None = None) -> dict:
    serve = nano_unet_serve_dir(root)
    meta_path = serve / "meta.json"
    if not meta_path.is_file():
        return {
            "registry_key": "nano_unet_cifar",
            "ready": False,
            "message": "bundle missing — run make ship-nano-unet-cifar",
        }
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["ready"] = is_nano_unet_shipped(root)
    return meta


def _tensor_to_png_b64(x: torch.Tensor) -> str:
    """x: (C,H,W) in [-1,1] → PNG base64."""
    import numpy as np
    from PIL import Image

    arr = ((x.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def predict_image_i2i(
    *,
    n: int = 4,
    timesteps: int | None = None,
    root: Path | None = None,
    seed: int = 42,
) -> Result[ImageSampleResult, ImagePredictError]:
    if n < 1 or n > 16:
        return fail(ImagePredictError("INVALID_N", "n must be in [1, 16]"))
    if not is_nano_unet_shipped(root):
        return fail(
            ImagePredictError(
                "BUNDLE_MISSING",
                "nano_unet_cifar not shipped — run make ship-nano-unet-cifar",
            )
        )
    serve = nano_unet_serve_dir(root)
    blob = torch.load(serve / "model.pt", map_location="cpu", weights_only=False)
    base_channels = int(blob.get("base_channels", 32))
    default_t = int(blob.get("timesteps", 20))
    img_size = int(blob.get("img_size", 32))
    t_steps = int(timesteps) if timesteps is not None else default_t
    device = resolve_device()
    torch.manual_seed(seed)
    model = NanoUNet(in_channels=3, base_channels=base_channels)
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    schedule = DDPMSchedule(timesteps=t_steps, device=device)
    with torch.no_grad():
        samples = sample_ddpm(model, schedule, n=n, shape=(3, img_size, img_size))
    pngs = [_tensor_to_png_b64(samples[i]) for i in range(n)]
    return ok(
        ImageSampleResult(
            n=n,
            img_size=img_size,
            timesteps=t_steps,
            png_base64=pngs,
            model_key="nano_unet_cifar",
            device=str(device),
        )
    )
