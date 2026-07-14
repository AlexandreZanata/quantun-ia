"""FID-R18 and LPIPS-proxy metrics for image nano experiments (no extra deps)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights, VGG16_Weights


def _to_imagenet_rgb(x: torch.Tensor) -> torch.Tensor:
    """Map [-1,1] NCHW images to ImageNet-normalized 224x224 RGB."""
    x = ((x.clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class ResNet18Features(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        net = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.backbone.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(_to_imagenet_rgb(x))
        return feats.flatten(1)


def frechet_distance(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor) -> float:
    """Closed-form Fréchet distance between two Gaussians (FID core via scipy sqrtm)."""
    import numpy as np
    from scipy import linalg

    m1 = mu1.detach().cpu().numpy().astype(np.float64)
    m2 = mu2.detach().cpu().numpy().astype(np.float64)
    s1 = sigma1.detach().cpu().numpy().astype(np.float64)
    s2 = sigma2.detach().cpu().numpy().astype(np.float64)
    diff = m1 - m2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))
    return value


@torch.no_grad()
def feature_stats(feat_model: nn.Module, images: torch.Tensor, *, batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(feat_model.parameters()).device
    feats: list[torch.Tensor] = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i : i + batch_size].to(device)
        feats.append(feat_model(batch))
    all_feats = torch.cat(feats, dim=0).float()
    mu = all_feats.mean(dim=0)
    centered = all_feats - mu
    sigma = (centered.T @ centered) / max(all_feats.shape[0] - 1, 1)
    return mu, sigma


@torch.no_grad()
def fid_r18(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 64,
    feat_model: ResNet18Features | None = None,
) -> float:
    model = feat_model or ResNet18Features().to(device)
    model.to(device)
    mu_r, sig_r = feature_stats(model, real, batch_size=batch_size)
    mu_f, sig_f = feature_stats(model, fake, batch_size=batch_size)
    return frechet_distance(mu_r, sig_r, mu_f, sig_f)


class VGG16LpipsProxy(nn.Module):
    """Cheap LPIPS-style proxy: MSE on early VGG16 features (not Kalantari LPIPS)."""

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice = nn.Sequential(*list(vgg.children())[:16])  # through relu3_3
        self.slice.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fx = self.slice(_to_imagenet_rgb(x))
        fy = self.slice(_to_imagenet_rgb(y))
        return F.mse_loss(fx, fy)


@torch.no_grad()
def lpips_proxy_mean(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    device: torch.device,
    n_pairs: int = 64,
) -> float:
    proxy = VGG16LpipsProxy().to(device)
    n = min(n_pairs, real.shape[0], fake.shape[0])
    return float(proxy(real[:n].to(device), fake[:n].to(device)).item())
