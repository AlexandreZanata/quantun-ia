"""Fast tests for FID-R18 / LPIPS-proxy helpers."""

from __future__ import annotations

import torch

from src.training.image_fid import frechet_distance, lpips_proxy_mean


def test_frechet_distance_identical_is_near_zero():
    mu = torch.zeros(4)
    cov = torch.eye(4)
    assert frechet_distance(mu, cov, mu, cov) < 1e-6


def test_lpips_proxy_identical_near_zero():
    x = torch.zeros(2, 3, 32, 32)
    # avoid downloading weights multiple times in CI — only if this is slow we skip
    val = lpips_proxy_mean(x, x, device=torch.device("cpu"), n_pairs=2)
    assert val >= 0.0
    assert val < 1e-5
