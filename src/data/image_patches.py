"""4x4 patch extract/reconstruct helpers for CIFAR-scale images."""

from __future__ import annotations

import torch


def extract_patches(images: torch.Tensor, *, patch: int = 4) -> torch.Tensor:
    """NCHW → (N, n_patches, C*patch*patch) non-overlapping patches."""
    if images.ndim != 4:
        raise ValueError(f"expected NCHW, got {tuple(images.shape)}")
    n, c, h, w = images.shape
    if h % patch or w % patch:
        raise ValueError(f"spatial {(h, w)} not divisible by patch={patch}")
    # unfold: (N, C*p*p, L)
    unfolded = torch.nn.functional.unfold(images, kernel_size=patch, stride=patch)
    n_patches = unfolded.shape[-1]
    return unfolded.transpose(1, 2).contiguous().view(n, n_patches, c * patch * patch)


def reconstruct_from_patches(
    patches: torch.Tensor,
    *,
    patch: int = 4,
    channels: int = 3,
    height: int = 32,
    width: int = 32,
) -> torch.Tensor:
    """(N, n_patches, C*p*p) → NCHW via fold."""
    n, n_patches, flat = patches.shape
    expected = channels * patch * patch
    if flat != expected:
        raise ValueError(f"patch flat dim {flat} != {expected}")
    unfolded = patches.view(n, n_patches, flat).transpose(1, 2).contiguous()
    return torch.nn.functional.fold(
        unfolded,
        output_size=(height, width),
        kernel_size=patch,
        stride=patch,
    )
