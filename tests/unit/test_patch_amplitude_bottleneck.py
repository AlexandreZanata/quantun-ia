"""Fast tests for patch helpers and amplitude bottleneck."""

from __future__ import annotations

import torch

from src.data.image_patches import extract_patches, reconstruct_from_patches
from src.quantum.patch_amplitude_bottleneck import (
    ClassicalPatchBottleneck,
    PatchAmplitudeBottleneck,
)


def test_extract_reconstruct_roundtrip():
    x = torch.randn(2, 3, 32, 32)
    patches = extract_patches(x, patch=4)
    assert patches.shape == (2, 64, 48)
    recon = reconstruct_from_patches(patches, patch=4)
    assert torch.allclose(recon, x, atol=1e-5)


def test_classical_bottleneck_shape():
    model = ClassicalPatchBottleneck(48, bottleneck=16)
    x = torch.randn(4, 48)
    assert model(x).shape == x.shape


def test_amplitude_bottleneck_shape():
    model = PatchAmplitudeBottleneck(48, n_qubits=4, n_layers=1)
    x = torch.randn(2, 48)
    out = model(x)
    assert out.shape == x.shape
