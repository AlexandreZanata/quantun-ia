"""Unit / smoke tests for NanoVAE + latent residual QNN (ci-scale)."""

from __future__ import annotations

import torch

from src.classical.nano_vae import NanoVAE, train_nano_vae, vae_loss
from src.quantum.latent_residual_qnn import LatentNoiseMLP, LatentResidualQNN
from src.training.image_ddpm import DDPMSchedule
from src.training.latent_ddpm import q_sample_latent, sample_latent_ddpm, train_latent_ddpm


def test_nano_vae_shapes_and_train_smoke():
    device = torch.device("cpu")
    model = NanoVAE(latent_dim=8, base_channels=8)
    x = torch.randn(4, 3, 32, 32)
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape
    assert mu.shape == (4, 8)
    loss = vae_loss(recon, x, mu, logvar)
    assert float(loss) >= 0.0
    hist = train_nano_vae(model, x, epochs=1, batch_size=2, lr=1e-3, device=device, seed=0)
    assert len(hist) == 1


def test_latent_residual_qnn_forward():
    model = LatentResidualQNN(latent_dim=8, hidden=16, n_qubits=4, n_layers=1)
    z = torch.randn(2, 8)
    t = torch.tensor([0, 1])
    out = model(z, t)
    assert out.shape == z.shape


def test_latent_ddpm_classical_smoke():
    device = torch.device("cpu")
    schedule = DDPMSchedule(timesteps=4, device=device)
    model = LatentNoiseMLP(8, hidden=16)
    z = torch.randn(8, 8)
    z_t, noise = q_sample_latent(schedule, z[:2], torch.tensor([0, 1]))
    assert z_t.shape == (2, 8)
    hist = train_latent_ddpm(model, z, schedule, epochs=1, batch_size=4, lr=1e-3, seed=0)
    assert len(hist) == 1
    samples = sample_latent_ddpm(model, schedule, n=2, latent_dim=8)
    assert samples.shape == (2, 8)
