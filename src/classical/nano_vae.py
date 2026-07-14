"""Compact convolutional VAE for 32x32 RGB (Phase J NanoVAE floor)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NanoVAE(nn.Module):
    """Encode 32x32 RGB → latent; decode back. Latent dim sized for 4-qubit heads."""

    def __init__(self, latent_dim: int = 8, *, base_channels: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        ch = base_channels
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 4, stride=2, padding=1),  # 16x16
            nn.SiLU(),
            nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1),  # 8x8
            nn.SiLU(),
            nn.Conv2d(ch * 2, ch * 4, 4, stride=2, padding=1),  # 4x4
            nn.SiLU(),
            nn.Flatten(),
        )
        flat = ch * 4 * 4 * 4
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (ch * 4, 4, 4)),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),  # 8
            nn.SiLU(),
            nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),  # 16
            nn.SiLU(),
            nn.ConvTranspose2d(ch, 3, 4, stride=2, padding=1),  # 32
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(self.fc_dec(z))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 0.001,
) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl


def train_nano_vae(
    model: NanoVAE,
    x_train: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
    beta: float = 0.001,
) -> list[float]:
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[float] = []
    for _ in range(epochs):
        perm = torch.randperm(x_train.shape[0])
        running = 0.0
        n_batches = 0
        model.train()
        for i in range(0, x_train.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            batch = x_train[idx].to(device)
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta=beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        history.append(running / max(n_batches, 1))
    return history


@torch.no_grad()
def encode_mu(model: NanoVAE, x: torch.Tensor, *, device: torch.device, batch_size: int = 128) -> torch.Tensor:
    model.eval()
    outs: list[torch.Tensor] = []
    for i in range(0, x.shape[0], batch_size):
        batch = x[i : i + batch_size].to(device)
        mu, _ = model.encode(batch)
        outs.append(mu.cpu())
    return torch.cat(outs, dim=0)
