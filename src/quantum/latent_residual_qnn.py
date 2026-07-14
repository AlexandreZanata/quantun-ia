"""Latent-space DDPM denoisers: classical MLP vs residual QNN (Phase J)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantum.hybrid_model import make_quantum_layer


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class LatentNoiseMLP(nn.Module):
    """Classical noise predictor for flat latents (B, D)."""

    def __init__(self, latent_dim: int, *, hidden: int = 64, time_dim: int = 32) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(t)
        return self.net(torch.cat([z_t, temb], dim=-1))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LatentResidualQNN(nn.Module):
    """Classical MLP path + 4q re-upload residual skip (H-Q3.1)."""

    def __init__(
        self,
        latent_dim: int,
        *,
        hidden: int = 64,
        time_dim: int = 32,
        n_qubits: int = 4,
        n_layers: int = 2,
        reupload: bool = True,
    ) -> None:
        super().__init__()
        self.classical = LatentNoiseMLP(latent_dim, hidden=hidden, time_dim=time_dim)
        self.n_qubits = n_qubits
        self.pre = nn.Sequential(nn.Linear(latent_dim + time_dim, n_qubits), nn.Tanh())
        self.qlayer = make_quantum_layer(n_qubits, n_layers, reupload=reupload)
        self.post = nn.Linear(n_qubits, latent_dim)
        self.time_mlp = self.classical.time_mlp

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # PennyLane TorchLayer expects CPU float tensors
        z_cpu = z_t if z_t.device.type == "cpu" else z_t.cpu()
        t_cpu = t if t.device.type == "cpu" else t.cpu()
        base = self.classical(z_cpu, t_cpu)
        temb = self.time_mlp(t_cpu)
        q_in = self.pre(torch.cat([z_cpu, temb], dim=-1))
        q_out = self.qlayer(q_in)
        residual = self.post(q_out)
        out = base + residual
        return out.to(z_t.device) if z_t.device.type != "cpu" else out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
