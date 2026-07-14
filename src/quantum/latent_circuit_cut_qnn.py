"""Latent DDPM residual head with circuit-cut effective 6q (Phase J / H-Q3.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.quantum.hybrid_model import make_quantum_layer
from src.quantum.latent_residual_qnn import LatentNoiseMLP, SinusoidalTimeEmb


class LatentCircuitCutQNN(nn.Module):
    """Classical MLP path + effective 6q residual via two overlapping 4q fragments.

    Angles[0:4] → fragment A; angles[2:6] → fragment B (2-wire overlap).
    Concatenate Pauli-Z expectations (8-d) → linear → residual on latent noise.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        hidden: int = 64,
        time_dim: int = 32,
        n_layers: int = 2,
        reupload: bool = True,
    ) -> None:
        super().__init__()
        self.n_effective_qubits = 6
        self.fragment_qubits = 4
        self.classical = LatentNoiseMLP(latent_dim, hidden=hidden, time_dim=time_dim)
        self.time_mlp = self.classical.time_mlp
        self.pre = nn.Sequential(
            nn.Linear(latent_dim + time_dim, self.n_effective_qubits),
            nn.Tanh(),
        )
        self.qlayer_a = make_quantum_layer(self.fragment_qubits, n_layers, reupload=reupload)
        self.qlayer_b = make_quantum_layer(self.fragment_qubits, n_layers, reupload=reupload)
        self.post = nn.Linear(self.fragment_qubits * 2, latent_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z_cpu = z_t if z_t.device.type == "cpu" else z_t.cpu()
        t_cpu = t if t.device.type == "cpu" else t.cpu()
        base = self.classical(z_cpu, t_cpu)
        temb = self.time_mlp(t_cpu)
        angles = self.pre(torch.cat([z_cpu, temb], dim=-1))
        frag_a = self.qlayer_a(angles[:, 0:4])
        frag_b = self.qlayer_b(angles[:, 2:6])
        residual = self.post(torch.cat([frag_a, frag_b], dim=-1))
        out = base + residual
        return out.to(z_t.device) if z_t.device.type != "cpu" else out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Re-export so callers can share time emb helpers if needed
__all__ = ["LatentCircuitCutQNN", "LatentNoiseMLP", "SinusoidalTimeEmb"]
