"""CLIP text → classical MLP vs angle-encoded QNN fusion for TinyDiT (H-Q3.5)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.quantum.hybrid_model import make_quantum_layer


class ClassicalTextTokenFusion(nn.Module):
    """Null-quantum ablate: CLIP embedding → MLP → fusion vector."""

    def __init__(self, clip_dim: int = 512, out_dim: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, clip_emb: torch.Tensor) -> torch.Tensor:
        return self.net(clip_emb)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class QuantumTextTokenFusion(nn.Module):
    """CLIP embedding → tanh angles → 4q re-upload → fusion vector (H-Q3.5)."""

    def __init__(
        self,
        clip_dim: int = 512,
        out_dim: int = 64,
        *,
        n_qubits: int = 4,
        n_layers: int = 2,
        reupload: bool = True,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.pre = nn.Sequential(nn.Linear(clip_dim, n_qubits), nn.Tanh())
        self.qlayer = make_quantum_layer(n_qubits, n_layers, reupload=reupload)
        self.post = nn.Linear(n_qubits, out_dim)

    def forward(self, clip_emb: torch.Tensor) -> torch.Tensor:
        # PennyLane TorchLayer expects CPU float tensors
        x_cpu = clip_emb if clip_emb.device.type == "cpu" else clip_emb.cpu()
        angles = self.pre(x_cpu)
        q_out = self.qlayer(angles)
        out = self.post(q_out)
        return out.to(clip_emb.device) if clip_emb.device.type != "cpu" else out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
