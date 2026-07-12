"""LargeNanoMLP backbone with frozen weights and trainable 4-qubit hybrid head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.classical.large_nano_mlp import LargeNanoMLP
from src.quantum.hybrid_model import make_quantum_layer
from src.quantum.noise_regularized_qnn import make_noisy_quantum_layer
from src.training.base_model import TrainableMixin


def _resolve_backbone_device(prefer: str | None = None) -> torch.device:
    choice = (prefer or "cpu").lower()
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LargeNanoHybrid(TrainableMixin, nn.Module):
    """
    Frozen LargeNanoMLP backbone (CUDA when available) + PennyLane QNN head (CPU).

    Classical blocks run on GPU; TorchLayer stays on CPU per PennyLane constraints.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden1: int = 2048,
        hidden2: int = 512,
        hidden3: int = 64,
        dropout: float = 0.3,
        n_qubits: int = 4,
        n_layers: int = 2,
        reupload: bool = True,
        depolarizing_p: float = 0.0,
        backbone_device: str | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden3 = hidden3
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.reupload = reupload
        self.depolarizing_p = float(depolarizing_p)
        self._backbone_device = _resolve_backbone_device(backbone_device)
        template = LargeNanoMLP(
            input_dim=input_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            hidden3=hidden3,
            dropout=dropout,
        )
        self.backbone = nn.Sequential(*list(template.net.children())[:-2])
        self.head_proj = nn.Linear(hidden3, n_qubits)
        if self.depolarizing_p > 0.0:
            self.qlayer = make_noisy_quantum_layer(
                n_qubits,
                n_layers,
                reupload=reupload,
                depolarizing_p=self.depolarizing_p,
            )
        else:
            self.qlayer = make_quantum_layer(n_qubits, n_layers, reupload=reupload)
        self.post = nn.Sequential(nn.Linear(n_qubits, 1), nn.Sigmoid())
        self._backbone_frozen = False
        if self._backbone_device.type == "cuda":
            self.backbone.to(self._backbone_device)

    def set_qlayer_trainable(self, *, enabled: bool) -> None:
        """Freeze or unfreeze PennyLane TorchLayer weights (head warm-start)."""
        for param in self.qlayer.parameters():
            param.requires_grad = enabled

    def export_noiseless_head_state(self) -> dict[str, torch.Tensor]:
        """Return head weights compatible with a noiseless LargeNanoHybrid (p=0)."""
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if not k.startswith("backbone.")
        }

    def freeze_backbone(self) -> None:
        """Stop gradient flow through the million-param classical trunk."""
        self._backbone_frozen = True
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def load_frozen_backbone_from_large_nano(self, state_dict: dict[str, Any]) -> int:
        """Copy LargeNanoMLP trunk weights (layers 0–8) into the frozen backbone."""
        mapped: dict[str, torch.Tensor] = {}
        for name in self.backbone.state_dict():
            src_key = f"net.{name}"
            if src_key in state_dict:
                mapped[name] = state_dict[src_key]
        missing = set(self.backbone.state_dict()) - set(mapped)
        if missing:
            raise KeyError(f"missing backbone weights: {sorted(missing)}")
        self.backbone.load_state_dict(mapped, strict=True)
        return len(mapped)

    def _backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        x_bb = x.to(self._backbone_device)
        if self._backbone_frozen:
            with torch.no_grad():
                features = self.backbone(x_bb)
            return features.to(torch.device("cpu"))
        features = self.backbone(x_bb)
        return features.to(torch.device("cpu"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._backbone_features(x)
        h = torch.tanh(self.head_proj(h))
        h = self.qlayer(h)
        return self.post(h).squeeze(-1)

    def to(self, *args: Any, **kwargs: Any) -> LargeNanoHybrid:
        super().to(*args, **kwargs)
        if self._backbone_device.type == "cuda":
            self.backbone.to(self._backbone_device)
        return self
