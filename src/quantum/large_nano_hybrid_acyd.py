"""Frozen LargeNanoMLP + seasonal QNN head (angle vs amplitude encoding) for ACYD."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from src.classical.large_nano_mlp import LargeNanoMLP
from src.data.acyd_cyclic import N_CYCLIC_FEATURES, extract_acyd_seasonal_cyclic
from src.quantum.hybrid_model import make_quantum_layer
from src.quantum.large_nano_hybrid import _resolve_backbone_device
from src.quantum.qnn_amplitude import make_amplitude_circuit
from src.training.base_model import TrainableMixin

EncodingMode = Literal["angle_seasonal", "amplitude"]


class LargeNanoHybridAcyd(TrainableMixin, nn.Module):
    """
    Frozen C4 backbone (for classical baseline parity) + trainable seasonal QNN head.

    The QNN reads cyclic sin/cos features from raw ACYD inputs, not backbone hidden states.
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
        encoding: EncodingMode = "angle_seasonal",
        backbone_device: str | None = None,
    ):
        super().__init__()
        if n_qubits != N_CYCLIC_FEATURES:
            msg = f"seasonal ACYD head requires n_qubits={N_CYCLIC_FEATURES}, got {n_qubits}"
            raise ValueError(msg)

        self.input_dim = input_dim
        self.encoding: EncodingMode = encoding
        self.n_qubits = n_qubits
        self._backbone_device = _resolve_backbone_device(backbone_device)
        template = LargeNanoMLP(
            input_dim=input_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            hidden3=hidden3,
            dropout=dropout,
        )
        self.backbone = nn.Sequential(*list(template.net.children())[:-2])
        self._backbone_frozen = False

        if encoding == "angle_seasonal":
            self.qlayer = make_quantum_layer(n_qubits, n_layers, reupload=reupload)
            self.post = nn.Sequential(nn.Linear(n_qubits, 1), nn.Sigmoid())
            self.amp_proj = None
        elif encoding == "amplitude":
            amp_dim = 2**n_qubits
            self.amp_proj = nn.Linear(N_CYCLIC_FEATURES, amp_dim)
            self.qlayer = make_amplitude_circuit(n_qubits, n_layers)
            self.post = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        else:
            msg = f"unsupported encoding: {encoding}"
            raise ValueError(msg)

        if self._backbone_device.type == "cuda":
            self.backbone.to(self._backbone_device)

    def freeze_backbone(self) -> None:
        self._backbone_frozen = True
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def load_frozen_backbone_from_large_nano(self, state_dict: dict[str, Any]) -> int:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cyclic = extract_acyd_seasonal_cyclic(x)
        if self.encoding == "angle_seasonal":
            h = torch.tanh(cyclic)
            h = self.qlayer(h)
            return self.post(h).squeeze(-1)

        assert self.amp_proj is not None
        h = self.amp_proj(cyclic)
        norm = h.norm(dim=1, keepdim=True).clamp(min=1e-8)
        h = h / norm
        out = self.qlayer(h).unsqueeze(1)
        return self.post(out).squeeze(-1)

    def to(self, *args: Any, **kwargs: Any) -> LargeNanoHybridAcyd:
        super().to(*args, **kwargs)
        if self._backbone_device.type == "cuda":
            self.backbone.to(self._backbone_device)
        return self
