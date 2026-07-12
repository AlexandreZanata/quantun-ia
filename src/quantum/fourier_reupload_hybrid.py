"""Fourier feature map into PennyLane re-upload QNN head on frozen ResidualNano."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.quantum.hybrid_model import make_quantum_layer
from src.training.base_model import TrainableMixin


def _resolve_backbone_device(prefer: str | None = None) -> torch.device:
    choice = (prefer or "cpu").lower()
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FourierAngleMap(nn.Module):
    """Project features → qubits, expand sin/cos harmonics, map back to qubit angles."""

    def __init__(self, input_dim: int, n_qubits: int, n_frequencies: int = 2) -> None:
        super().__init__()
        if n_frequencies < 1:
            msg = "n_frequencies must be >= 1"
            raise ValueError(msg)
        self.n_qubits = n_qubits
        self.n_frequencies = n_frequencies
        self.pre = nn.Linear(input_dim, n_qubits)
        self.post = nn.Linear(n_qubits * 2 * n_frequencies, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.pre(x))
        harmonics: list[torch.Tensor] = []
        for k in range(1, self.n_frequencies + 1):
            scale = float(2**k) * torch.pi
            harmonics.append(torch.sin(scale * z))
            harmonics.append(torch.cos(scale * z))
        stacked = torch.cat(harmonics, dim=-1)
        return torch.tanh(self.post(stacked))


class FourierReuploadHybrid(TrainableMixin, nn.Module):
    """Frozen ResidualNano trunk + flat or Fourier-mapped 4-qubit re-upload head."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden: int = 512,
        n_blocks: int = 3,
        bottleneck: int = 64,
        dropout: float = 0.2,
        n_qubits: int = 4,
        n_layers: int = 2,
        encoding: str = "flat",
        n_frequencies: int = 2,
        backbone_device: str | None = None,
    ) -> None:
        super().__init__()
        if encoding not in {"flat", "fourier"}:
            msg = f"encoding must be 'flat' or 'fourier', got {encoding}"
            raise ValueError(msg)
        self.input_dim = input_dim
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.bottleneck = bottleneck
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.n_frequencies = n_frequencies
        self._backbone_device = _resolve_backbone_device(backbone_device)

        template = ResidualNanoMLP(
            input_dim,
            hidden=hidden,
            n_blocks=n_blocks,
            bottleneck=bottleneck,
            dropout=dropout,
        )
        self.stem = template.stem
        self.blocks = template.blocks
        self.feature_head = nn.Sequential(*list(template.head.children())[:3])

        if encoding == "fourier":
            self.encoder: nn.Module = FourierAngleMap(bottleneck, n_qubits, n_frequencies)
        else:
            self.encoder = nn.Sequential(nn.Linear(bottleneck, n_qubits), nn.Tanh())
        self.qlayer = make_quantum_layer(n_qubits, n_layers, reupload=True)
        self.post = nn.Linear(n_qubits, 1)
        self.out_act = nn.Sigmoid()
        self._backbone_frozen = False
        if self._backbone_device.type == "cuda":
            self.stem.to(self._backbone_device)
            self.blocks.to(self._backbone_device)
            self.feature_head.to(self._backbone_device)

    def freeze_backbone(self) -> None:
        self._backbone_frozen = True
        for module in (self.stem, self.blocks, self.feature_head):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def load_frozen_backbone_from_residual_nano(self, state_dict: dict[str, Any]) -> int:
        stem_sd = {k[len("stem.") :]: v for k, v in state_dict.items() if k.startswith("stem.")}
        blocks_sd = {
            k[len("blocks.") :]: v for k, v in state_dict.items() if k.startswith("blocks.")
        }
        feat_sd: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("head.0.") or k.startswith("head.1.") or k.startswith("head.2."):
                feat_sd[k[len("head.") :]] = v
        self.stem.load_state_dict(stem_sd, strict=True)
        self.blocks.load_state_dict(blocks_sd, strict=True)
        self.feature_head.load_state_dict(feat_sd, strict=True)
        return len(stem_sd) + len(blocks_sd) + len(feat_sd)

    def _backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        x_bb = x.to(self._backbone_device)

        def _forward(inp: torch.Tensor) -> torch.Tensor:
            h = self.stem(inp)
            for block in self.blocks:
                h = block(h)
            return self.feature_head(h)

        if self._backbone_frozen:
            with torch.no_grad():
                features = _forward(x_bb)
            return features.to(torch.device("cpu"))
        return _forward(x_bb).to(torch.device("cpu"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._backbone_features(x)
        q_in = self.encoder(feats)
        q_out = self.qlayer(q_in)
        return self.out_act(self.post(q_out)).squeeze(-1)

    def to(self, *args: Any, **kwargs: Any) -> FourierReuploadHybrid:
        super().to(*args, **kwargs)
        if self._backbone_device.type == "cuda":
            self.stem.to(self._backbone_device)
            self.blocks.to(self._backbone_device)
            self.feature_head.to(self._backbone_device)
        return self
