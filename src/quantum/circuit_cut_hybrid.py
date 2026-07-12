"""Frozen ResidualNano + circuit-cut effective 6q head (two overlapping 4q fragments)."""

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


class CircuitCutSixQubitHybrid(TrainableMixin, nn.Module):
    """Frozen ResidualNano trunk + effective 6q head via two overlapping 4q fragments.

    Bottleneck → 6 angles. Fragment A uses angles[0:4]; fragment B uses angles[2:6]
    (2-wire overlap). Concatenate Pauli-Z expectations (8-d) → linear → sigmoid.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden: int = 512,
        n_blocks: int = 3,
        bottleneck: int = 64,
        dropout: float = 0.2,
        n_layers: int = 2,
        reupload: bool = True,
        backbone_device: str | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.bottleneck = bottleneck
        self.n_effective_qubits = 6
        self.fragment_qubits = 4
        self.n_layers = n_layers
        self.reupload = reupload
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

        self.head_proj = nn.Linear(bottleneck, self.n_effective_qubits)
        self.qlayer_a = make_quantum_layer(self.fragment_qubits, n_layers, reupload=reupload)
        self.qlayer_b = make_quantum_layer(self.fragment_qubits, n_layers, reupload=reupload)
        self.post = nn.Linear(self.fragment_qubits * 2, 1)
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
        feats = self._backbone_features(x).to(self.head_proj.weight.device)
        angles = torch.tanh(self.head_proj(feats))
        frag_a = self.qlayer_a(angles[:, 0:4])
        frag_b = self.qlayer_b(angles[:, 2:6])
        q_out = torch.cat([frag_a, frag_b], dim=-1)
        return self.out_act(self.post(q_out)).squeeze(-1)

    def to(self, *args: Any, **kwargs: Any) -> CircuitCutSixQubitHybrid:
        super().to(*args, **kwargs)
        if self._backbone_device.type == "cuda":
            self.stem.to(self._backbone_device)
            self.blocks.to(self._backbone_device)
            self.feature_head.to(self._backbone_device)
        return self


class ClassicalBottleneckHead(TrainableMixin, nn.Module):
    """Frozen ResidualNano trunk + classical linear head on bottleneck features."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden: int = 512,
        n_blocks: int = 3,
        bottleneck: int = 64,
        dropout: float = 0.2,
        backbone_device: str | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck = bottleneck
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
        self.head = nn.Linear(bottleneck, 1)
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
        # Backbone may run on CUDA then return CPU; head follows trainer device.
        feats = self._backbone_features(x).to(self.head.weight.device)
        return self.out_act(self.head(feats)).squeeze(-1)

    def to(self, *args: Any, **kwargs: Any) -> ClassicalBottleneckHead:
        super().to(*args, **kwargs)
        if self._backbone_device.type == "cuda":
            self.stem.to(self._backbone_device)
            self.blocks.to(self._backbone_device)
            self.feature_head.to(self._backbone_device)
        return self
