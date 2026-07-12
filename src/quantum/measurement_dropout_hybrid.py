"""Frozen ResidualNano + QNN head with stochastic measurement masking."""

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


class MeasurementDropoutHybrid(TrainableMixin, nn.Module):
    """Frozen ResidualNano trunk + 4-qubit re-upload head with measurement dropout.

    Measurement dropout applies Bernoulli masks to Pauli-Z expectation vectors
    (train-time regularization). Optional MC averaging at eval via ``forward_mc``.
    """

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
        reupload: bool = True,
        measurement_dropout: float = 0.2,
        backbone_device: str | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= measurement_dropout < 1.0:
            msg = f"measurement_dropout must be in [0, 1), got {measurement_dropout}"
            raise ValueError(msg)
        self.input_dim = input_dim
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.bottleneck = bottleneck
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.reupload = reupload
        self.measurement_dropout = float(measurement_dropout)
        self._force_mc_mask = False
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

        self.head_proj = nn.Linear(bottleneck, n_qubits)
        self.qlayer = make_quantum_layer(n_qubits, n_layers, reupload=reupload)
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

    def _mask_measurements(self, q_out: torch.Tensor) -> torch.Tensor:
        p = self.measurement_dropout
        if p <= 0.0:
            return q_out
        apply = self.training or self._force_mc_mask
        if not apply:
            return q_out
        keep = 1.0 - p
        mask = torch.bernoulli(torch.full_like(q_out, keep)) / max(keep, 1e-6)
        return q_out * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._backbone_features(x)
        q_in = torch.tanh(self.head_proj(feats))
        q_out = self._mask_measurements(self.qlayer(q_in))
        return self.out_act(self.post(q_out)).squeeze(-1)

    def forward_mc(self, x: torch.Tensor, *, n_samples: int = 16) -> torch.Tensor:
        """MC mean of stochastic measurement masks (eval-time calibration)."""
        if n_samples < 1:
            msg = "n_samples must be >= 1"
            raise ValueError(msg)
        if self.measurement_dropout <= 0.0 or n_samples == 1:
            was_training = self.training
            self.eval()
            try:
                return self.forward(x)
            finally:
                self.train(was_training)

        was_training = self.training
        self.eval()
        self._force_mc_mask = True
        try:
            samples: list[torch.Tensor] = []
            with torch.no_grad():
                for _ in range(n_samples):
                    samples.append(self.forward(x))
            return torch.stack(samples, dim=0).mean(dim=0)
        finally:
            self._force_mc_mask = False
            self.train(was_training)

    def to(self, *args: Any, **kwargs: Any) -> MeasurementDropoutHybrid:
        super().to(*args, **kwargs)
        if self._backbone_device.type == "cuda":
            self.stem.to(self._backbone_device)
            self.blocks.to(self._backbone_device)
            self.feature_head.to(self._backbone_device)
        return self
