"""Classical and unitary mid-block couplings for TinyDiT (Phase J / H-Q3.3)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalAffineCoupling(nn.Module):
    """RealNVP-style affine coupling on the last feature dimension."""

    def __init__(self, dim: int, hidden: int | None = None) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("dim must be >= 2 for coupling")
        self.dim = dim
        self.split = dim // 2
        h = hidden if hidden is not None else max(dim, 32)
        out = 2 * (dim - self.split)
        self.net = nn.Sequential(
            nn.Linear(self.split, h),
            nn.SiLU(),
            nn.Linear(h, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        xa, xb = x[..., : self.split], x[..., self.split :]
        params = self.net(xa)
        log_s, t = params.chunk(2, dim=-1)
        s = torch.tanh(log_s)
        yb = xb * torch.exp(s) + t
        return torch.cat([xa, yb], dim=-1)


class UnitaryFlowCoupling(nn.Module):
    """
    Quantum-inspired volume-preserving coupling: adjacent feature pairs mixed
    by learnable Givens (SO(2)) rotations — a block-diagonal unitary on ℝ^D.
    """

    def __init__(self, dim: int, n_layers: int = 2) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("dim must be >= 2 for coupling")
        self.dim = dim
        self.n_layers = n_layers
        n_pairs = dim // 2
        self.angles = nn.Parameter(torch.zeros(n_layers, n_pairs))
        # Residual MLP after unitary remix (match classical capacity roughly)
        self.post = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.post[-1].weight)
        nn.init.zeros_(self.post[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = x
        d = self.dim
        for layer in range(self.n_layers):
            flat = h.reshape(-1, d)
            even = flat[:, 0::2]
            odd = flat[:, 1::2]
            n_pairs = even.shape[-1]
            theta = self.angles[layer, :n_pairs]
            c = torch.cos(theta)
            s = torch.sin(theta)
            # Broadcast over batch*tokens
            c = c.unsqueeze(0)
            s = s.unsqueeze(0)
            y0 = c * even - s * odd
            y1 = s * even + c * odd
            # Interleave back
            out = torch.empty_like(flat)
            out[:, 0::2] = y0
            out[:, 1::2] = y1
            if d % 2 == 1:
                out[:, -1] = flat[:, -1]
            h = out.view_as(h)
        return h + self.post(h)


def coupling_module(kind: str, dim: int, **kwargs: object) -> nn.Module:
    k = kind.lower().strip()
    if k in {"classical", "affine"}:
        hidden = kwargs.get("hidden")
        return ClassicalAffineCoupling(dim, hidden=int(hidden) if hidden is not None else None)
    if k in {"unitary", "quantum", "givens"}:
        n_layers = int(kwargs.get("n_layers", 2))  # type: ignore[arg-type]
        return UnitaryFlowCoupling(dim, n_layers=n_layers)
    raise ValueError(f"unknown coupling kind: {kind!r}")


class TinyDiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TinyDiT(nn.Module):
    """
    Patch Transformer denoiser for 32×32 RGB (Nano DiT floor for H-Q3.3).

    ``coupling`` is injected once after the first half of transformer blocks.
    """

    def __init__(
        self,
        *,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        dim: int = 64,
        depth: int = 4,
        n_heads: int = 4,
        time_dim: int = 128,
        coupling: str = "classical",
        coupling_layers: int = 2,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.grid = img_size // patch_size
        self.n_tokens = self.grid * self.grid
        patch_dim = in_channels * patch_size * patch_size

        self.time_mlp = nn.Sequential(
            _SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([TinyDiTBlock(dim, n_heads) for _ in range(depth)])
        self.mid_index = depth // 2
        self.coupling = coupling_module(
            coupling,
            dim,
            n_layers=coupling_layers,
            hidden=dim * 2,
        )
        self.coupling_kind = coupling
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, patch_dim)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.reshape(b, self.n_tokens, c * p * p)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        p = self.patch_size
        c = self.in_channels
        g = self.grid
        x = tokens.reshape(b, g, g, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.reshape(b, c, g * p, g * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(self._patchify(x)) + self.pos_embed
        t_emb = self.time_mlp(t).unsqueeze(1)
        tokens = tokens + t_emb
        for i, block in enumerate(self.blocks):
            tokens = block(tokens)
            if i + 1 == self.mid_index:
                tokens = self.coupling(tokens)
        tokens = self.out_proj(self.out_norm(tokens))
        return self._unpatchify(tokens)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _SinusoidalTimeEmbedding(nn.Module):
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
