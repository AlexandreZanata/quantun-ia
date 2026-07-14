"""
EXP 107 — Patch amplitude bottleneck vs classical (Phase J / H-Q3.2).

Publication (RTX 4060 + PennyLane CPU):
  MLFLOW_DISABLE=1 QML_DEVICE=cuda \\
    python experiments/exp_107_patch_amplitude_bottleneck/run.py --profile publication --write-results
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.image_patches import extract_patches, reconstruct_from_patches
from src.data.open_images import load_cifar10_nchw
from src.quantum.patch_amplitude_bottleneck import (
    ClassicalPatchBottleneck,
    PatchAmplitudeBottleneck,
)
from src.training.config import load_experiment_config
from src.training.device import resolve_device
from src.training.image_fid import fid_r18
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id

EXP_KEY = "exp_107_patch_amplitude_bottleneck"
EXP_ID = "exp_107"
PATCH = 4
PATCH_DIM = 3 * PATCH * PATCH  # 48


@dataclass(frozen=True)
class PatchAmplitudeResult:
    classical_params: int
    quantum_params: int
    classical_mse: float
    quantum_mse: float
    fid_classical: float
    fid_quantum: float
    fid_abs_delta: float
    elapsed_s: float
    device: str
    profile: str
    hypothesis_confirmed: bool


def gate_passed(result: PatchAmplitudeResult, *, max_abs_delta: float = 1.0) -> bool:
    return bool(result.hypothesis_confirmed and result.fid_abs_delta <= max_abs_delta)


def _train_ae(
    model: torch.nn.Module,
    patches: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    logger=None,
) -> float:
    torch.manual_seed(seed)
    model.train()
    # Quantum model stays on CPU; classical can be CUDA
    if next(model.parameters()).device.type == "cpu":
        device = torch.device("cpu")
    else:
        model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(patches), batch_size=batch_size, shuffle=True)
    last = float("nan")
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = F.mse_loss(pred, batch.to(pred.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        last = running / max(n_batches, 1)
        if logger is not None:
            logger.log(epoch, loss=last)
    return last


@torch.no_grad()
def _reconstruct_images(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    model.eval()
    outs: list[torch.Tensor] = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i : i + batch_size]
        patches = extract_patches(batch, patch=PATCH)  # (B, P, 48)
        b, n_p, d = patches.shape
        flat = patches.reshape(b * n_p, d)
        # route device: classical CUDA, quantum CPU
        if next(model.parameters()).device.type == "cpu":
            recon_flat = model(flat.cpu())
        else:
            recon_flat = model(flat.to(device))
        recon_patches = recon_flat.view(b, n_p, d).to(batch.device)
        outs.append(reconstruct_from_patches(recon_patches, patch=PATCH).clamp(-1, 1))
    return torch.cat(outs, dim=0)


def run_exp_107(*, profile: str = "ci", verbose: bool = True) -> PatchAmplitudeResult:
    os.environ.setdefault("MLFLOW_DISABLE", "1")
    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    epochs_c = int(cfg.get("classical_epochs", 2))
    epochs_q = int(cfg.get("quantum_epochs", 1))
    batch_size = int(cfg.get("batch_size", 8))
    q_batch = int(cfg.get("quantum_batch_size", batch_size))
    lr = float(cfg.get("lr", 2e-4))
    n_train = int(cfg.get("n_train", 32))
    n_val = int(cfg.get("n_val", 16))
    n_fid = int(cfg.get("n_fid_samples", 16))
    max_abs = float(cfg.get("max_fid_abs_delta", 1.0))
    n_qubits = int(cfg.get("n_qubits", 4))
    n_layers = int(cfg.get("n_layers", 2))
    bottleneck = int(cfg.get("bottleneck", 16))

    init_correlation_id()
    device = resolve_device()
    cpu = torch.device("cpu")
    torch.manual_seed(seed)

    x_train_np, _ = load_cifar10_nchw(split="train", n_take=n_train, seed=seed)
    x_val_np, _ = load_cifar10_nchw(split="val", n_take=max(n_val, n_fid), seed=seed + 1)
    x_train = torch.from_numpy(x_train_np)
    x_val = torch.from_numpy(x_val_np)
    real_ref = x_val[:n_fid].to(device)

    train_patches = extract_patches(x_train, patch=PATCH).reshape(-1, PATCH_DIM)
    max_q_patches = int(cfg.get("max_quantum_patches", train_patches.shape[0]))
    if max_q_patches < train_patches.shape[0]:
        rng = torch.Generator().manual_seed(seed + 7)
        idx = torch.randperm(train_patches.shape[0], generator=rng)[:max_q_patches]
        quantum_patches = train_patches[idx]
    else:
        quantum_patches = train_patches

    classical = ClassicalPatchBottleneck(PATCH_DIM, bottleneck=bottleneck)
    quantum = PatchAmplitudeBottleneck(PATCH_DIM, n_qubits=n_qubits, n_layers=n_layers)

    logger = ExperimentLogger(EXP_ID, "patch_amplitude_bottleneck", seed=seed, profile=profile)
    t0 = time.perf_counter()
    if verbose:
        print(
            f"device={device} n_train={n_train} patches={train_patches.shape[0]} "
            f"classical_ep={epochs_c} quantum_ep={epochs_q}",
            flush=True,
        )
        print("training classical patch bottleneck …", flush=True)

    mse_c = _train_ae(
        classical.to(device),
        train_patches,
        epochs=epochs_c,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
        logger=None,
    )

    if verbose:
        print("training quantum amplitude bottleneck (CPU) …", flush=True)
    mse_q = _train_ae(
        quantum.to(cpu),
        quantum_patches,
        epochs=epochs_q,
        batch_size=q_batch,
        lr=lr,
        device=cpu,
        seed=seed + 1,
        logger=logger,
    )

    if verbose:
        print("reconstruct + FID-R18 …", flush=True)
    imgs_c = _reconstruct_images(classical, real_ref.cpu(), device=device, batch_size=min(32, n_fid))
    imgs_q = _reconstruct_images(quantum, real_ref.cpu(), device=cpu, batch_size=min(16, n_fid))
    fid_c = fid_r18(real_ref, imgs_c.to(device), device=device)
    fid_q = fid_r18(real_ref, imgs_q.to(device), device=device)
    abs_delta = abs(fid_q - fid_c)
    confirmed = abs_delta <= max_abs
    elapsed = time.perf_counter() - t0

    logger.finish(
        elapsed,
        eval_set="cifar10_val",
        classical_params=classical.count_parameters(),
        quantum_params=quantum.count_parameters(),
        classical_mse=mse_c,
        quantum_mse=mse_q,
        fid_r18_classical=fid_c,
        fid_r18_quantum=fid_q,
        fid_abs_delta=abs_delta,
        hypothesis_confirmed=confirmed,
        device=str(device),
    )

    result = PatchAmplitudeResult(
        classical_params=classical.count_parameters(),
        quantum_params=quantum.count_parameters(),
        classical_mse=mse_c,
        quantum_mse=mse_q,
        fid_classical=fid_c,
        fid_quantum=fid_q,
        fid_abs_delta=abs_delta,
        elapsed_s=elapsed,
        device=str(device),
        profile=profile,
        hypothesis_confirmed=confirmed,
    )
    if verbose:
        print(
            f"MSE c={mse_c:.4f} q={mse_q:.4f} FID c={fid_c:.2f} q={fid_q:.2f} "
            f"|Δ|={abs_delta:.2f} confirmed={confirmed} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    return result


def write_results_md(result: PatchAmplitudeResult) -> Path:
    path = Path(__file__).resolve().parent / "results.md"
    verdict = "Confirmed" if result.hypothesis_confirmed else "Rejected"
    path.write_text(
        "\n".join(
            [
                f"# Results — EXP 107: Patch amplitude bottleneck ({date.today().isoformat()})",
                "",
                f"**Verdict:** {verdict}",
                f"**Profile:** `{result.profile}` · **Device:** `{result.device}`",
                f"**Classical params:** {result.classical_params:,} · **Quantum params:** {result.quantum_params:,}",
                "",
                "## Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Classical patch MSE | {result.classical_mse:.4f} |",
                f"| Quantum patch MSE | {result.quantum_mse:.4f} |",
                f"| FID-R18 classical recon | {result.fid_classical:.2f} |",
                f"| FID-R18 quantum recon | {result.fid_quantum:.2f} |",
                f"| |Δ FID| | {result.fid_abs_delta:.2f} |",
                f"| Elapsed (s) | {result.elapsed_s:.1f} |",
                "",
                "## Gate (H-Q3.2)",
                "",
                "- Parity: `|FID_q − FID_classical| ≤ 1.0`.",
                f"- Outcome: **{verdict}** (|Δ|={result.fid_abs_delta:.2f}).",
                "",
                "## Ablation suggestion",
                "",
                "- What if you use grayscale 4×4 patches (exact 16-d amp, no RGB projection)?",
                "",
                f"*Logged via ExperimentLogger · {datetime.now().isoformat()}*",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("ci", "publication"), default="ci")
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args(argv)
    result = run_exp_107(profile=args.profile)
    if args.write_results:
        out = write_results_md(result)
        print(f"wrote {out}", flush=True)
    return 0 if gate_passed(result) or args.profile == "ci" else 1


if __name__ == "__main__":
    raise SystemExit(main())
