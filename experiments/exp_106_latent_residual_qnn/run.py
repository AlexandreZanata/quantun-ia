"""
EXP 106 — Latent residual QNN on frozen NanoVAE (Phase J / H-Q3.1).

Publication (RTX 4060 + PennyLane CPU):
  MLFLOW_DISABLE=1 QML_DEVICE=cuda \\
    python experiments/exp_106_latent_residual_qnn/run.py --profile publication --write-results
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classical.nano_vae import NanoVAE, encode_mu, train_nano_vae
from src.data.open_images import load_cifar10_nchw
from src.quantum.latent_residual_qnn import LatentNoiseMLP, LatentResidualQNN
from src.training.config import load_experiment_config
from src.training.device import resolve_device
from src.training.image_ddpm import DDPMSchedule
from src.training.image_fid import fid_r18
from src.training.latent_ddpm import sample_latent_ddpm, train_latent_ddpm
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id

EXP_KEY = "exp_106_latent_residual_qnn"
EXP_ID = "exp_106"


@dataclass(frozen=True)
class LatentResidualResult:
    vae_params: int
    classical_params: int
    quantum_params: int
    vae_recon_loss: float
    fid_classical: float
    fid_quantum: float
    fid_delta: float
    parity_ok: bool
    advantage_ok: bool
    elapsed_s: float
    device: str
    profile: str
    hypothesis_confirmed: bool


def gate_passed(result: LatentResidualResult, *, parity_slack: float = 1.0) -> bool:
    return bool(result.hypothesis_confirmed and result.fid_quantum <= result.fid_classical + parity_slack)


def run_exp_106(*, profile: str = "ci", verbose: bool = True) -> LatentResidualResult:
    os.environ.setdefault("MLFLOW_DISABLE", "1")
    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    latent_dim = int(cfg.get("latent_dim", 8))
    vae_epochs = int(cfg.get("vae_epochs", 2))
    classical_epochs = int(cfg.get("classical_epochs", 2))
    quantum_epochs = int(cfg.get("quantum_epochs", 1))
    batch_size = int(cfg.get("batch_size", 8))
    quantum_batch = int(cfg.get("quantum_batch_size", batch_size))
    lr = float(cfg.get("lr", 2e-4))
    timesteps = int(cfg.get("timesteps", 10))
    n_train = int(cfg.get("n_train", 32))
    n_val = int(cfg.get("n_val", 16))
    n_fid = int(cfg.get("n_fid_samples", 16))
    parity_slack = float(cfg.get("parity_fid_slack", 1.0))
    advantage_gain = float(cfg.get("advantage_fid_gain", 2.0))
    n_qubits = int(cfg.get("n_qubits", 4))
    n_layers = int(cfg.get("n_layers", 2))

    init_correlation_id()
    device = resolve_device()
    cpu = torch.device("cpu")
    torch.manual_seed(seed)

    x_train_np, _ = load_cifar10_nchw(split="train", n_take=n_train, seed=seed)
    x_val_np, _ = load_cifar10_nchw(split="val", n_take=max(n_val, n_fid), seed=seed + 1)
    x_train = torch.from_numpy(x_train_np)
    x_val = torch.from_numpy(x_val_np)
    real_ref = x_val[:n_fid].to(device)
    if real_ref.shape[0] < n_fid:
        reps = (n_fid + real_ref.shape[0] - 1) // real_ref.shape[0]
        real_ref = real_ref.repeat(reps, 1, 1, 1)[:n_fid]

    logger = ExperimentLogger(EXP_ID, "latent_residual_qnn", seed=seed, profile=profile)
    t0 = time.perf_counter()

    vae = NanoVAE(latent_dim=latent_dim, base_channels=int(cfg.get("vae_base_channels", 32)))
    if verbose:
        print(f"device={device} training NanoVAE latent_dim={latent_dim} …", flush=True)
    vae_hist = train_nano_vae(
        vae,
        x_train,
        epochs=vae_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
    )
    vae_loss = float(vae_hist[-1]) if vae_hist else float("nan")
    vae_params = sum(p.numel() for p in vae.parameters())
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    z_train = encode_mu(vae, x_train, device=device)
    # Normalize latents for DDPM stability
    z_mean = z_train.mean(dim=0, keepdim=True)
    z_std = z_train.std(dim=0, keepdim=True).clamp_min(1e-3)
    z_train_n = (z_train - z_mean) / z_std

    classical = LatentNoiseMLP(latent_dim, hidden=int(cfg.get("head_hidden", 64)))
    quantum = LatentResidualQNN(
        latent_dim,
        hidden=int(cfg.get("head_hidden", 64)),
        n_qubits=n_qubits,
        n_layers=n_layers,
        reupload=True,
    )

    sched_cuda = DDPMSchedule(timesteps=timesteps, device=device)
    sched_cpu = DDPMSchedule(timesteps=timesteps, device=cpu)

    if verbose:
        print("training classical latent DDPM …", flush=True)
    train_latent_ddpm(
        classical.to(device),
        z_train_n,
        sched_cuda,
        epochs=classical_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        logger=None,
    )

    if verbose:
        print("training quantum residual latent DDPM (CPU QNN) …", flush=True)
    train_latent_ddpm(
        quantum.to(cpu),
        z_train_n,
        sched_cpu,
        epochs=quantum_epochs,
        batch_size=quantum_batch,
        lr=lr,
        seed=seed + 1,
        logger=logger,
    )

    if verbose:
        print("sampling + decoding + FID-R18 …", flush=True)

    with torch.no_grad():
        z_c = sample_latent_ddpm(classical, sched_cuda, n=n_fid, latent_dim=latent_dim)
        z_q = sample_latent_ddpm(quantum, sched_cpu, n=n_fid, latent_dim=latent_dim)
        # unnormalize and decode
        z_c = z_c * z_std.to(z_c.device) + z_mean.to(z_c.device)
        z_q = z_q * z_std.to(z_q.device) + z_mean.to(z_q.device)
        imgs_c = vae.decode(z_c.to(device)).clamp(-1, 1)
        imgs_q = vae.decode(z_q.to(device)).clamp(-1, 1)
        fid_c = fid_r18(real_ref, imgs_c, device=device)
        fid_q = fid_r18(real_ref, imgs_q, device=device)

    delta = fid_q - fid_c
    parity_ok = fid_q <= fid_c + parity_slack
    advantage_ok = fid_q <= fid_c - advantage_gain
    confirmed = parity_ok
    elapsed = time.perf_counter() - t0

    logger.finish(
        elapsed,
        eval_set="cifar10_val",
        vae_params=vae_params,
        classical_params=classical.count_parameters(),
        quantum_params=quantum.count_parameters(),
        vae_recon_loss=vae_loss,
        fid_r18_classical=fid_c,
        fid_r18_quantum=fid_q,
        fid_delta=delta,
        parity_ok=parity_ok,
        advantage_ok=advantage_ok,
        hypothesis_confirmed=confirmed,
        device=str(device),
    )

    result = LatentResidualResult(
        vae_params=vae_params,
        classical_params=classical.count_parameters(),
        quantum_params=quantum.count_parameters(),
        vae_recon_loss=vae_loss,
        fid_classical=fid_c,
        fid_quantum=fid_q,
        fid_delta=delta,
        parity_ok=parity_ok,
        advantage_ok=advantage_ok,
        elapsed_s=elapsed,
        device=str(device),
        profile=profile,
        hypothesis_confirmed=confirmed,
    )
    if verbose:
        print(
            f"FID classical={fid_c:.2f} quantum={fid_q:.2f} Δ={delta:.2f} "
            f"parity={parity_ok} advantage={advantage_ok} confirmed={confirmed} "
            f"elapsed_s={elapsed:.1f}",
            flush=True,
        )
    return result


def write_results_md(result: LatentResidualResult) -> Path:
    path = Path(__file__).resolve().parent / "results.md"
    verdict = "Confirmed" if result.hypothesis_confirmed else "Rejected"
    path.write_text(
        "\n".join(
            [
                f"# Results — EXP 106: Latent residual QNN ({date.today().isoformat()})",
                "",
                f"**Verdict:** {verdict}",
                f"**Profile:** `{result.profile}` · **Device:** `{result.device}`",
                f"**VAE params:** {result.vae_params:,} · Classical {result.classical_params:,} · "
                f"Quantum {result.quantum_params:,}",
                "",
                "## Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| VAE train loss | {result.vae_recon_loss:.4f} |",
                f"| FID-R18 classical latent | {result.fid_classical:.2f} |",
                f"| FID-R18 quantum residual | {result.fid_quantum:.2f} |",
                f"| Δ FID (q − classical) | {result.fid_delta:.2f} |",
                f"| Parity (≤ +1.0) | {result.parity_ok} |",
                f"| Advantage (≤ −2.0) | {result.advantage_ok} |",
                f"| Elapsed (s) | {result.elapsed_s:.1f} |",
                "",
                "## Gate (H-Q3.1)",
                "",
                "- Parity: quantum FID ≤ classical + 1.0; advantage only at ≤ classical − 2.0.",
                f"- Outcome: **{verdict}** (Δ={result.fid_delta:.2f}).",
                "",
                "## Ablation suggestion",
                "",
                "- What if you remove the classical MLP path (pure QNN residual only)?",
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
    result = run_exp_106(profile=args.profile)
    if args.write_results:
        out = write_results_md(result)
        print(f"wrote {out}", flush=True)
    return 0 if gate_passed(result) or args.profile == "ci" else 1


if __name__ == "__main__":
    raise SystemExit(main())
