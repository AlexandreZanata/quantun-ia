"""
EXP 108 — Quantum flow coupling in TinyDiT mid-block (Phase J / H-Q3.3).

Publication (RTX 4060):
  MLFLOW_DISABLE=1 QML_DEVICE=cuda \\
    python experiments/exp_108_quantum_flow_coupling/run.py --profile publication --write-results
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

from src.classical.tiny_dit import TinyDiT
from src.data.open_images import load_cifar10_nchw
from src.training.config import load_experiment_config
from src.training.device import resolve_device
from src.training.image_ddpm import DDPMSchedule, sample_ddpm, train_ddpm
from src.training.image_fid import fid_r18
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id

EXP_KEY = "exp_108_quantum_flow_coupling"
EXP_ID = "exp_108"


@dataclass(frozen=True)
class FlowCouplingResult:
    classical_params: int
    unitary_params: int
    fid_classical: float
    fid_unitary: float
    fid_relative_improvement: float
    final_loss_classical: float
    final_loss_unitary: float
    elapsed_s: float
    device: str
    profile: str
    hypothesis_confirmed: bool


def gate_passed(result: FlowCouplingResult, *, min_rel: float = 0.05) -> bool:
    return bool(result.hypothesis_confirmed and result.fid_relative_improvement >= min_rel)


def _train_arm(
    *,
    coupling: str,
    x_train: torch.Tensor,
    schedule: DDPMSchedule,
    cfg: dict,
    seed: int,
    logger: ExperimentLogger | None,
) -> tuple[TinyDiT, list[float]]:
    model = TinyDiT(
        img_size=32,
        patch_size=int(cfg.get("patch_size", 4)),
        dim=int(cfg.get("dim", 64)),
        depth=int(cfg.get("depth", 4)),
        n_heads=int(cfg.get("n_heads", 4)),
        time_dim=int(cfg.get("time_dim", 128)),
        coupling=coupling,
        coupling_layers=int(cfg.get("coupling_layers", 2)),
    )
    history = train_ddpm(
        model,
        x_train,
        schedule,
        epochs=int(cfg.get("epochs", 2)),
        batch_size=int(cfg.get("batch_size", 8)),
        lr=float(cfg.get("lr", 2e-4)),
        seed=seed,
        logger=logger,
    )
    return model, history


def run_exp_108(*, profile: str = "ci", verbose: bool = True) -> FlowCouplingResult:
    os.environ.setdefault("MLFLOW_DISABLE", "1")
    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    n_train = int(cfg.get("n_train", 32))
    n_val = int(cfg.get("n_val", 16))
    n_fid = int(cfg.get("n_fid_samples", 16))
    timesteps = int(cfg.get("timesteps", 20))
    min_rel = float(cfg.get("min_relative_fid_improvement", 0.05))

    init_correlation_id()
    device = resolve_device()
    torch.manual_seed(seed)

    x_train_np, _ = load_cifar10_nchw(split="train", n_take=n_train, seed=seed)
    x_val_np, _ = load_cifar10_nchw(split="val", n_take=max(n_val, n_fid), seed=seed + 1)
    x_train = torch.from_numpy(x_train_np)
    x_val = torch.from_numpy(x_val_np)
    real_ref = x_val[:n_fid].to(device)
    if real_ref.shape[0] < n_fid:
        reps = (n_fid + real_ref.shape[0] - 1) // real_ref.shape[0]
        real_ref = real_ref.repeat(reps, 1, 1, 1)[:n_fid]

    schedule = DDPMSchedule(timesteps=timesteps, device=device)
    logger = ExperimentLogger(EXP_ID, "quantum_flow_coupling", seed=seed, profile=profile)
    t0 = time.perf_counter()

    if verbose:
        print(
            f"device={device} n_train={n_train} epochs={cfg.get('epochs')} "
            f"T={timesteps} dim={cfg.get('dim')} depth={cfg.get('depth')}",
            flush=True,
        )
        print("training classical affine coupling TinyDiT …", flush=True)

    model_c, hist_c = _train_arm(
        coupling="classical",
        x_train=x_train,
        schedule=schedule,
        cfg=cfg,
        seed=seed,
        logger=logger,
    )
    if verbose:
        print("training unitary (quantum-inspired) coupling TinyDiT …", flush=True)
    model_u, hist_u = _train_arm(
        coupling="unitary",
        x_train=x_train,
        schedule=schedule,
        cfg=cfg,
        seed=seed + 7,
        logger=logger,
    )

    if verbose:
        print("sampling + FID-R18 …", flush=True)
    samples_c = sample_ddpm(model_c, schedule, n=n_fid, shape=(3, 32, 32))
    samples_u = sample_ddpm(model_u, schedule, n=n_fid, shape=(3, 32, 32))
    fid_c = float(fid_r18(real_ref, samples_c, device=device))
    fid_u = float(fid_r18(real_ref, samples_u, device=device))
    rel = (fid_c - fid_u) / max(fid_c, 1e-8)
    confirmed = rel >= min_rel

    elapsed = time.perf_counter() - t0
    result = FlowCouplingResult(
        classical_params=model_c.count_parameters(),
        unitary_params=model_u.count_parameters(),
        fid_classical=fid_c,
        fid_unitary=fid_u,
        fid_relative_improvement=rel,
        final_loss_classical=float(hist_c[-1]) if hist_c else float("nan"),
        final_loss_unitary=float(hist_u[-1]) if hist_u else float("nan"),
        elapsed_s=elapsed,
        device=str(device),
        profile=profile,
        hypothesis_confirmed=confirmed,
    )
    logger.finish(
        elapsed,
        eval_set="cifar10_val",
        fid_classical=fid_c,
        fid_unitary=fid_u,
        fid_relative_improvement=rel,
        hypothesis_confirmed=confirmed,
        device=str(device),
    )
    if verbose:
        print(
            f"FID c={fid_c:.2f} u={fid_u:.2f} rel={rel:.3f} "
            f"confirmed={confirmed} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    return result


def write_results(result: FlowCouplingResult, path: Path) -> None:
    verdict = "Confirmed" if result.hypothesis_confirmed else "Rejected"
    lines = [
        f"# Results — EXP 108: Quantum flow coupling TinyDiT ({date.today().isoformat()})",
        "",
        f"**Verdict:** {verdict}",
        f"**Profile:** `{result.profile}` · **Device:** `{result.device}`",
        f"**Classical params:** {result.classical_params:,} · **Unitary params:** {result.unitary_params:,}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| FID-R18 classical coupling | {result.fid_classical:.2f} |",
        f"| FID-R18 unitary coupling | {result.fid_unitary:.2f} |",
        f"| Relative improvement | {result.fid_relative_improvement:.3f} |",
        f"| Final loss classical | {result.final_loss_classical:.4f} |",
        f"| Final loss unitary | {result.final_loss_unitary:.4f} |",
        f"| Elapsed (s) | {result.elapsed_s:.1f} |",
        "",
        "## Gate (H-Q3.3)",
        "",
        "- Win: `FID_unitary ≤ FID_classical × 0.95` (≥ 5% relative).",
        f"- Outcome: **{verdict}** (rel={result.fid_relative_improvement:.3f}).",
        "",
        "## Ablation suggestion",
        "",
        "- What if you move the unitary coupling to every other block (full flow stack)?",
        "",
        f"*Logged via ExperimentLogger · {datetime.now().isoformat(timespec='microseconds')}*",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()
    result = run_exp_108(profile=args.profile, verbose=True)
    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        write_results(result, out)
        print(f"wrote {out}", flush=True)
    return 0 if result.hypothesis_confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())
