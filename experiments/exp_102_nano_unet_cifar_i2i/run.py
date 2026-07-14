"""
EXP 102 — NanoUNet CIFAR-10 I2I classical floor (Phase H).

Publication (RTX 4060):
  MLFLOW_DISABLE=1 QML_DEVICE=cuda \\
    python experiments/exp_102_nano_unet_cifar_i2i/run.py --profile publication --write-results
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

from src.classical.nano_unet import NanoUNet
from src.data.open_images import load_cifar10_nchw
from src.training.config import load_experiment_config
from src.training.device import resolve_device
from src.training.image_ddpm import DDPMSchedule, sample_ddpm, train_ddpm
from src.training.image_fid import fid_r18, lpips_proxy_mean
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id

EXP_KEY = "exp_102_nano_unet_cifar_i2i"
EXP_ID = "exp_102"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NanoUNetCifarResult:
    n_params: int
    final_train_loss: float
    val_denoise_mse: float
    fid_model: float
    fid_noise: float
    fid_relative_improvement: float
    lpips_proxy: float
    elapsed_s: float
    device: str
    profile: str
    hypothesis_confirmed: bool


def gate_passed(result: NanoUNetCifarResult, *, min_rel_fid: float = 0.20) -> bool:
    return bool(result.hypothesis_confirmed and result.fid_relative_improvement >= min_rel_fid)


def run_exp_102(*, profile: str = "ci", verbose: bool = True) -> NanoUNetCifarResult:
    os.environ.setdefault("MLFLOW_DISABLE", "1")
    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    epochs = int(cfg.get("epochs", 2))
    batch_size = int(cfg.get("batch_size", 8))
    lr = float(cfg.get("lr", 2e-4))
    timesteps = int(cfg.get("timesteps", 20))
    base_channels = int(cfg.get("base_channels", 32))
    n_train = int(cfg.get("n_train", 32))
    n_val = int(cfg.get("n_val", 16))
    n_fid = int(cfg.get("n_fid_samples", 16))
    min_rel = float(cfg.get("min_relative_fid_improvement", 0.20))

    init_correlation_id()
    device = resolve_device()
    torch.manual_seed(seed)

    x_train_np, _ = load_cifar10_nchw(split="train", n_take=n_train, seed=seed)
    x_val_np, _ = load_cifar10_nchw(split="val", n_take=n_val, seed=seed + 1)
    x_train = torch.from_numpy(x_train_np)
    x_val = torch.from_numpy(x_val_np)

    model = NanoUNet(in_channels=3, base_channels=base_channels)
    n_params = model.count_parameters()
    schedule = DDPMSchedule(timesteps=timesteps, device=device)

    logger = ExperimentLogger(EXP_ID, "nano_unet_ddpm", seed=seed, profile=profile)
    t0 = time.perf_counter()
    if verbose:
        print(
            f"device={device} params={n_params:,} n_train={n_train} "
            f"epochs={epochs} T={timesteps} base_ch={base_channels}",
            flush=True,
        )

    history = train_ddpm(
        model,
        x_train,
        schedule,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        logger=logger,
    )
    final_loss = float(history[-1]) if history else float("nan")
    val_metrics = model.evaluate(x_val.to(device), timesteps=timesteps)
    val_mse = float(val_metrics["denoise_mse"])

    with torch.no_grad():
        model_samples = sample_ddpm(model, schedule, n=n_fid, shape=(3, 32, 32))
        noise_samples = torch.randn(n_fid, 3, 32, 32, device=device).clamp(-1, 1)
        real_ref = x_val[:n_fid].to(device)
        if real_ref.shape[0] < n_fid:
            # tile if val slice smaller than n_fid
            reps = (n_fid + real_ref.shape[0] - 1) // real_ref.shape[0]
            real_ref = real_ref.repeat(reps, 1, 1, 1)[:n_fid]

        fid_model = fid_r18(real_ref, model_samples, device=device)
        fid_noise = fid_r18(real_ref, noise_samples, device=device)
        rel = 1.0 - (fid_model / fid_noise) if fid_noise > 0 else 0.0
        lpips = lpips_proxy_mean(real_ref, model_samples, device=device, n_pairs=min(64, n_fid))

    elapsed = time.perf_counter() - t0
    confirmed = rel >= min_rel

    logger.finish(
        elapsed,
        test_accuracy=None,
        eval_set="cifar10_val",
        n_params=n_params,
        final_train_loss=final_loss,
        val_denoise_mse=val_mse,
        fid_r18_model=fid_model,
        fid_r18_noise=fid_noise,
        fid_relative_improvement=rel,
        lpips_proxy=lpips,
        hypothesis_confirmed=confirmed,
        device=str(device),
    )

    result = NanoUNetCifarResult(
        n_params=n_params,
        final_train_loss=final_loss,
        val_denoise_mse=val_mse,
        fid_model=fid_model,
        fid_noise=fid_noise,
        fid_relative_improvement=rel,
        lpips_proxy=lpips,
        elapsed_s=elapsed,
        device=str(device),
        profile=profile,
        hypothesis_confirmed=confirmed,
    )
    if verbose:
        print(
            f"loss={final_loss:.4f} val_mse={val_mse:.6f} "
            f"FID-R18 model={fid_model:.2f} noise={fid_noise:.2f} "
            f"rel_improv={rel:.3f} LPIPS-proxy={lpips:.4f} "
            f"confirmed={confirmed} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    return result


def write_results_md(result: NanoUNetCifarResult) -> Path:
    path = Path(__file__).resolve().parent / "results.md"
    verdict = "Confirmed" if result.hypothesis_confirmed else "Rejected"
    path.write_text(
        "\n".join(
            [
                f"# Results — EXP 102: NanoUNet CIFAR-10 I2I ({date.today().isoformat()})",
                "",
                f"**Verdict:** {verdict}",
                f"**Profile:** `{result.profile}` · **Device:** `{result.device}`",
                f"**Params:** {result.n_params:,}",
                "",
                "## Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Final train noise-MSE | {result.final_train_loss:.4f} |",
                f"| Val denoise MSE | {result.val_denoise_mse:.6f} |",
                f"| FID-R18 (model vs val) | {result.fid_model:.2f} |",
                f"| FID-R18 (noise null vs val) | {result.fid_noise:.2f} |",
                f"| Relative FID improvement | {result.fid_relative_improvement:.3f} |",
                f"| LPIPS-proxy (VGG) | {result.lpips_proxy:.4f} |",
                f"| Elapsed (s) | {result.elapsed_s:.1f} |",
                "",
                "## Gate",
                "",
                "- Primary H0: relative FID improvement ≥ 0.20 vs noise null.",
                f"- Outcome: **{verdict}** (Δ_rel = {result.fid_relative_improvement:.3f}).",
                "",
                "## Ablation suggestion",
                "",
                "- What if you halve `base_channels` (capacity) vs double training epochs?",
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
    result = run_exp_102(profile=args.profile)
    if args.write_results:
        out = write_results_md(result)
        print(f"wrote {out}", flush=True)
    return 0 if gate_passed(result) or args.profile == "ci" else 1


if __name__ == "__main__":
    raise SystemExit(main())
