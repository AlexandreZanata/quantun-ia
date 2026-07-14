"""
EXP 105b — GV-ALR on NanoUNet DDPM (Phase I / H-I3).

Publication (RTX 4060):
  MLFLOW_DISABLE=1 QML_DEVICE=cuda \\
    python experiments/exp_105b_gv_alr_image_ddpm/run.py --profile publication --write-results
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
from src.training.adaptive_lr import AdaptiveLRConfig
from src.training.config import load_experiment_config
from src.training.device import resolve_device
from src.training.image_ddpm import DDPMSchedule, sample_ddpm, train_ddpm, train_ddpm_gvalr
from src.training.image_fid import fid_r18
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id

EXP_KEY = "exp_105b_gv_alr_image_ddpm"
EXP_ID = "exp_105b"


@dataclass(frozen=True)
class GvAlrImageResult:
    n_params: int
    fixed_epochs: int
    adaptive_epochs: int
    epoch_fraction: float
    fid_fixed: float
    fid_gvalr: float
    relative_fid_delta: float
    fixed_wall_s: float
    gvalr_wall_s: float
    elapsed_s: float
    device: str
    profile: str
    hypothesis_confirmed: bool


def gate_passed(
    result: GvAlrImageResult,
    *,
    max_epoch_fraction: float = 0.70,
    max_relative_fid_delta: float = 0.03,
) -> bool:
    epoch_ok = result.adaptive_epochs <= int(result.fixed_epochs * max_epoch_fraction + 1e-9)
    fid_ok = abs(result.relative_fid_delta) <= max_relative_fid_delta
    return bool(result.hypothesis_confirmed and epoch_ok and fid_ok)


def _fid(
    model: NanoUNet,
    schedule: DDPMSchedule,
    real_ref: torch.Tensor,
    *,
    n_fid: int,
    device: torch.device,
) -> float:
    samples = sample_ddpm(model, schedule, n=n_fid, shape=(3, 32, 32))
    return fid_r18(real_ref, samples, device=device)


def run_exp_105b(*, profile: str = "ci", verbose: bool = True) -> GvAlrImageResult:
    os.environ.setdefault("MLFLOW_DISABLE", "1")
    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    batch_size = int(cfg.get("batch_size", 8))
    lr = float(cfg.get("lr", 2e-4))
    timesteps = int(cfg.get("timesteps", 20))
    base_channels = int(cfg.get("base_channels", 32))
    n_train = int(cfg.get("n_train", 32))
    n_val = int(cfg.get("n_val", 16))
    n_fid = int(cfg.get("n_fid_samples", 16))
    fixed_epochs = int(cfg.get("fixed_epochs", 4))
    adaptive_epochs = int(cfg.get("adaptive_epochs", 2))
    max_epoch_fraction = float(cfg.get("max_epoch_fraction", 0.70))
    max_rel_fid = float(cfg.get("max_relative_fid_delta", 0.03))
    adapt_cfg_raw = dict(cfg.get("adaptive_lr", {}))
    adapt_cfg = AdaptiveLRConfig(
        base_lr=float(adapt_cfg_raw.get("base_lr", lr)),
        var_target=float(adapt_cfg_raw.get("var_target", 0.015)),
        min_scale=float(adapt_cfg_raw.get("min_scale", 0.25)),
        max_scale=float(adapt_cfg_raw.get("max_scale", 4.0)),
        warmup_epochs=int(adapt_cfg_raw.get("warmup_epochs", 1)),
        adapt_every=int(adapt_cfg_raw.get("adapt_every", 1)),
    )

    init_correlation_id()
    device = resolve_device()
    torch.manual_seed(seed)

    x_train_np, _ = load_cifar10_nchw(split="train", n_take=n_train, seed=seed)
    x_val_np, _ = load_cifar10_nchw(split="val", n_take=max(n_val, n_fid), seed=seed + 1)
    x_train = torch.from_numpy(x_train_np)
    real_ref = torch.from_numpy(x_val_np[:n_fid]).to(device)
    if real_ref.shape[0] < n_fid:
        reps = (n_fid + real_ref.shape[0] - 1) // real_ref.shape[0]
        real_ref = real_ref.repeat(reps, 1, 1, 1)[:n_fid]

    schedule = DDPMSchedule(timesteps=timesteps, device=device)
    model_fixed = NanoUNet(in_channels=3, base_channels=base_channels)
    model_gvalr = NanoUNet(in_channels=3, base_channels=base_channels)
    n_params = model_fixed.count_parameters()

    logger = ExperimentLogger(EXP_ID, "nano_unet_gvalr", seed=seed, profile=profile)
    t0 = time.perf_counter()
    if verbose:
        print(
            f"device={device} params={n_params:,} n_train={n_train} "
            f"fixed_ep={fixed_epochs} gvalr_ep={adaptive_epochs} T={timesteps}",
            flush=True,
        )
        print("training fixed-LR …", flush=True)

    t_fixed0 = time.perf_counter()
    train_ddpm(
        model_fixed,
        x_train,
        schedule,
        epochs=fixed_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        logger=None,
    )
    fixed_wall = time.perf_counter() - t_fixed0

    if verbose:
        print("training GV-ALR …", flush=True)
    t_gv0 = time.perf_counter()
    hist = train_ddpm_gvalr(
        model_gvalr,
        x_train,
        schedule,
        epochs=adaptive_epochs,
        batch_size=batch_size,
        adaptive_config=adapt_cfg,
        seed=seed + 1,
        logger=logger,
    )
    gvalr_wall = time.perf_counter() - t_gv0

    if verbose:
        print("computing FID-R18 …", flush=True)
    fid_fixed = _fid(model_fixed, schedule, real_ref, n_fid=n_fid, device=device)
    fid_gvalr = _fid(model_gvalr, schedule, real_ref, n_fid=n_fid, device=device)
    rel_delta = (fid_gvalr - fid_fixed) / fid_fixed if fid_fixed > 0 else float("inf")
    epoch_frac = adaptive_epochs / fixed_epochs if fixed_epochs > 0 else float("inf")
    confirmed = (
        adaptive_epochs <= int(fixed_epochs * max_epoch_fraction + 1e-9)
        and abs(rel_delta) <= max_rel_fid
    )
    elapsed = time.perf_counter() - t0

    logger.finish(
        elapsed,
        eval_set="cifar10_val",
        n_params=n_params,
        fixed_epochs=fixed_epochs,
        adaptive_epochs=adaptive_epochs,
        epoch_fraction=epoch_frac,
        fid_r18_fixed=fid_fixed,
        fid_r18_gvalr=fid_gvalr,
        relative_fid_delta=rel_delta,
        fixed_wall_s=fixed_wall,
        gvalr_wall_s=gvalr_wall,
        final_train_loss=float(hist[-1]) if hist else None,
        hypothesis_confirmed=confirmed,
        adaptive_lr=True,
        device=str(device),
    )

    result = GvAlrImageResult(
        n_params=n_params,
        fixed_epochs=fixed_epochs,
        adaptive_epochs=adaptive_epochs,
        epoch_fraction=epoch_frac,
        fid_fixed=fid_fixed,
        fid_gvalr=fid_gvalr,
        relative_fid_delta=rel_delta,
        fixed_wall_s=fixed_wall,
        gvalr_wall_s=gvalr_wall,
        elapsed_s=elapsed,
        device=str(device),
        profile=profile,
        hypothesis_confirmed=confirmed,
    )
    if verbose:
        print(
            f"FID fixed={fid_fixed:.2f} gvalr={fid_gvalr:.2f} "
            f"rel_delta={rel_delta:.3f} epoch_frac={epoch_frac:.3f} "
            f"confirmed={confirmed} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    return result


def write_results_md(result: GvAlrImageResult) -> Path:
    path = Path(__file__).resolve().parent / "results.md"
    verdict = "Confirmed" if result.hypothesis_confirmed else "Rejected"
    path.write_text(
        "\n".join(
            [
                f"# Results — EXP 105b: GV-ALR on NanoUNet DDPM ({date.today().isoformat()})",
                "",
                f"**Verdict:** {verdict}",
                f"**Profile:** `{result.profile}` · **Device:** `{result.device}`",
                f"**Params:** {result.n_params:,}",
                "",
                "## Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Fixed epochs | {result.fixed_epochs} |",
                f"| GV-ALR epochs | {result.adaptive_epochs} |",
                f"| Epoch fraction | {result.epoch_fraction:.3f} |",
                f"| FID-R18 fixed-LR | {result.fid_fixed:.2f} |",
                f"| FID-R18 GV-ALR | {result.fid_gvalr:.2f} |",
                f"| Relative FID delta | {result.relative_fid_delta:.3f} |",
                f"| Fixed wall (s) | {result.fixed_wall_s:.1f} |",
                f"| GV-ALR wall (s) | {result.gvalr_wall_s:.1f} |",
                f"| Elapsed (s) | {result.elapsed_s:.1f} |",
                "",
                "## Gate (H-I3)",
                "",
                "- FID within ±3% relative **and** epochs ≤ 70% of fixed.",
                f"- Outcome: **{verdict}** (Δ={result.relative_fid_delta:.3f}, "
                f"frac={result.epoch_fraction:.3f}).",
                "",
                "## Ablation suggestion",
                "",
                "- What if you adapt every N mini-batches instead of once per epoch?",
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
    result = run_exp_105b(profile=args.profile)
    if args.write_results:
        out = write_results_md(result)
        print(f"wrote {out}", flush=True)
    return 0 if gate_passed(result) or args.profile == "ci" else 1


if __name__ == "__main__":
    raise SystemExit(main())
