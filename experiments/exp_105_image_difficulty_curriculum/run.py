"""
EXP 105 — Image difficulty curriculum on CIFAR (Phase I / H-I2).

Publication (RTX 4060):
  MLFLOW_DISABLE=1 QML_DEVICE=cuda \\
    python experiments/exp_105_image_difficulty_curriculum/run.py --profile publication --write-results
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
from src.training.distillation import relative_fid_improvement
from src.training.image_curriculum import (
    sort_by_random,
    sort_by_sharpness,
    train_staged_ddpm_curriculum,
)
from src.training.image_ddpm import DDPMSchedule, sample_ddpm
from src.training.image_fid import fid_r18
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id

EXP_KEY = "exp_105_image_difficulty_curriculum"
EXP_ID = "exp_105"


@dataclass(frozen=True)
class CurriculumImageResult:
    n_params: int
    fid_random: float
    fid_curriculum: float
    relative_win: float
    elapsed_s: float
    device: str
    profile: str
    hypothesis_confirmed: bool


def gate_passed(result: CurriculumImageResult, *, min_rel: float = 0.05) -> bool:
    return bool(result.hypothesis_confirmed and result.relative_win >= min_rel)


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


def run_exp_105(*, profile: str = "ci", verbose: bool = True) -> CurriculumImageResult:
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
    n_stages = int(cfg.get("n_stages", 4))
    epochs_per_stage = int(cfg.get("epochs_per_stage", 1))
    refine_epochs = int(cfg.get("refine_epochs", 1))
    min_rel = float(cfg.get("min_relative_fid_win", 0.05))

    init_correlation_id()
    device = resolve_device()
    torch.manual_seed(seed)

    x_train_np, _ = load_cifar10_nchw(split="train", n_take=n_train, seed=seed)
    x_val_np, _ = load_cifar10_nchw(split="val", n_take=max(n_val, n_fid), seed=seed + 1)
    real_ref = torch.from_numpy(x_val_np[:n_fid]).to(device)
    if real_ref.shape[0] < n_fid:
        reps = (n_fid + real_ref.shape[0] - 1) // real_ref.shape[0]
        real_ref = real_ref.repeat(reps, 1, 1, 1)[:n_fid]

    schedule = DDPMSchedule(timesteps=timesteps, device=device)
    images_curr = sort_by_sharpness(x_train_np)
    images_rand = sort_by_random(x_train_np, seed=seed)

    model_rand = NanoUNet(in_channels=3, base_channels=base_channels)
    model_curr = NanoUNet(in_channels=3, base_channels=base_channels)
    n_params = model_curr.count_parameters()

    logger = ExperimentLogger(EXP_ID, "nano_unet_curriculum", seed=seed, profile=profile)
    t0 = time.perf_counter()
    if verbose:
        print(
            f"device={device} params={n_params:,} n_train={n_train} "
            f"stages={n_stages}×{epochs_per_stage}+refine={refine_epochs} T={timesteps}",
            flush=True,
        )
        print("training random-order staged …", flush=True)

    train_staged_ddpm_curriculum(
        model_rand,
        images_rand,
        schedule,
        n_stages=n_stages,
        epochs_per_stage=epochs_per_stage,
        refine_epochs=refine_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        logger=None,
    )

    if verbose:
        print("training sharpness curriculum …", flush=True)
    hist = train_staged_ddpm_curriculum(
        model_curr,
        images_curr,
        schedule,
        n_stages=n_stages,
        epochs_per_stage=epochs_per_stage,
        refine_epochs=refine_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed + 1,
        logger=logger,
    )

    if verbose:
        print("computing FID-R18 …", flush=True)
    fid_random = _fid(model_rand, schedule, real_ref, n_fid=n_fid, device=device)
    fid_curriculum = _fid(model_curr, schedule, real_ref, n_fid=n_fid, device=device)
    win = relative_fid_improvement(fid_random, fid_curriculum)
    confirmed = win >= min_rel
    elapsed = time.perf_counter() - t0

    logger.finish(
        elapsed,
        eval_set="cifar10_val",
        n_params=n_params,
        fid_r18_random=fid_random,
        fid_r18_curriculum=fid_curriculum,
        relative_win_vs_random=win,
        final_train_loss=float(hist[-1]) if hist else None,
        hypothesis_confirmed=confirmed,
        device=str(device),
    )

    result = CurriculumImageResult(
        n_params=n_params,
        fid_random=fid_random,
        fid_curriculum=fid_curriculum,
        relative_win=win,
        elapsed_s=elapsed,
        device=str(device),
        profile=profile,
        hypothesis_confirmed=confirmed,
    )
    if verbose:
        print(
            f"FID random={fid_random:.2f} curriculum={fid_curriculum:.2f} "
            f"win={win:.3f} confirmed={confirmed} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    return result


def write_results_md(result: CurriculumImageResult) -> Path:
    path = Path(__file__).resolve().parent / "results.md"
    verdict = "Confirmed" if result.hypothesis_confirmed else "Rejected"
    path.write_text(
        "\n".join(
            [
                f"# Results — EXP 105: Image difficulty curriculum ({date.today().isoformat()})",
                "",
                f"**Verdict:** {verdict}",
                f"**Profile:** `{result.profile}` · **Device:** `{result.device}`",
                f"**Params:** {result.n_params:,}",
                "",
                "## Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| FID-R18 random staged | {result.fid_random:.2f} |",
                f"| FID-R18 sharpness curriculum | {result.fid_curriculum:.2f} |",
                f"| Relative FID win vs random | {result.relative_win:.3f} |",
                f"| Elapsed (s) | {result.elapsed_s:.1f} |",
                "",
                "## Gate (H-I2)",
                "",
                "- Relative FID win ≥ 0.05 vs random staged (matched epoch budget).",
                f"- Outcome: **{verdict}** (win={result.relative_win:.3f}).",
                "",
                "## Ablation suggestion",
                "",
                "- What if you order by FFT high-frequency energy instead of Laplacian variance?",
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
    result = run_exp_105(profile=args.profile)
    if args.write_results:
        out = write_results_md(result)
        print(f"wrote {out}", flush=True)
    return 0 if gate_passed(result) or args.profile == "ci" else 1


if __name__ == "__main__":
    raise SystemExit(main())
