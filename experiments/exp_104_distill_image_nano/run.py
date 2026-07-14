"""
EXP 104 — Teacher→NanoUNet image distill (Phase I / H-I1).

Publication (RTX 4060):
  MLFLOW_DISABLE=1 QML_DEVICE=cuda \\
    python experiments/exp_104_distill_image_nano/run.py --profile publication --write-results
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
from src.training.image_ddpm import DDPMSchedule, sample_ddpm, train_ddpm, train_ddpm_distill
from src.training.image_fid import fid_r18
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id

EXP_KEY = "exp_104_distill_image_nano"
EXP_ID = "exp_104"


@dataclass(frozen=True)
class DistillImageResult:
    teacher_params: int
    student_params: int
    fid_teacher: float
    fid_hard: float
    fid_distill: float
    distill_vs_teacher_ratio: float
    relative_win_vs_hard: float
    elapsed_s: float
    device: str
    profile: str
    hypothesis_confirmed: bool


def gate_passed(
    result: DistillImageResult,
    *,
    max_teacher_ratio: float = 1.10,
    min_hard_win: float = 0.05,
) -> bool:
    return bool(
        result.hypothesis_confirmed
        and result.distill_vs_teacher_ratio <= max_teacher_ratio
        and result.relative_win_vs_hard >= min_hard_win
    )


def _fid_for(
    model: NanoUNet,
    schedule: DDPMSchedule,
    real_ref: torch.Tensor,
    *,
    n_fid: int,
    device: torch.device,
) -> float:
    samples = sample_ddpm(model, schedule, n=n_fid, shape=(3, 32, 32))
    return fid_r18(real_ref, samples, device=device)


def run_exp_104(*, profile: str = "ci", verbose: bool = True) -> DistillImageResult:
    os.environ.setdefault("MLFLOW_DISABLE", "1")
    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    epochs = int(cfg.get("epochs", 2))
    batch_size = int(cfg.get("batch_size", 8))
    lr = float(cfg.get("lr", 2e-4))
    timesteps = int(cfg.get("timesteps", 20))
    teacher_ch = int(cfg.get("teacher_base_channels", 64))
    student_ch = int(cfg.get("student_base_channels", 32))
    n_train = int(cfg.get("n_train", 32))
    n_val = int(cfg.get("n_val", 16))
    n_fid = int(cfg.get("n_fid_samples", 16))
    alpha = float(cfg.get("distill_alpha", 0.7))
    max_ratio = float(cfg.get("max_teacher_fid_ratio", 1.10))
    min_hard_win = float(cfg.get("min_relative_fid_win_vs_hard", 0.05))

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
    teacher = NanoUNet(in_channels=3, base_channels=teacher_ch)
    hard = NanoUNet(in_channels=3, base_channels=student_ch)
    distill = NanoUNet(in_channels=3, base_channels=student_ch)
    teacher_params = teacher.count_parameters()
    student_params = distill.count_parameters()

    logger = ExperimentLogger(EXP_ID, "nano_unet_distill", seed=seed, profile=profile)
    t0 = time.perf_counter()
    if verbose:
        print(
            f"device={device} teacher_params={teacher_params:,} student_params={student_params:,} "
            f"n_train={n_train} epochs={epochs} T={timesteps} alpha={alpha}",
            flush=True,
        )

    if verbose:
        print("training teacher …", flush=True)
    train_ddpm(
        teacher,
        x_train,
        schedule,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        logger=None,
    )

    if verbose:
        print("training hard student …", flush=True)
    train_ddpm(
        hard,
        x_train,
        schedule,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed + 1,
        logger=None,
    )

    if verbose:
        print("training distill student …", flush=True)
    hist = train_ddpm_distill(
        distill,
        teacher,
        x_train,
        schedule,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        alpha=alpha,
        seed=seed + 2,
        logger=logger,
    )

    if verbose:
        print("computing FID-R18 …", flush=True)
    fid_teacher = _fid_for(teacher, schedule, real_ref, n_fid=n_fid, device=device)
    fid_hard = _fid_for(hard, schedule, real_ref, n_fid=n_fid, device=device)
    fid_distill = _fid_for(distill, schedule, real_ref, n_fid=n_fid, device=device)
    ratio = fid_distill / fid_teacher if fid_teacher > 0 else float("inf")
    win_vs_hard = relative_fid_improvement(fid_hard, fid_distill)
    confirmed = ratio <= max_ratio and win_vs_hard >= min_hard_win
    elapsed = time.perf_counter() - t0

    logger.finish(
        elapsed,
        eval_set="cifar10_val",
        teacher_params=teacher_params,
        student_params=student_params,
        fid_r18_teacher=fid_teacher,
        fid_r18_hard=fid_hard,
        fid_r18_distill=fid_distill,
        distill_vs_teacher_ratio=ratio,
        relative_win_vs_hard=win_vs_hard,
        distill_alpha=alpha,
        final_train_loss=float(hist[-1]) if hist else None,
        hypothesis_confirmed=confirmed,
        device=str(device),
    )

    result = DistillImageResult(
        teacher_params=teacher_params,
        student_params=student_params,
        fid_teacher=fid_teacher,
        fid_hard=fid_hard,
        fid_distill=fid_distill,
        distill_vs_teacher_ratio=ratio,
        relative_win_vs_hard=win_vs_hard,
        elapsed_s=elapsed,
        device=str(device),
        profile=profile,
        hypothesis_confirmed=confirmed,
    )
    if verbose:
        print(
            f"FID teacher={fid_teacher:.2f} hard={fid_hard:.2f} distill={fid_distill:.2f} "
            f"ratio={ratio:.3f} win_vs_hard={win_vs_hard:.3f} "
            f"confirmed={confirmed} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    return result


def write_results_md(result: DistillImageResult) -> Path:
    path = Path(__file__).resolve().parent / "results.md"
    verdict = "Confirmed" if result.hypothesis_confirmed else "Rejected"
    path.write_text(
        "\n".join(
            [
                f"# Results — EXP 104: Teacher→NanoUNet image distill ({date.today().isoformat()})",
                "",
                f"**Verdict:** {verdict}",
                f"**Profile:** `{result.profile}` · **Device:** `{result.device}`",
                f"**Teacher params:** {result.teacher_params:,} · **Student params:** {result.student_params:,}",
                "",
                "## Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| FID-R18 teacher | {result.fid_teacher:.2f} |",
                f"| FID-R18 hard student | {result.fid_hard:.2f} |",
                f"| FID-R18 distill student | {result.fid_distill:.2f} |",
                f"| Distill / teacher FID ratio | {result.distill_vs_teacher_ratio:.3f} |",
                f"| Relative FID win vs hard | {result.relative_win_vs_hard:.3f} |",
                f"| Elapsed (s) | {result.elapsed_s:.1f} |",
                "",
                "## Gate (H-I1)",
                "",
                "- Distill FID ≤ teacher × 1.10 **and** relative win vs hard ≥ 0.05.",
                f"- Outcome: **{verdict}** (ratio={result.distill_vs_teacher_ratio:.3f}, "
                f"win={result.relative_win_vs_hard:.3f}).",
                "",
                "## Ablation suggestion",
                "",
                "- What if you remove soft targets (alpha=0) vs pure teacher (alpha=1)?",
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
    result = run_exp_104(profile=args.profile)
    if args.write_results:
        out = write_results_md(result)
        print(f"wrote {out}", flush=True)
    return 0 if gate_passed(result) or args.profile == "ci" else 1


if __name__ == "__main__":
    raise SystemExit(main())
