"""
EXP 111 — Measurement-scheduled CFG on Flickr8k (Phase J / H-Q3.6).

Publication (RTX 4060):
  MLFLOW_DISABLE=1 QML_DEVICE=cuda \\
    python experiments/exp_111_measurement_scheduled_cfg/run.py --profile publication --write-results
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
from src.data.open_captions import is_flickr8k_ready, load_flickr8k_batch
from src.quantum.text_quantum_fusion import ClassicalTextTokenFusion
from src.training.config import load_experiment_config
from src.training.device import resolve_device
from src.training.image_clip import clip_score, encode_clip_text
from src.training.image_ddpm import DDPMSchedule
from src.training.measurement_cfg import (
    sample_ddpm_classical_cfg,
    sample_ddpm_measurement_schedule,
    train_ddpm_cfg,
)
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id

EXP_KEY = "exp_111_measurement_scheduled_cfg"
EXP_ID = "exp_111"


@dataclass(frozen=True)
class MeasurementCfgResult:
    params: int
    clip_cfg: float
    clip_meas: float
    clip_delta: float
    final_loss: float
    elapsed_s: float
    device: str
    profile: str
    hypothesis_confirmed: bool


def gate_passed(result: MeasurementCfgResult, *, min_gap: float = 0.5) -> bool:
    return bool(result.hypothesis_confirmed and result.clip_delta >= min_gap)


def _make_dit(cfg: dict, text_dim: int) -> TinyDiT:
    return TinyDiT(
        img_size=int(cfg.get("img_size", 32)),
        patch_size=int(cfg.get("patch_size", 4)),
        dim=int(cfg.get("dim", 64)),
        depth=int(cfg.get("depth", 4)),
        n_heads=int(cfg.get("n_heads", 4)),
        time_dim=int(cfg.get("time_dim", 128)),
        coupling="classical",
        text_dim=text_dim,
        use_cross_attn=True,
    )


def run_exp_111(*, profile: str = "ci", verbose: bool = True) -> MeasurementCfgResult:
    os.environ.setdefault("MLFLOW_DISABLE", "1")
    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    epochs = int(cfg.get("epochs", 2))
    batch_size = int(cfg.get("batch_size", 8))
    lr = float(cfg.get("lr", 2e-4))
    timesteps = int(cfg.get("timesteps", 20))
    n_train = int(cfg.get("n_train", 32))
    n_eval = int(cfg.get("n_eval", 16))
    img_size = int(cfg.get("img_size", 32))
    text_dim = int(cfg.get("text_dim", 64))
    clip_dim = int(cfg.get("clip_dim", 512))
    p_uncond = float(cfg.get("p_uncond", 0.1))
    guidance_scale = float(cfg.get("guidance_scale", 2.0))
    keep_floor = float(cfg.get("keep_floor", 0.25))
    min_gap = float(cfg.get("min_clip_gap", 0.5))

    if not is_flickr8k_ready():
        raise FileNotFoundError(
            "Flickr8k caption pack not ready — run make data-open-images-captions "
            "&& make data-open-caption-splits"
        )

    init_correlation_id()
    device = resolve_device()
    torch.manual_seed(seed)

    train_batch = load_flickr8k_batch("train", n_take=n_train, img_size=img_size, seed=seed)
    eval_batch = load_flickr8k_batch("val", n_take=n_eval, img_size=img_size, seed=seed + 1)
    x_train = torch.from_numpy(train_batch["images"])
    captions_train = list(train_batch["captions"])
    captions_eval = list(eval_batch["captions"])

    if verbose:
        print("encoding CLIP text features …", flush=True)
    clip_train = encode_clip_text(captions_train, device=device)
    clip_eval = encode_clip_text(captions_eval, device=device)
    if clip_train.shape[-1] != clip_dim:
        clip_dim = int(clip_train.shape[-1])

    schedule = DDPMSchedule(timesteps=timesteps, device=device)
    logger = ExperimentLogger(EXP_ID, "measurement_scheduled_cfg", seed=seed, profile=profile)
    t0 = time.perf_counter()

    fusion = ClassicalTextTokenFusion(clip_dim=clip_dim, out_dim=text_dim)
    model = _make_dit(cfg, text_dim)
    if verbose:
        print(
            f"device={device} n_train={n_train} epochs={epochs} T={timesteps} "
            f"cfg_w={guidance_scale} keep_floor={keep_floor}",
            flush=True,
        )
        print("training CFG TinyDiT (shared checkpoint) …", flush=True)
    hist = train_ddpm_cfg(
        model,
        fusion,
        x_train,
        clip_train,
        schedule,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        p_uncond=p_uncond,
        seed=seed,
        logger=logger,
    )

    if verbose:
        print("sampling classical CFG + measurement schedule …", flush=True)
    shape = (3, img_size, img_size)
    samples_cfg = sample_ddpm_classical_cfg(
        model,
        fusion,
        clip_eval,
        schedule,
        guidance_scale=guidance_scale,
        shape=shape,
    )
    samples_meas = sample_ddpm_measurement_schedule(
        model,
        fusion,
        clip_eval,
        schedule,
        guidance_scale=guidance_scale,
        keep_floor=keep_floor,
        shape=shape,
    )
    clip_c = float(clip_score(samples_cfg, captions_eval, device=device))
    clip_m = float(clip_score(samples_meas, captions_eval, device=device))
    delta = clip_m - clip_c
    confirmed = delta >= min_gap
    elapsed = time.perf_counter() - t0
    params = model.count_parameters() + fusion.count_parameters()
    logger.finish(
        elapsed,
        eval_set="flickr8k_val",
        clip_cfg=clip_c,
        clip_meas=clip_m,
        clip_delta=delta,
        hypothesis_confirmed=confirmed,
        params=params,
        device=str(device),
    )
    result = MeasurementCfgResult(
        params=params,
        clip_cfg=clip_c,
        clip_meas=clip_m,
        clip_delta=delta,
        final_loss=float(hist[-1]) if hist else float("nan"),
        elapsed_s=elapsed,
        device=str(device),
        profile=profile,
        hypothesis_confirmed=confirmed,
    )
    if verbose:
        print(
            f"CLIP cfg={clip_c:.2f} meas={clip_m:.2f} Δ={delta:.2f} "
            f"confirmed={confirmed} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    return result


def write_results(result: MeasurementCfgResult, path: Path) -> None:
    verdict = "Confirmed" if result.hypothesis_confirmed else "Rejected"
    path.write_text(
        "\n".join(
            [
                f"# Results — EXP 111: Measurement-scheduled CFG ({date.today().isoformat()})",
                "",
                f"**Verdict:** {verdict}",
                f"**Profile:** `{result.profile}` · **Device:** `{result.device}`",
                f"**Params:** {result.params:,}",
                "",
                "## Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| CLIPScore classical CFG | {result.clip_cfg:.2f} |",
                f"| CLIPScore measurement schedule | {result.clip_meas:.2f} |",
                f"| Δ CLIP (meas − cfg) | {result.clip_delta:.2f} |",
                f"| Final train loss | {result.final_loss:.4f} |",
                f"| Elapsed (s) | {result.elapsed_s:.1f} |",
                "",
                "## Gate (H-Q3.6)",
                "",
                "- Win: `CLIP_meas ≥ CLIP_cfg + 0.5`.",
                f"- Outcome: **{verdict}** (Δ={result.clip_delta:.2f}).",
                "",
                "## Ablation suggestion",
                "",
                "- What if keep_floor tracks continuous Softmax masks instead of Bernoulli?",
                "",
                f"*Logged via ExperimentLogger · {datetime.now().isoformat()}*",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("ci", "publication"), default="ci")
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args(argv)
    result = run_exp_111(profile=args.profile)
    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        write_results(result, out)
        print(f"wrote {out}", flush=True)
    return 0 if gate_passed(result) or args.profile == "ci" else 1


if __name__ == "__main__":
    raise SystemExit(main())
