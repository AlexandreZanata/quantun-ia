"""
EXP 103 — TinyDiT T2I floor on Flickr8k (Phase H / H-T4; closes G-T3).

Publication (RTX 4060):
  MLFLOW_DISABLE=1 QML_DEVICE=cuda \\
    python experiments/exp_103_tiny_dit_flickr_t2i/run.py --profile publication --write-results
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

from src.classical.tiny_dit import HashCaptionEmbedder, TinyDiT
from src.data.open_captions import is_flickr8k_ready, load_flickr8k_batch
from src.training.config import load_experiment_config
from src.training.device import resolve_device
from src.training.image_clip import clip_score
from src.training.image_ddpm import DDPMSchedule, sample_ddpm_captioned, train_ddpm_captioned
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id

EXP_KEY = "exp_103_tiny_dit_flickr_t2i"
EXP_ID = "exp_103"


@dataclass(frozen=True)
class TinyDitT2IResult:
    n_params: int
    clip_model: float
    clip_noise: float
    clip_null: float
    delta_vs_noise: float
    delta_vs_null: float
    final_loss: float
    elapsed_s: float
    device: str
    profile: str
    hypothesis_confirmed: bool


def gate_passed(result: TinyDitT2IResult, *, min_null_gap: float = 0.5) -> bool:
    return bool(
        result.hypothesis_confirmed
        and result.clip_model >= result.clip_noise
        and result.delta_vs_null >= min_null_gap
    )


def run_exp_103(*, profile: str = "ci", verbose: bool = True) -> TinyDitT2IResult:
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
    dim = int(cfg.get("dim", 64))
    depth = int(cfg.get("depth", 4))
    text_dim = int(cfg.get("text_dim", 64))
    min_null_gap = float(cfg.get("min_clip_null_gap", 0.5))

    if not is_flickr8k_ready():
        raise FileNotFoundError(
            "Flickr8k caption pack not ready — run "
            "`make data-open-images-captions` then `make data-open-caption-splits`"
        )

    init_correlation_id()
    device = resolve_device()
    torch.manual_seed(seed)

    train_batch = load_flickr8k_batch(
        "train", n_take=n_train, img_size=img_size, seed=seed
    )
    eval_batch = load_flickr8k_batch(
        "val", n_take=n_eval, img_size=img_size, seed=seed + 1
    )
    x_train = torch.from_numpy(train_batch["images"])
    captions_train = list(train_batch["captions"])
    captions_eval = list(eval_batch["captions"])

    embedder = HashCaptionEmbedder(dim=text_dim)
    model = TinyDiT(
        img_size=img_size,
        patch_size=int(cfg.get("patch_size", 4)),
        dim=dim,
        depth=depth,
        n_heads=int(cfg.get("n_heads", 4)),
        time_dim=int(cfg.get("time_dim", 128)),
        coupling="classical",
        text_dim=text_dim,
    )
    n_params = model.count_parameters() + embedder.count_parameters()
    schedule = DDPMSchedule(timesteps=timesteps, device=device)
    logger = ExperimentLogger(EXP_ID, "tiny_dit_t2i", seed=seed, profile=profile)
    t0 = time.perf_counter()

    if verbose:
        print(
            f"device={device} n_train={n_train} epochs={epochs} T={timesteps} "
            f"dim={dim} depth={depth} img={img_size}",
            flush=True,
        )

    history = train_ddpm_captioned(
        model,
        embedder,
        x_train,
        captions_train,
        schedule,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        logger=logger,
    )

    if verbose:
        print("sampling + CLIPScore …", flush=True)
    samples = sample_ddpm_captioned(
        model,
        embedder,
        captions_eval,
        schedule,
        shape=(3, img_size, img_size),
    )
    noise = torch.randn_like(samples)
    null_caps = [""] * len(captions_eval)
    clip_m = float(clip_score(samples, captions_eval, device=device))
    clip_n = float(clip_score(noise, captions_eval, device=device))
    clip_null = float(clip_score(samples, null_caps, device=device))
    d_noise = clip_m - clip_n
    d_null = clip_m - clip_null
    confirmed = clip_m >= clip_n and d_null >= min_null_gap
    elapsed = time.perf_counter() - t0

    logger.finish(
        elapsed,
        eval_set="flickr8k_val",
        clip_model=clip_m,
        clip_noise=clip_n,
        clip_null=clip_null,
        delta_vs_noise=d_noise,
        delta_vs_null=d_null,
        hypothesis_confirmed=confirmed,
        n_params=n_params,
        device=str(device),
    )
    result = TinyDitT2IResult(
        n_params=n_params,
        clip_model=clip_m,
        clip_noise=clip_n,
        clip_null=clip_null,
        delta_vs_noise=d_noise,
        delta_vs_null=d_null,
        final_loss=float(history[-1]) if history else float("nan"),
        elapsed_s=elapsed,
        device=str(device),
        profile=profile,
        hypothesis_confirmed=confirmed,
    )
    if verbose:
        print(
            f"CLIP model={clip_m:.2f} noise={clip_n:.2f} null={clip_null:.2f} "
            f"Δnull={d_null:.2f} confirmed={confirmed} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    return result


def write_results(result: TinyDitT2IResult, path: Path) -> None:
    verdict = "Confirmed" if result.hypothesis_confirmed else "Rejected"
    path.write_text(
        "\n".join(
            [
                f"# Results — EXP 103: TinyDiT Flickr8k T2I ({date.today().isoformat()})",
                "",
                f"**Verdict:** {verdict}",
                f"**Profile:** `{result.profile}` · **Device:** `{result.device}`",
                f"**Params (model+embedder):** {result.n_params:,}",
                "",
                "## Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| CLIPScore model | {result.clip_model:.2f} |",
                f"| CLIPScore noise | {result.clip_noise:.2f} |",
                f"| CLIPScore null caption | {result.clip_null:.2f} |",
                f"| Δ vs noise | {result.delta_vs_noise:.2f} |",
                f"| Δ vs null | {result.delta_vs_null:.2f} |",
                f"| Final loss | {result.final_loss:.4f} |",
                f"| Elapsed (s) | {result.elapsed_s:.1f} |",
                "",
                "## Gate (H-T4 / H1)",
                "",
                "- CLIP_model ≥ CLIP_noise and CLIP_model ≥ CLIP_null + 0.5.",
                f"- Outcome: **{verdict}** (Δnull={result.delta_vs_null:.2f}).",
                "",
                "## Ablation suggestion",
                "",
                "- What if you replace hash embedder with frozen OpenCLIP text features?",
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
    result = run_exp_103(profile=args.profile)
    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        write_results(result, out)
        print(f"wrote {out}", flush=True)
    return 0 if gate_passed(result) or args.profile == "ci" else 1


if __name__ == "__main__":
    raise SystemExit(main())
