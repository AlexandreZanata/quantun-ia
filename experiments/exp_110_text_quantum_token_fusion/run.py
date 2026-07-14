"""
EXP 110 — Text–quantum token fusion on Flickr8k (Phase J / H-Q3.5).

Publication (RTX 4060 + PennyLane CPU QNN):
  MLFLOW_DISABLE=1 QML_DEVICE=cuda \\
    python experiments/exp_110_text_quantum_token_fusion/run.py --profile publication --write-results
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
from src.quantum.text_quantum_fusion import ClassicalTextTokenFusion, QuantumTextTokenFusion
from src.training.config import load_experiment_config
from src.training.device import resolve_device
from src.training.image_clip import clip_score, encode_clip_text
from src.training.image_ddpm import (
    DDPMSchedule,
    sample_ddpm_clip_features,
    train_ddpm_clip_features,
)
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id

EXP_KEY = "exp_110_text_quantum_token_fusion"
EXP_ID = "exp_110"


@dataclass(frozen=True)
class TextQuantumFusionResult:
    classical_params: int
    quantum_params: int
    clip_classical: float
    clip_quantum: float
    clip_delta: float
    final_loss_classical: float
    final_loss_quantum: float
    elapsed_s: float
    device: str
    profile: str
    hypothesis_confirmed: bool


def gate_passed(result: TextQuantumFusionResult, *, min_gap: float = 0.5) -> bool:
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


def run_exp_110(*, profile: str = "ci", verbose: bool = True) -> TextQuantumFusionResult:
    os.environ.setdefault("MLFLOW_DISABLE", "1")
    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    epochs_c = int(cfg.get("classical_epochs", cfg.get("epochs", 2)))
    epochs_q = int(cfg.get("quantum_epochs", 1))
    batch_size = int(cfg.get("batch_size", 8))
    quantum_batch = int(cfg.get("quantum_batch_size", batch_size))
    lr = float(cfg.get("lr", 2e-4))
    timesteps = int(cfg.get("timesteps", 20))
    n_train = int(cfg.get("n_train", 32))
    n_eval = int(cfg.get("n_eval", 16))
    img_size = int(cfg.get("img_size", 32))
    text_dim = int(cfg.get("text_dim", 64))
    clip_dim = int(cfg.get("clip_dim", 512))
    n_qubits = int(cfg.get("n_qubits", 4))
    n_layers = int(cfg.get("n_layers", 2))
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
    logger = ExperimentLogger(EXP_ID, "text_quantum_token_fusion", seed=seed, profile=profile)
    t0 = time.perf_counter()

    fusion_c = ClassicalTextTokenFusion(clip_dim=clip_dim, out_dim=text_dim)
    model_c = _make_dit(cfg, text_dim)
    if verbose:
        print(
            f"device={device} n_train={n_train} classical_ep={epochs_c} "
            f"quantum_ep={epochs_q} T={timesteps}",
            flush=True,
        )
        print("training classical (null-quantum) fusion TinyDiT …", flush=True)
    hist_c = train_ddpm_clip_features(
        model_c,
        fusion_c,
        x_train,
        clip_train,
        schedule,
        epochs=epochs_c,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        logger=None,
        fusion_on_cpu=False,
    )

    fusion_q = QuantumTextTokenFusion(
        clip_dim=clip_dim,
        out_dim=text_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        reupload=True,
    )
    model_q = _make_dit(cfg, text_dim)
    if verbose:
        print("training quantum text fusion TinyDiT (CPU QNN) …", flush=True)
    hist_q = train_ddpm_clip_features(
        model_q,
        fusion_q,
        x_train,
        clip_train,
        schedule,
        epochs=epochs_q,
        batch_size=quantum_batch,
        lr=lr,
        seed=seed + 1,
        logger=logger,
        fusion_on_cpu=True,
    )

    if verbose:
        print("sampling + CLIPScore …", flush=True)
    samples_c = sample_ddpm_clip_features(
        model_c, fusion_c, clip_eval, schedule, shape=(3, img_size, img_size), fusion_on_cpu=False
    )
    samples_q = sample_ddpm_clip_features(
        model_q, fusion_q, clip_eval, schedule, shape=(3, img_size, img_size), fusion_on_cpu=True
    )
    clip_c = float(clip_score(samples_c, captions_eval, device=device))
    clip_q = float(clip_score(samples_q, captions_eval, device=device))
    delta = clip_q - clip_c
    confirmed = delta >= min_gap
    elapsed = time.perf_counter() - t0

    c_params = model_c.count_parameters() + fusion_c.count_parameters()
    q_params = model_q.count_parameters() + fusion_q.count_parameters()
    logger.finish(
        elapsed,
        eval_set="flickr8k_val",
        clip_classical=clip_c,
        clip_quantum=clip_q,
        clip_delta=delta,
        hypothesis_confirmed=confirmed,
        classical_params=c_params,
        quantum_params=q_params,
        device=str(device),
    )
    result = TextQuantumFusionResult(
        classical_params=c_params,
        quantum_params=q_params,
        clip_classical=clip_c,
        clip_quantum=clip_q,
        clip_delta=delta,
        final_loss_classical=float(hist_c[-1]) if hist_c else float("nan"),
        final_loss_quantum=float(hist_q[-1]) if hist_q else float("nan"),
        elapsed_s=elapsed,
        device=str(device),
        profile=profile,
        hypothesis_confirmed=confirmed,
    )
    if verbose:
        print(
            f"CLIP c={clip_c:.2f} q={clip_q:.2f} Δ={delta:.2f} "
            f"confirmed={confirmed} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    return result


def write_results(result: TextQuantumFusionResult, path: Path) -> None:
    verdict = "Confirmed" if result.hypothesis_confirmed else "Rejected"
    path.write_text(
        "\n".join(
            [
                f"# Results — EXP 110: Text–quantum token fusion ({date.today().isoformat()})",
                "",
                f"**Verdict:** {verdict}",
                f"**Profile:** `{result.profile}` · **Device:** `{result.device}`",
                f"**Classical params:** {result.classical_params:,} · **Quantum params:** {result.quantum_params:,}",
                "",
                "## Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| CLIPScore classical (null-quantum) | {result.clip_classical:.2f} |",
                f"| CLIPScore quantum fusion | {result.clip_quantum:.2f} |",
                f"| Δ CLIP (q − classical) | {result.clip_delta:.2f} |",
                f"| Final loss classical | {result.final_loss_classical:.4f} |",
                f"| Final loss quantum | {result.final_loss_quantum:.4f} |",
                f"| Elapsed (s) | {result.elapsed_s:.1f} |",
                "",
                "## Gate (H-Q3.5)",
                "",
                "- Win: `CLIP_q ≥ CLIP_classical + 0.5`.",
                f"- Outcome: **{verdict}** (Δ={result.clip_delta:.2f}).",
                "",
                "## Ablation suggestion",
                "",
                "- What if you fuse CLIP token sequences (not pooled) via multi-token cross-attn?",
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
    result = run_exp_110(profile=args.profile)
    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        write_results(result, out)
        print(f"wrote {out}", flush=True)
    return 0 if gate_passed(result) or args.profile == "ci" else 1


if __name__ == "__main__":
    raise SystemExit(main())
