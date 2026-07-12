"""
EXP 066 — Depolarizing noise vs noiseless hybrid on ACYD temporal test split (H-Q10 / H-Q5).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_066_noise_reg_acyd/run.py --profile publication --write-results
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.open_parquet import load_open_parquet_splits
from src.quantum.hybrid_model import HybridSandwich
from src.quantum.noise_regularized_qnn import NoiseRegularizedHybridSandwich
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_066_noise_reg_acyd"
EXP_ID = "exp_066"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NoiseRegAcydResult:
    n_train_rows: int
    n_val_rows: int
    n_test_rows: int
    n_params_noiseless: int
    n_params_noisy: int
    noiseless_test_auc: float
    noisy_test_auc: float
    auc_advantage_pp: float
    min_auc_advantage_pp: float
    depolarizing_p: float
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _train_hybrid(
    model: torch.nn.Module,
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: dict,
    seed: int,
    profile: str,
    model_name: str,
) -> None:
    train_model_batched(
        model,
        x_train,
        y_train,
        EXP_ID,
        model_name,
        epochs=int(cfg["epochs"]),
        lr=float(cfg.get("learning_rate", 0.01)),
        batch_size=int(cfg.get("batch_size", 512)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        X_val=x_val,
        y_val=y_val,
        seed=seed,
        profile=profile,
        save_checkpoints=bool(cfg.get("save_checkpoints", False)),
        device="cuda",
    )


def gate_passed(result: NoiseRegAcydResult) -> bool:
    return result.auc_advantage_pp >= result.min_auc_advantage_pp


def run_exp_066(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> NoiseRegAcydResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_soy_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 50_107)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 5_830)
    depolarizing_p = float(cfg.get("depolarizing_p", 0.03))
    min_auc_pp = float(cfg.get("min_auc_advantage_pp", 0.5))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 066 — Noise-reg hybrid on ACYD | profile={profile} | "
            f"p={depolarizing_p} | train={n_train or 'all'}"
        )
        print(f"Gate: noisy test ROC-AUC ≥ noiseless + {min_auc_pp} pp")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    x_train, y_train, x_val, y_val, x_test, y_test, _scaler = load_open_parquet_splits(
        dataset_id,
        ROOT,
        n_train_rows=n_train,
        n_val_rows=n_val,
        random_state=seed,
    )
    input_dim = int(x_train.shape[1])

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    noiseless = HybridSandwich(
        input_dim=input_dim,
        n_qubits=int(cfg.get("n_qubits", 4)),
        n_layers=int(cfg.get("n_layers", 2)),
        reupload=bool(cfg.get("reupload", True)),
    )
    n_noiseless = count_parameters(noiseless)
    if verbose:
        print("Training noiseless hybrid...", flush=True)
    _train_hybrid(
        noiseless,
        x_train=x_train_t,
        y_train=y_train_t,
        x_val=x_val_t,
        y_val=y_val_t,
        cfg=cfg,
        seed=seed,
        profile=profile,
        model_name="hybrid_noiseless",
    )
    noiseless_auc = float(evaluate_with_auc(noiseless, x_test_t, y_test_t)["roc_auc"])

    noisy = NoiseRegularizedHybridSandwich(
        input_dim=input_dim,
        n_qubits=int(cfg.get("n_qubits", 4)),
        n_layers=int(cfg.get("n_layers", 2)),
        reupload=bool(cfg.get("reupload", True)),
        depolarizing_p=depolarizing_p,
    )
    n_noisy = count_parameters(noisy)
    if verbose:
        print("Training noisy hybrid...", flush=True)
    _train_hybrid(
        noisy,
        x_train=x_train_t,
        y_train=y_train_t,
        x_val=x_val_t,
        y_val=y_val_t,
        cfg=cfg,
        seed=seed + 1,
        profile=profile,
        model_name="hybrid_noisy",
    )
    noisy_auc = float(evaluate_with_auc(noisy, x_test_t, y_test_t)["roc_auc"])
    advantage_pp = (noisy_auc - noiseless_auc) * 100.0
    elapsed = time.perf_counter() - t0

    log_event(
        "info",
        "exp_066 noise reg summary",
        exp_id=EXP_ID,
        profile=profile,
        depolarizing_p=depolarizing_p,
        noiseless_test_auc=noiseless_auc,
        noisy_test_auc=noisy_auc,
        auc_advantage_pp=round(advantage_pp, 3),
    )

    if verbose:
        status = "OK" if advantage_pp >= min_auc_pp else "FAIL"
        print(
            f"noiseless test ROC-AUC={noiseless_auc:.4f} | noisy={noisy_auc:.4f} | "
            f"Δ={advantage_pp:.2f} pp [{status}] | {elapsed:.1f}s",
            flush=True,
        )

    return NoiseRegAcydResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_test_rows=len(y_test),
        n_params_noiseless=n_noiseless,
        n_params_noisy=n_noisy,
        noiseless_test_auc=noiseless_auc,
        noisy_test_auc=noisy_auc,
        auc_advantage_pp=round(advantage_pp, 3),
        min_auc_advantage_pp=min_auc_pp,
        depolarizing_p=depolarizing_p,
        elapsed_s=round(elapsed, 3),
    )


def _summarize(result: NoiseRegAcydResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest_negative"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 066 SUMMARY",
            f"{'=' * 60}",
            f"Depolarizing p: {result.depolarizing_p}",
            f"Train: {result.n_train_rows:,} | Val: {result.n_val_rows:,} | "
            f"Test (temporal ≥2022): {result.n_test_rows:,}",
            f"Noiseless test ROC-AUC: {result.noiseless_test_auc:.4f} "
            f"({result.n_params_noiseless:,} params)",
            f"Noisy test ROC-AUC: {result.noisy_test_auc:.4f} ({result.n_params_noisy:,} params)",
            f"Advantage: {result.auc_advantage_pp:+.2f} pp (gate ≥ {result.min_auc_advantage_pp} pp)",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: NoiseRegAcydResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest negative"
    return "\n".join(
        [
            "# Results — EXP 066: Depolarizing noise on ACYD hybrid QNN (H-Q10 / H-Q5)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "**Eval split:** temporal test (years ≥ 2022)",
            "",
            "## Validation gate (ROC-AUC)",
            "",
            f"- Depolarizing p: **{result.depolarizing_p}**",
            f"- Train rows: **{result.n_train_rows:,}** · Val: **{result.n_val_rows:,}** · "
            f"Test: **{result.n_test_rows:,}**",
            f"- Noiseless test ROC-AUC: **{result.noiseless_test_auc:.4f}**",
            f"- Noisy test ROC-AUC: **{result.noisy_test_auc:.4f}**",
            f"- Advantage: **{result.auc_advantage_pp:+.2f} pp**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — noisy vs noiseless hybrid on temporal test "
            f"(gate ≥ {result.min_auc_advantage_pp} pp).",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 066 — depolarizing noise on ACYD hybrid")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_066(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
