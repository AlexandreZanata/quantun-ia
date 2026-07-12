"""
EXP 088 — Pauli/shadow features → NarrowDeepNano vs classical head (ACYD maize, H-Q2.3).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_088_shadow_features_nano_maize/run.py \\
    --profile publication --write-results
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classical.narrow_deep_nano import NarrowDeepNano
from src.data.open_parquet import load_open_parquet_splits
from src.quantum.shadow_features import PauliShadowFeatureEncoder
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_088_shadow_features_nano_maize"
EXP_ID = "exp_088"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ShadowFeaturesNanoResult:
    n_train_rows: int
    n_val_rows: int
    n_shadow_features: int
    logistic_val_auc: float
    classical_narrow_val_auc: float
    shadow_narrow_val_auc: float
    histgb_val_auc: float
    shadow_vs_classical_pp: float
    shadow_vs_logistic_pp: float
    min_vs_classical_pp: float
    min_vs_logistic_pp: float
    n_params_classical: int
    n_params_shadow: int
    feature_extract_s: float
    elapsed_s: float
    profile: str


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _training_uses_cuda() -> bool:
    return os.environ.get("QML_DEVICE", "auto").lower() == "cuda" and torch.cuda.is_available()


def _gate_passed(result: ShadowFeaturesNanoResult) -> bool:
    vs_classical = result.shadow_vs_classical_pp >= result.min_vs_classical_pp
    vs_logistic = result.shadow_vs_logistic_pp >= result.min_vs_logistic_pp
    return vs_classical and vs_logistic


def _train_narrow(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    profile: str,
    seed: int,
    model_name: str,
) -> tuple[float, int]:
    model = NarrowDeepNano(
        int(x_train.shape[1]),
        width=int(cfg.get("narrow_width", 512)),
        depth=int(cfg.get("narrow_depth", 3)),
        bottleneck=int(cfg.get("narrow_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    n_params = count_parameters(model)
    x_tr = torch.tensor(x_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    x_va = torch.tensor(x_val, dtype=torch.float32)
    y_va = torch.tensor(y_val, dtype=torch.float32)
    train_model_batched(
        model,
        x_tr,
        y_tr,
        EXP_ID,
        model_name,
        epochs=int(cfg["epochs"]),
        lr=float(cfg.get("learning_rate", 0.001)),
        batch_size=int(cfg.get("batch_size", 2048)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        X_val=x_va,
        y_val=y_va,
        seed=seed,
        profile=profile,
        save_checkpoints=bool(cfg.get("save_checkpoints", False)),
    )
    device = torch.device("cuda" if _training_uses_cuda() else "cpu")
    auc = float(evaluate_with_auc(model.to(device), x_va.to(device), y_va.to(device))["roc_auc"])
    return auc, n_params


def run_exp_088(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> ShadowFeaturesNanoResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_maize_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 200_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 50_000)
    n_qubits = int(cfg.get("n_qubits", 4))
    n_layers = int(cfg.get("n_layers", 1))
    n_features = int(cfg.get("n_shadow_features", 64))
    min_vs_classical = float(cfg.get("min_vs_classical_pp", -0.5))
    min_vs_logistic = float(cfg.get("min_vs_logistic_pp", 2.0))
    hgb_max_iter = int(cfg.get("histgb_max_iter", 100))
    progress_every = int(cfg.get("feature_progress_every", 0))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 088 — Shadow/Pauli features → NarrowDeep | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(
            f"Gate: shadow ≥ classical − {abs(min_vs_classical)} pp | "
            f"shadow ≥ logistic + {min_vs_logistic} pp"
        )
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    x_train, y_train, x_val, y_val, _xt, _yt, _scaler = load_open_parquet_splits(
        dataset_id,
        ROOT,
        n_train_rows=n_train,
        n_val_rows=n_val,
        random_state=seed,
    )
    input_dim = int(x_train.shape[1])

    logistic = LogisticRegression(max_iter=500, random_state=seed)
    logistic.fit(x_train, y_train)
    logistic_auc = float(roc_auc_score(y_val, logistic.predict_proba(x_val)[:, 1]))
    if verbose:
        print(f"Logistic AUC={logistic_auc:.4f}", flush=True)

    histgb = HistGradientBoostingClassifier(max_iter=hgb_max_iter, random_state=seed)
    histgb.fit(x_train, y_train)
    histgb_auc = float(roc_auc_score(y_val, histgb.predict_proba(x_val)[:, 1]))
    if verbose:
        print(f"HistGB AUC={histgb_auc:.4f} (honesty floor)", flush=True)

    classical_auc, n_params_cl = _train_narrow(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        cfg=cfg,
        profile=profile,
        seed=seed,
        model_name="narrow_deep_raw",
    )
    if verbose:
        print(
            f"Classical NarrowDeep AUC={classical_auc:.4f} | params={n_params_cl:,}",
            flush=True,
        )

    encoder = PauliShadowFeatureEncoder(
        input_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_features=n_features,
        seed=seed,
    )
    t_feat = time.perf_counter()
    if verbose:
        print(
            f"Extracting {n_features}-d Pauli features "
            f"(qubits={n_qubits}, layers={n_layers})…",
            flush=True,
        )
    x_train_s = encoder.transform(x_train, progress_every=progress_every)
    x_val_s = encoder.transform(x_val, progress_every=progress_every)
    feature_extract_s = time.perf_counter() - t_feat
    if verbose:
        print(f"Feature extract done in {feature_extract_s:.1f}s", flush=True)

    shadow_auc, n_params_sh = _train_narrow(
        x_train=x_train_s,
        y_train=y_train,
        x_val=x_val_s,
        y_val=y_val,
        cfg=cfg,
        profile=profile,
        seed=seed,
        model_name="narrow_deep_shadow",
    )
    shadow_vs_cl = (shadow_auc - classical_auc) * 100.0
    shadow_vs_log = (shadow_auc - logistic_auc) * 100.0
    elapsed = time.perf_counter() - t0

    if verbose:
        ok_cl = "OK" if shadow_vs_cl >= min_vs_classical else "FAIL"
        ok_log = "OK" if shadow_vs_log >= min_vs_logistic else "FAIL"
        print(
            f"Shadow NarrowDeep AUC={shadow_auc:.4f} | "
            f"Δ vs classical={shadow_vs_cl:.2f} pp [{ok_cl}] | "
            f"Δ vs logistic={shadow_vs_log:.2f} pp [{ok_log}] | "
            f"params={n_params_sh:,}",
            flush=True,
        )

    log_event(
        "info",
        "exp_088 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        logistic_val_auc=round(logistic_auc, 6),
        classical_narrow_val_auc=round(classical_auc, 6),
        shadow_narrow_val_auc=round(shadow_auc, 6),
        histgb_val_auc=round(histgb_auc, 6),
        shadow_vs_classical_pp=round(shadow_vs_cl, 3),
        shadow_vs_logistic_pp=round(shadow_vs_log, 3),
        feature_extract_s=round(feature_extract_s, 3),
        elapsed_s=round(elapsed, 3),
    )

    return ShadowFeaturesNanoResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_shadow_features=n_features,
        logistic_val_auc=logistic_auc,
        classical_narrow_val_auc=classical_auc,
        shadow_narrow_val_auc=shadow_auc,
        histgb_val_auc=histgb_auc,
        shadow_vs_classical_pp=shadow_vs_cl,
        shadow_vs_logistic_pp=shadow_vs_log,
        min_vs_classical_pp=min_vs_classical,
        min_vs_logistic_pp=min_vs_logistic,
        n_params_classical=n_params_cl,
        n_params_shadow=n_params_sh,
        feature_extract_s=round(feature_extract_s, 3),
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: ShadowFeaturesNanoResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            f"# Results — EXP 088 Shadow / Pauli features → NarrowDeepNano",
            "",
            f"**Profile:** `{result.profile}`  ",
            f"**Verdict:** {verdict}  ",
            f"**Train / val rows:** {result.n_train_rows} / {result.n_val_rows}  ",
            f"**Shadow dims:** {result.n_shadow_features}  ",
            f"**Elapsed:** {result.elapsed_s:.1f}s (features {result.feature_extract_s:.1f}s)",
            "",
            "| Model | Val ROC-AUC | Notes |",
            "|-------|-------------|-------|",
            f"| LogisticRegression | {result.logistic_val_auc:.4f} | raw 37-d |",
            f"| NarrowDeepNano (raw) | {result.classical_narrow_val_auc:.4f} | "
            f"{result.n_params_classical:,} params |",
            f"| NarrowDeepNano (shadow) | {result.shadow_narrow_val_auc:.4f} | "
            f"{result.n_params_shadow:,} params |",
            f"| HistGB (honesty) | {result.histgb_val_auc:.4f} | not primary gate |",
            "",
            f"- Δ shadow − classical = **{result.shadow_vs_classical_pp:.2f} pp** "
            f"(need ≥ {result.min_vs_classical_pp:.1f})",
            f"- Δ shadow − logistic = **{result.shadow_vs_logistic_pp:.2f} pp** "
            f"(need ≥ {result.min_vs_logistic_pp:.1f})",
            "",
            "## Interpretation",
            "",
            (
                "Pauli/shadow feature map matched the classical NarrowDeep parity gate "
                "and cleared logistic."
                if verdict == "accepted"
                else "Pauli/shadow feature map did not clear H-Q2.3 gates — "
                "do not claim quantum feature advantage on maize."
            ),
            "",
            "## Limitations",
            "",
            "- Analytic `default.qubit` Pauli expectations (infinite-shot shadow limit).",
            "- Single seed; temporal val only.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    result = run_exp_088(
        profile=args.profile,
        verbose=not args.quiet,
        require_cuda=not args.allow_cpu,
    )
    summary = _summarize(result)
    print(summary)
    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(summary, encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
