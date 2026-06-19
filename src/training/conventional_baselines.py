"""Conventional sklearn/XGBoost baselines for open-tabular benchmark comparisons."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from src.classical.large_nano_mlp import LargeNanoMLP
from src.classical.xgboost_baseline import XGBoostShallow
from src.data.open_parquet import load_open_parquet_splits
from src.training.config import load_experiment_config
from src.training.trainer import count_parameters

EXP_032_KEY = "exp_032_large_nano_higgs"
EXP_060_KEY = "exp_060_large_nano_acyd_soy"
EXP_069_KEY = "exp_069_large_nano_nihr"
DEFAULT_HIGGS_SERVE_DIR = Path("dist/serve_models/large_nano_mlp_higgs")
DEFAULT_ACYD_ARTIFACT_DIR = Path("artifacts/exp_060/large_nano_mlp/seed_42")
DEFAULT_NIHR_ARTIFACT_DIR = Path("artifacts/exp_069/large_nano_mlp/seed_42")


@dataclass(frozen=True)
class ConventionalBaselineScore:
    model_key: str
    display_name: str
    roc_auc: float
    accuracy: float
    n_params: int
    train_s: float
    source: str


@dataclass(frozen=True)
class ConventionalComparisonResult:
    profile: str
    n_train_rows: int
    n_val_rows: int
    scores: tuple[ConventionalBaselineScore, ...]
    best_conventional_auc: float
    nano_auc: float
    advantage_vs_best_conventional_pp: float
    min_advantage_pp: float
    elapsed_s: float


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _eval_sklearn(estimator, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    proba = estimator.predict_proba(x)[:, 1]
    preds = (proba >= 0.5).astype(np.float32)
    return float(roc_auc_score(y, proba)), float(accuracy_score(y, preds))


def _eval_sklearn_pr(estimator, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    proba = estimator.predict_proba(x)[:, 1]
    preds = (proba >= 0.5).astype(np.float32)
    pr = average_precision_score(y, proba)
    return float(pr), float(accuracy_score(y, preds))


@torch.no_grad()
def _eval_torch(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    *,
    primary_metric: str = "roc_auc",
) -> tuple[float, float]:
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    proba = model(xt).detach().cpu().numpy().reshape(-1)
    preds = (proba >= 0.5).astype(np.float32)
    if primary_metric == "pr_auc":
        return float(average_precision_score(y, proba)), float(accuracy_score(y, preds))
    return float(roc_auc_score(y, proba)), float(accuracy_score(y, preds))


def _load_nano_checkpoint(
    input_dim: int,
    weights_dir: Path,
    device: torch.device,
    *,
    missing_hint: str,
) -> LargeNanoMLP:
    weights = weights_dir / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"LargeNanoMLP checkpoint not found at {weights} — {missing_hint}")
    model = LargeNanoMLP(input_dim=input_dim)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def _load_shipped_nano(input_dim: int, serve_dir: Path, device: torch.device) -> LargeNanoMLP:
    return _load_nano_checkpoint(
        input_dim,
        serve_dir,
        device,
        missing_hint="run qml-ship --model large_nano_mlp_higgs",
    )


def _run_conventional_open_comparison(
    root: Path,
    *,
    dataset_id: str,
    profile: str,
    train_exp_key: str,
    cmp_exp_key: str,
    nano_weights_dir: Path,
    max_train_rows: int,
    max_val_rows: int,
    xgb_exp_id: str,
    primary_metric: str = "roc_auc",
) -> ConventionalComparisonResult:
    """Compare a LargeNanoMLP checkpoint vs conventional tabular baselines on val split."""
    cfg = load_experiment_config(train_exp_key, profile=profile)
    cmp_cfg = load_experiment_config(cmp_exp_key, profile=profile)
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), max_train_rows)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), max_val_rows)
    epochs = int(cfg.get("epochs", 12))
    batch_size = int(cfg.get("batch_size", 2048))
    lr = float(cfg.get("learning_rate", 0.001))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    min_advantage_pp = float(cmp_cfg.get("min_advantage_pp", 0.5))
    xgb_lr = float(cmp_cfg.get("xgboost_learning_rate", 0.1))
    xgb_estimators = int(cmp_cfg.get("xgboost_n_estimators", 50))
    hgb_max_iter = int(cmp_cfg.get("histgb_max_iter", 100))
    hidden1 = int(cfg.get("hidden1", 2048))
    hidden2 = int(cfg.get("hidden2", 512))
    hidden3 = int(cfg.get("hidden3", 64))
    metric = str(cmp_cfg.get("primary_metric", primary_metric))
    eval_sklearn = _eval_sklearn_pr if metric == "pr_auc" else _eval_sklearn

    t0 = time.perf_counter()
    x_train, y_train, x_val, y_val, _, _, _ = load_open_parquet_splits(
        dataset_id,
        root,
        n_train_rows=n_train,
        n_val_rows=n_val,
        random_state=seed,
    )
    input_dim = int(x_train.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores: list[ConventionalBaselineScore] = []

    t_model = time.perf_counter()
    nano = _load_nano_checkpoint(
        input_dim,
        nano_weights_dir,
        device,
        missing_hint=f"train {train_exp_key} first",
    )
    auc, acc = _eval_torch(nano, x_val, y_val, device, primary_metric=metric)
    scores.append(
        ConventionalBaselineScore(
            model_key="large_nano_mlp",
            display_name="LargeNanoMLP (quantun-ia)",
            roc_auc=auc,
            accuracy=acc,
            n_params=count_parameters(nano),
            train_s=time.perf_counter() - t_model,
            source=str(nano_weights_dir / "best.pt"),
        )
    )

    t_model = time.perf_counter()
    logistic = LogisticRegression(max_iter=500, random_state=seed)
    logistic.fit(x_train, y_train)
    auc, acc = eval_sklearn(logistic, x_val, y_val)
    scores.append(
        ConventionalBaselineScore(
            model_key="logistic_regression",
            display_name="LogisticRegression (sklearn)",
            roc_auc=auc,
            accuracy=acc,
            n_params=int(logistic.coef_.size + logistic.intercept_.size),
            train_s=time.perf_counter() - t_model,
            source="sklearn.linear_model.LogisticRegression",
        )
    )

    t_model = time.perf_counter()
    sklearn_mlp = MLPClassifier(
        hidden_layer_sizes=(hidden1, hidden2, hidden3),
        activation="relu",
        solver="adam",
        alpha=weight_decay,
        batch_size=min(batch_size, len(y_train)),
        learning_rate_init=lr,
        max_iter=epochs,
        random_state=seed,
        early_stopping=False,
        verbose=False,
    )
    sklearn_mlp.fit(x_train, y_train)
    auc, acc = eval_sklearn(sklearn_mlp, x_val, y_val)
    n_mlp_params = sum(w.size for w in sklearn_mlp.coefs_) + sum(b.size for b in sklearn_mlp.intercepts_)
    scores.append(
        ConventionalBaselineScore(
            model_key="sklearn_mlp_matched",
            display_name=f"MLPClassifier (sklearn, {hidden1}-{hidden2}-{hidden3})",
            roc_auc=auc,
            accuracy=acc,
            n_params=int(n_mlp_params),
            train_s=time.perf_counter() - t_model,
            source="sklearn.neural_network.MLPClassifier",
        )
    )

    t_model = time.perf_counter()
    histgb = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.1,
        max_iter=hgb_max_iter,
        random_state=seed,
    )
    histgb.fit(x_train, y_train)
    auc, acc = eval_sklearn(histgb, x_val, y_val)
    scores.append(
        ConventionalBaselineScore(
            model_key="hist_gradient_boosting",
            display_name="HistGradientBoosting (sklearn)",
            roc_auc=auc,
            accuracy=acc,
            n_params=0,
            train_s=time.perf_counter() - t_model,
            source="sklearn.ensemble.HistGradientBoostingClassifier",
        )
    )

    t_model = time.perf_counter()
    xgb = XGBoostShallow(
        input_dim=input_dim,
        n_estimators=xgb_estimators,
        learning_rate=xgb_lr,
        random_state=seed,
    )
    xgb.train(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        xgb_exp_id,
        "xgboost_shallow",
        seed=seed,
        profile=profile,
        save_checkpoints=False,
    )
    auc, acc = eval_sklearn(xgb.estimator, x_val, y_val)
    scores.append(
        ConventionalBaselineScore(
            model_key="xgboost_shallow",
            display_name="XGBoost shallow (xgboost)",
            roc_auc=auc,
            accuracy=acc,
            n_params=xgb.count_sklearn_parameters(),
            train_s=time.perf_counter() - t_model,
            source="xgboost.XGBClassifier",
        )
    )

    conventional_aucs = [s.roc_auc for s in scores if s.model_key != "large_nano_mlp"]
    best_conventional = max(conventional_aucs)
    nano_auc = next(s.roc_auc for s in scores if s.model_key == "large_nano_mlp")
    advantage_pp = (nano_auc - best_conventional) * 100.0

    return ConventionalComparisonResult(
        profile=profile,
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        scores=tuple(scores),
        best_conventional_auc=best_conventional,
        nano_auc=nano_auc,
        advantage_vs_best_conventional_pp=advantage_pp,
        min_advantage_pp=min_advantage_pp,
        elapsed_s=round(time.perf_counter() - t0, 3),
    )


def run_conventional_higgs_comparison(
    root: Path,
    *,
    profile: str = "ci",
    serve_dir: Path | None = None,
    checkpoint_exp_key: str = EXP_032_KEY,
) -> ConventionalComparisonResult:
    """Compare shipped LargeNanoMLP vs conventional tabular baselines on HIGGS val."""
    bundle_dir = serve_dir or (root / DEFAULT_HIGGS_SERVE_DIR)
    return _run_conventional_open_comparison(
        root,
        dataset_id="higgs_v1",
        profile=profile,
        train_exp_key=checkpoint_exp_key,
        cmp_exp_key="exp_058_conventional_higgs_baselines",
        nano_weights_dir=bundle_dir,
        max_train_rows=805_000,
        max_val_rows=172_500,
        xgb_exp_id="exp_058",
    )


def run_conventional_acyd_comparison(
    root: Path,
    *,
    profile: str = "ci",
    weights_dir: Path | None = None,
    checkpoint_exp_key: str = EXP_060_KEY,
) -> ConventionalComparisonResult:
    """Compare exp_060 LargeNanoMLP vs conventional tabular baselines on ACYD val."""
    artifact_dir = weights_dir or (root / DEFAULT_ACYD_ARTIFACT_DIR)
    return _run_conventional_open_comparison(
        root,
        dataset_id="acyd_soy_brazil_v1",
        profile=profile,
        train_exp_key=checkpoint_exp_key,
        cmp_exp_key="exp_061_conventional_acyd_baselines",
        nano_weights_dir=artifact_dir,
        max_train_rows=50_107,
        max_val_rows=5_830,
        xgb_exp_id="exp_061",
    )


def run_conventional_nihr_comparison(
    root: Path,
    *,
    profile: str = "ci",
    weights_dir: Path | None = None,
    checkpoint_exp_key: str = EXP_069_KEY,
) -> ConventionalComparisonResult:
    """Compare exp_069 LargeNanoMLP vs conventional tabular baselines on NIHR val (PR-AUC)."""
    artifact_dir = weights_dir or (root / DEFAULT_NIHR_ARTIFACT_DIR)
    return _run_conventional_open_comparison(
        root,
        dataset_id="nihr_cv_synthetic_v1",
        profile=profile,
        train_exp_key=checkpoint_exp_key,
        cmp_exp_key="exp_076_conventional_nihr_baselines",
        nano_weights_dir=artifact_dir,
        max_train_rows=70_000,
        max_val_rows=15_000,
        xgb_exp_id="exp_076",
        primary_metric="pr_auc",
    )


def gate_passed(result: ConventionalComparisonResult) -> bool:
    return result.advantage_vs_best_conventional_pp >= result.min_advantage_pp
