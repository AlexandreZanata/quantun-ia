"""Conventional sklearn/XGBoost baselines for open-tabular benchmark comparisons."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from src.classical.large_nano_mlp import LargeNanoMLP
from src.classical.xgboost_baseline import XGBoostShallow
from src.data.open_parquet import load_open_parquet_splits
from src.training.config import load_experiment_config
from src.training.trainer import count_parameters

EXP_032_KEY = "exp_032_large_nano_higgs"
DEFAULT_SERVE_DIR = Path("dist/serve_models/large_nano_mlp_higgs")


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


@torch.no_grad()
def _eval_torch(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[float, float]:
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    proba = model(xt).detach().cpu().numpy().reshape(-1)
    preds = (proba >= 0.5).astype(np.float32)
    return float(roc_auc_score(y, proba)), float(accuracy_score(y, preds))


def _load_shipped_nano(input_dim: int, serve_dir: Path, device: torch.device) -> LargeNanoMLP:
    weights = serve_dir / "best.pt"
    if not weights.is_file():
        msg = f"Shipped LargeNanoMLP not found at {weights} — run qml-ship --model large_nano_mlp_higgs"
        raise FileNotFoundError(msg)
    model = LargeNanoMLP(input_dim=input_dim)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def run_conventional_higgs_comparison(
    root: Path,
    *,
    profile: str = "ci",
    serve_dir: Path | None = None,
    checkpoint_exp_key: str = EXP_032_KEY,
) -> ConventionalComparisonResult:
    """Compare shipped LargeNanoMLP vs conventional tabular baselines on HIGGS val."""
    cfg = load_experiment_config(checkpoint_exp_key, profile=profile)
    cmp_cfg = load_experiment_config("exp_058_conventional_higgs_baselines", profile=profile)
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 805_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 172_500)
    epochs = int(cfg.get("epochs", 12))
    batch_size = int(cfg.get("batch_size", 2048))
    lr = float(cfg.get("learning_rate", 0.001))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    min_advantage_pp = float(cmp_cfg.get("min_advantage_pp", 0.5))
    xgb_lr = float(cmp_cfg.get("xgboost_learning_rate", 0.1))
    xgb_estimators = int(cmp_cfg.get("xgboost_n_estimators", 50))
    hgb_max_iter = int(cmp_cfg.get("histgb_max_iter", 100))

    bundle_dir = serve_dir or (root / DEFAULT_SERVE_DIR)
    t0 = time.perf_counter()

    x_train, y_train, x_val, y_val, _, _, _ = load_open_parquet_splits(
        "higgs_v1",
        root,
        n_train_rows=n_train,
        n_val_rows=n_val,
        random_state=seed,
    )
    input_dim = int(x_train.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores: list[ConventionalBaselineScore] = []

    t_model = time.perf_counter()
    nano = _load_shipped_nano(input_dim, bundle_dir, device)
    auc, acc = _eval_torch(nano, x_val, y_val, device)
    scores.append(
        ConventionalBaselineScore(
            model_key="large_nano_mlp",
            display_name="LargeNanoMLP (quantun-ia)",
            roc_auc=auc,
            accuracy=acc,
            n_params=count_parameters(nano),
            train_s=time.perf_counter() - t_model,
            source=str(bundle_dir / "best.pt"),
        )
    )

    t_model = time.perf_counter()
    logistic = LogisticRegression(max_iter=500, random_state=seed)
    logistic.fit(x_train, y_train)
    auc, acc = _eval_sklearn(logistic, x_val, y_val)
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
        hidden_layer_sizes=(2048, 512, 64),
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
    auc, acc = _eval_sklearn(sklearn_mlp, x_val, y_val)
    n_mlp_params = sum(w.size for w in sklearn_mlp.coefs_) + sum(b.size for b in sklearn_mlp.intercepts_)
    scores.append(
        ConventionalBaselineScore(
            model_key="sklearn_mlp_matched",
            display_name="MLPClassifier (sklearn, 2048-512-64)",
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
    auc, acc = _eval_sklearn(histgb, x_val, y_val)
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
        "exp_058",
        "xgboost_shallow",
        seed=seed,
        profile=profile,
        save_checkpoints=False,
    )
    auc, acc = _eval_sklearn(xgb.estimator, x_val, y_val)
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


def gate_passed(result: ConventionalComparisonResult) -> bool:
    return result.advantage_vs_best_conventional_pp >= result.min_advantage_pp
