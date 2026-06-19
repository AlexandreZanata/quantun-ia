"""Unit tests for conventional HIGGS baseline comparison."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.classical.large_nano_mlp import LargeNanoMLP
from src.training.conventional_baselines import (
    ConventionalBaselineScore,
    ConventionalComparisonResult,
    gate_passed,
)


def test_gate_passed_when_nano_beats_conventional():
    scores = (
        ConventionalBaselineScore(
            "large_nano_mlp", "nano", 0.90, 0.8, 1_000_000, 1.0, "ckpt"
        ),
        ConventionalBaselineScore(
            "logistic_regression", "logistic", 0.80, 0.7, 29, 1.0, "sklearn"
        ),
    )
    result = ConventionalComparisonResult(
        profile="ci",
        n_train_rows=1000,
        n_val_rows=200,
        scores=scores,
        best_conventional_auc=0.80,
        nano_auc=0.90,
        advantage_vs_best_conventional_pp=10.0,
        min_advantage_pp=0.5,
        elapsed_s=1.0,
    )
    assert gate_passed(result)


def test_gate_fails_when_nano_below_threshold():
    scores = (
        ConventionalBaselineScore(
            "large_nano_mlp", "nano", 0.801, 0.8, 1_000_000, 1.0, "ckpt"
        ),
        ConventionalBaselineScore(
            "hist_gradient_boosting", "hgb", 0.800, 0.7, 0, 1.0, "sklearn"
        ),
    )
    result = ConventionalComparisonResult(
        profile="ci",
        n_train_rows=1000,
        n_val_rows=200,
        scores=scores,
        best_conventional_auc=0.800,
        nano_auc=0.801,
        advantage_vs_best_conventional_pp=0.1,
        min_advantage_pp=0.5,
        elapsed_s=1.0,
    )
    assert not gate_passed(result)


def test_run_conventional_higgs_comparison_ci(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    root = tmp_path
    serve = root / "dist/serve_models/large_nano_mlp_higgs"
    serve.mkdir(parents=True)

    model = LargeNanoMLP(input_dim=4)
    torch.save(model.state_dict(), serve / "best.pt")

    rng = np.random.default_rng(42)
    n_train, n_val = 200, 50
    x_train = rng.normal(size=(n_train, 4)).astype(np.float32)
    y_train = (x_train[:, 0] > 0).astype(np.float32)
    x_val = rng.normal(size=(n_val, 4)).astype(np.float32)
    y_val = (x_val[:, 0] > 0).astype(np.float32)

    def _fake_load(_dataset_id, _root, **kwargs):
        return x_train, y_train, x_val, y_val, x_val, y_val, None

    monkeypatch.setattr(
        "src.training.conventional_baselines.load_open_parquet_splits",
        _fake_load,
    )

    from src.training.conventional_baselines import run_conventional_higgs_comparison

    result = run_conventional_higgs_comparison(root, profile="ci", serve_dir=serve)
    assert result.n_train_rows == 200
    assert len(result.scores) == 5
    assert any(s.model_key == "large_nano_mlp" for s in result.scores)
    assert any(s.model_key == "xgboost_shallow" for s in result.scores)


def test_run_conventional_nihr_comparison_ci(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    root = tmp_path
    weights = root / "artifacts/exp_069/large_nano_mlp/seed_42"
    weights.mkdir(parents=True)

    model = LargeNanoMLP(input_dim=4)
    torch.save(model.state_dict(), weights / "best.pt")

    rng = np.random.default_rng(42)
    n_train, n_val = 200, 50
    x_train = rng.normal(size=(n_train, 4)).astype(np.float32)
    y_train = (x_train[:, 0] > 0.2).astype(np.float32)
    x_val = rng.normal(size=(n_val, 4)).astype(np.float32)
    y_val = (x_val[:, 0] > 0.2).astype(np.float32)

    def _fake_load(_dataset_id, _root, **kwargs):
        return x_train, y_train, x_val, y_val, x_val, y_val, None

    monkeypatch.setattr(
        "src.training.conventional_baselines.load_open_parquet_splits",
        _fake_load,
    )

    from src.training.conventional_baselines import run_conventional_nihr_comparison

    result = run_conventional_nihr_comparison(root, profile="ci", weights_dir=weights)
    assert result.n_train_rows == 200
    assert len(result.scores) == 5
    assert result.nano_auc >= 0.0
    assert result.nano_auc <= 1.0


def test_run_conventional_gobug_comparison_ci(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    root = tmp_path
    weights = root / "artifacts/exp_070/large_nano_mlp/seed_42"
    weights.mkdir(parents=True)

    model = LargeNanoMLP(input_dim=4)
    torch.save(model.state_dict(), weights / "best.pt")

    rng = np.random.default_rng(42)
    n_train, n_val = 200, 50
    x_train = rng.normal(size=(n_train, 4)).astype(np.float32)
    y_train = (x_train[:, 0] > 0.2).astype(np.float32)
    x_val = rng.normal(size=(n_val, 4)).astype(np.float32)
    y_val = (x_val[:, 0] > 0.2).astype(np.float32)

    def _fake_load(_dataset_id, _root, **kwargs):
        return x_train, y_train, x_val, y_val, x_val, y_val, None

    monkeypatch.setattr(
        "src.training.conventional_baselines.load_open_parquet_splits",
        _fake_load,
    )

    from src.training.conventional_baselines import run_conventional_gobug_comparison

    result = run_conventional_gobug_comparison(root, profile="ci", weights_dir=weights)
    assert result.n_train_rows == 200
    assert len(result.scores) == 5
    assert result.nano_auc >= 0.0
    assert result.nano_auc <= 1.0
