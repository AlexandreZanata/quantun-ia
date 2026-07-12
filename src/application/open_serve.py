"""Serve artifacts for Phase L open datasets — checkpoint publish and feature contracts."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.classical.large_nano_mlp import LargeNanoMLP
from src.data.open_parquet import load_open_parquet_splits
from src.quantum.large_nano_hybrid import LargeNanoHybrid
from src.training.checkpoints import (
    checkpoint_path,
    load_checkpoint_bundle,
    resolve_checkpoint_dir,
    save_checkpoint,
    save_scaler,
)

OPEN_DATASET_FEATURES: dict[str, int] = {
    "higgs_v1": 28,
    "synthea_cv_risk_v1": 40,
    "nihr_cv_synthetic_v1": 13,
    "code_defects_gobug_v1": 23,
    "acyd_soy_brazil_v1": 37,
    "acyd_maize_brazil_v1": 37,
}

DEFAULT_SERVE_EXP_ID = "exp_032"
DEFAULT_SERVE_MODEL = "large_nano_mlp"
DEFAULT_SERVE_DATASET = "higgs_v1"
DEFAULT_SERVE_SEED = 42

DEFAULT_HYBRID_SERVE_EXP_ID = "exp_037"
DEFAULT_HYBRID_SERVE_MODEL = "large_nano_hybrid"


def open_dataset_feature_count(dataset_id: str) -> int:
    """Return expected raw feature count for an open dataset id."""
    if dataset_id not in OPEN_DATASET_FEATURES:
        msg = f"unknown open dataset: {dataset_id}"
        raise KeyError(msg)
    return OPEN_DATASET_FEATURES[dataset_id]


def load_open_holdout_rows(
    dataset_id: str,
    root: Path,
    *,
    n_rows: int,
    split: str = "test",
    random_state: int = 42,
) -> list[list[float]]:
    """Load raw (unscaled) feature rows from an open dataset split."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.data.open_manifest import get_dataset, load_manifest

    manifest = load_manifest(root / "data" / "open" / "manifest.json")
    dataset = get_dataset(manifest, dataset_id)
    if not dataset.get("ready"):
        msg = f"{dataset_id} is not ready"
        raise ValueError(msg)

    processed = root / "data" / "open" / dataset["path"]
    frame = pd.read_parquet(processed / dataset["files"][split])
    feature_cols = [f"feature_{i}" for i in range(int(dataset["n_features"]))]
    features = frame[feature_cols].to_numpy(dtype=np.float32)

    if n_rows < len(features):
        indices = np.arange(len(features))
        selected, _ = train_test_split(
            indices,
            train_size=n_rows,
            stratify=frame["label"].to_numpy(),
            random_state=random_state,
        )
        features = features[selected]

    return features.tolist()


def publish_large_nano_serve_artifact(
    root: Path,
    *,
    exp_id: str = DEFAULT_SERVE_EXP_ID,
    model_name: str = DEFAULT_SERVE_MODEL,
    dataset_id: str = DEFAULT_SERVE_DATASET,
    seed: int = DEFAULT_SERVE_SEED,
) -> Path:
    """Publish nanotrainer-compatible checkpoint + scaler for API/batch/chatbot."""
    source_dir = checkpoint_path(exp_id, model_name, seed)
    weights_path = source_dir / "best.pt"
    if not weights_path.is_file():
        msg = f"training checkpoint missing: {weights_path}"
        raise FileNotFoundError(msg)

    n_features = open_dataset_feature_count(dataset_id)
    x_train, _, _, _, _, _, scaler = load_open_parquet_splits(
        dataset_id,
        root,
        random_state=seed,
    )
    if int(x_train.shape[1]) != n_features:
        msg = f"expected {n_features} features, got {x_train.shape[1]}"
        raise ValueError(msg)

    payload = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    config = dict(payload.get("config", {}))
    config["input_dim"] = n_features
    config["dataset_id"] = dataset_id
    metadata = dict(payload.get("metadata", {}))
    metadata["serve_published"] = True

    model = LargeNanoMLP(input_dim=n_features)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    target_dir = resolve_checkpoint_dir(exp_id, model_name, dataset_id, seed=seed)
    save_checkpoint(model, target_dir, config=config, metadata=metadata)
    save_scaler(scaler, target_dir)
    return target_dir


def _hybrid_arch_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int | bool]:
    return {
        "hidden1": int(state_dict["backbone.0.weight"].shape[0]),
        "hidden2": int(state_dict["backbone.3.weight"].shape[0]),
        "hidden3": int(state_dict["backbone.6.weight"].shape[0]),
        "n_qubits": int(state_dict["head_proj.weight"].shape[0]),
    }


def build_large_nano_hybrid_for_inference(
    state_dict: dict,
    *,
    n_features: int,
    config: dict | None = None,
) -> LargeNanoHybrid:
    """Reconstruct hybrid model architecture from checkpoint weights for serve inference."""
    cfg = {**_hybrid_arch_from_state_dict(state_dict), **(config or {})}
    model = LargeNanoHybrid(
        input_dim=n_features,
        hidden1=int(cfg.get("hidden1", 2048)),
        hidden2=int(cfg.get("hidden2", 512)),
        hidden3=int(cfg.get("hidden3", 64)),
        dropout=float(cfg.get("dropout", 0.3)),
        n_qubits=int(cfg.get("n_qubits", 4)),
        n_layers=int(cfg.get("n_layers", 2)),
        reupload=bool(cfg.get("reupload", True)),
        backbone_device="cuda" if _serve_uses_cuda_backbone() else "cpu",
    )
    model.load_state_dict(state_dict)
    model.freeze_backbone()
    return model


def _serve_uses_cuda_backbone() -> bool:
    import os

    import torch

    return os.environ.get("QML_DEVICE", "cpu").lower() == "cuda" and torch.cuda.is_available()


def publish_large_nano_hybrid_serve_artifact(
    root: Path,
    *,
    exp_id: str = DEFAULT_HYBRID_SERVE_EXP_ID,
    model_name: str = DEFAULT_HYBRID_SERVE_MODEL,
    dataset_id: str = DEFAULT_SERVE_DATASET,
    seed: int = DEFAULT_SERVE_SEED,
) -> Path:
    """Publish hybrid nanotrainer checkpoint + scaler for API/batch/chatbot."""
    source_dir = checkpoint_path(exp_id, model_name, seed)
    weights_path = source_dir / "best.pt"
    if not weights_path.is_file():
        msg = f"hybrid training checkpoint missing: {weights_path}"
        raise FileNotFoundError(msg)

    n_features = open_dataset_feature_count(dataset_id)
    x_train, _, _, _, _, _, scaler = load_open_parquet_splits(
        dataset_id,
        root,
        random_state=seed,
    )
    if int(x_train.shape[1]) != n_features:
        msg = f"expected {n_features} features, got {x_train.shape[1]}"
        raise ValueError(msg)

    payload = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    train_config = dict(payload.get("config", {}))
    metadata = dict(payload.get("metadata", {}))
    metadata["serve_published"] = True

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    arch = _hybrid_arch_from_state_dict(state_dict)
    model = build_large_nano_hybrid_for_inference(
        state_dict,
        n_features=n_features,
        config=train_config,
    )

    serve_config = {
        **train_config,
        **arch,
        "input_dim": n_features,
        "dataset_id": dataset_id,
        "dropout": float(train_config.get("dropout", 0.3)),
        "n_layers": int(train_config.get("n_layers", 2)),
        "reupload": bool(train_config.get("reupload", True)),
    }

    target_dir = resolve_checkpoint_dir(exp_id, model_name, dataset_id, seed=seed)
    save_checkpoint(model, target_dir, config=serve_config, metadata=metadata)
    save_scaler(scaler, target_dir)
    return target_dir


def ensure_large_nano_hybrid_serve_artifact(
    root: Path,
    *,
    exp_id: str = DEFAULT_HYBRID_SERVE_EXP_ID,
    model_name: str = DEFAULT_HYBRID_SERVE_MODEL,
    dataset_id: str = DEFAULT_SERVE_DATASET,
    seed: int = DEFAULT_SERVE_SEED,
) -> Path:
    """Return hybrid serve checkpoint dir, publishing from exp_037 weights if needed."""
    target_dir = resolve_checkpoint_dir(exp_id, model_name, dataset_id, seed=seed)
    if (target_dir / "best.pt").is_file() and (target_dir / "scaler.joblib").is_file():
        return target_dir
    return publish_large_nano_hybrid_serve_artifact(
        root,
        exp_id=exp_id,
        model_name=model_name,
        dataset_id=dataset_id,
        seed=seed,
    )


def ensure_large_nano_serve_artifact(
    root: Path,
    *,
    exp_id: str = DEFAULT_SERVE_EXP_ID,
    model_name: str = DEFAULT_SERVE_MODEL,
    dataset_id: str = DEFAULT_SERVE_DATASET,
    seed: int = DEFAULT_SERVE_SEED,
) -> Path:
    """Return serve checkpoint dir, publishing from exp_032 training weights if needed."""
    target_dir = resolve_checkpoint_dir(exp_id, model_name, dataset_id, seed=seed)
    if (target_dir / "best.pt").is_file() and (target_dir / "scaler.joblib").is_file():
        return target_dir
    return publish_large_nano_serve_artifact(
        root,
        exp_id=exp_id,
        model_name=model_name,
        dataset_id=dataset_id,
        seed=seed,
    )


def verify_serve_artifact(
    exp_id: str,
    model_name: str,
    dataset_id: str,
    *,
    seed: int = DEFAULT_SERVE_SEED,
) -> None:
    """Validate serve checkpoint bundle exists and input_dim matches dataset."""
    bundle = load_checkpoint_bundle(exp_id, model_name, dataset_id, seed=seed)
    expected = open_dataset_feature_count(dataset_id)
    actual = int(bundle.config.get("input_dim", 0))
    if actual != expected:
        msg = f"checkpoint input_dim {actual} != {expected} for {dataset_id}"
        raise ValueError(msg)
