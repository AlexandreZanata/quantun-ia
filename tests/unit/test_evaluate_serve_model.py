"""Unit tests — evaluate_serve_model helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.evaluate_serve_model import (
    SERVE_MODELS,
    ConfusionStats,
    EvaluateServeModelDTO,
    _confusion_stats,
    list_serve_models,
    load_open_split_labeled,
)


def test_list_serve_models_includes_higgs_and_synthea():
    labels = {m.label for m in list_serve_models()}
    assert "LargeNanoMLP — HIGGS" in labels
    assert "LargeNano Hybrid — HIGGS" in labels
    assert "LargeNanoMLP — Synthea CV" in labels
    assert len(SERVE_MODELS) == 3


def test_confusion_stats_perfect_predictions():
    stats = _confusion_stats([0, 0, 1, 1], [0, 0, 1, 1])
    assert stats == ConfusionStats(true_negative=2, false_positive=0, false_negative=0, true_positive=2)


def test_load_open_split_labeled_stratified_cap(tmp_path: Path):
    root = tmp_path
    processed = root / "data" / "open" / "higgs" / "processed" / "v1"
    processed.mkdir(parents=True)
    (root / "data" / "open").mkdir(parents=True, exist_ok=True)
    (root / "data" / "open" / "manifest.json").write_text(
        """
{
  "version": 1,
  "datasets": [{
    "id": "higgs_v1",
    "ready": true,
    "path": "higgs/processed/v1",
    "n_features": 3,
    "files": {"val": "val.parquet"}
  }]
}
""".strip(),
        encoding="utf-8",
    )
    rng = np.random.default_rng(42)
    n = 200
    frame = pd.DataFrame(
        {
            "feature_0": rng.normal(size=n),
            "feature_1": rng.normal(size=n),
            "feature_2": rng.normal(size=n),
            "label": rng.integers(0, 2, size=n),
        }
    )
    frame.to_parquet(processed / "val.parquet", index=False)

    features, labels = load_open_split_labeled(
        "higgs_v1",
        root,
        split="val",
        n_rows=50,
        random_state=42,
    )
    assert len(features) == 50
    assert len(labels) == 50
    assert len(features[0]) == 3


def test_evaluate_dto_defaults():
    dto = EvaluateServeModelDTO(
        exp_id="exp_032",
        model_name="large_nano_mlp",
        dataset="higgs_v1",
    )
    assert dto.split == "val"
    assert dto.n_rows == 5000
    assert dto.seed == 42
