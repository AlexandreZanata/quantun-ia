"""Integration smoke — evaluate serve model imports and shapes."""

from __future__ import annotations

import pytest

from src.application.evaluate_serve_model import EvaluateServeModelDTO, execute


def test_evaluate_serve_model_import():
    dto = EvaluateServeModelDTO(
        exp_id="exp_032",
        model_name="large_nano_mlp",
        dataset="higgs_v1",
        n_rows=64,
    )
    assert dto.chunk_size == 2048


@pytest.mark.integration
def test_evaluate_serve_model_smoke_on_higgs():
    """Score 256 val rows if HIGGS data + exp_032 checkpoint exist."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    manifest = root / "data" / "open" / "manifest.json"
    ckpt = root / "artifacts" / "exp_032" / "large_nano_mlp" / "seed_42" / "best.pt"
    if not manifest.is_file() or not ckpt.is_file():
        pytest.skip("HIGGS manifest or exp_032 checkpoint missing")

    outcome = execute(
        EvaluateServeModelDTO(
            exp_id="exp_032",
            model_name="large_nano_mlp",
            dataset="higgs_v1",
            n_rows=256,
            chunk_size=128,
        )
    )
    from src.shared.result import Fail, Ok

    assert isinstance(outcome, Ok), outcome
    r = outcome.value
    assert r.n_rows == 256
    assert 0.5 < r.roc_auc <= 1.0
    assert 0.0 <= r.accuracy <= 1.0
    assert len(r.fpr) == len(r.tpr)
    assert len(r.sample_rows) == 10
