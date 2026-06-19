"""Real gate — serve model evaluation on RTX 4060 (5K HIGGS val rows)."""

from __future__ import annotations

import pytest

from src.application.evaluate_serve_model import EvaluateServeModelDTO, execute
from src.shared.result import Ok

pytestmark = pytest.mark.real


def test_evaluate_large_nano_higgs_gate():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for serve evaluation gate")

    outcome = execute(
        EvaluateServeModelDTO(
            exp_id="exp_032",
            model_name="large_nano_mlp",
            dataset="higgs_v1",
            split="val",
            n_rows=5000,
            chunk_size=2048,
        )
    )
    assert isinstance(outcome, Ok), outcome
    r = outcome.value
    assert r.n_rows == 5000
    assert r.roc_auc > 0.80
    assert r.accuracy > 0.75
    assert r.confusion.true_positive + r.confusion.true_negative > 0
