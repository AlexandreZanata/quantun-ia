"""Real gate — exp_029 batch vs API parity on RTX 4060 (569 rows)."""

from __future__ import annotations

import pytest

from experiments.exp_029_batch_calc_parity.run import MAX_DELTA, run_exp_029

pytestmark = pytest.mark.real


def test_exp_029_batch_api_parity_publication(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_029 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")

    result = run_exp_029(profile="publication", verbose=False, bootstrap_checkpoint=True)
    assert result.n_rows == 569
    assert result.max_delta < MAX_DELTA
