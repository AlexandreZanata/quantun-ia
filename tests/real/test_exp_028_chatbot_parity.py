"""Real gate — exp_028 chatbot tool vs API parity on RTX 4060."""

from __future__ import annotations

import pytest

from experiments.exp_028_chatbot_tool_parity.run import MAX_DELTA, MAX_LATENCY_S, run_exp_028

pytestmark = pytest.mark.real


def test_exp_028_chatbot_tool_api_parity_ci(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_028 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")

    results = run_exp_028(profile="ci", verbose=False, bootstrap_checkpoint=True)
    assert len(results) == 10
    assert all(r.has_disclaimer for r in results)
    assert max(r.max_delta for r in results) < MAX_DELTA
    assert max(r.latency_s for r in results) <= MAX_LATENCY_S
