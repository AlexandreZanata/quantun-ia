"""Real gate — exp_056 re-upload depth curriculum ladder (RTX 4060)."""

from __future__ import annotations

import pytest

from experiments.exp_056_reupload_curriculum_ladder.run import gate_passed, run_exp_056

pytestmark = pytest.mark.real


def test_exp_056_reupload_ladder_ci_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_056 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_056(profile="ci", verbose=False, require_cuda=True)

    assert result.n_rungs == 3
    assert len(result.rung_results) == 3
    for rung in result.rung_results:
        assert 0.0 <= rung.curriculum_score <= 1.0
        assert 0.0 <= rung.fixed_score <= 1.0
    assert gate_passed(result)
