"""Real gate — exp_053 dynamic entanglement schedule on breast cancer (RTX 4060)."""

from __future__ import annotations

import pytest

from experiments.exp_053_entangle_schedule_bc.run import FIXED_TOPOLOGIES, gate_passed, run_exp_053

pytestmark = pytest.mark.real


def test_exp_053_entangle_schedule_ci_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_053 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_053(profile="ci", verbose=False, require_cuda=True)

    assert result.n_seeds == 1
    assert result.n_stages == 2
    assert result.n_train_rows > 0
    assert result.n_holdout_rows > 0
    for topo in FIXED_TOPOLOGIES:
        assert topo in result.mean_by_topology
        assert 0.0 < result.mean_by_topology[topo] <= 1.0
    assert 0.0 < result.mean_schedule <= 1.0
    assert gate_passed(result)
