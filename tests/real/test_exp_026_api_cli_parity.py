"""Real gate — exp_026 API vs CLI holdout parity on GPU."""

from __future__ import annotations

import pytest

from experiments.exp_026_real_app_e2e.run import MAX_DELTA_PP, run_exp_026

pytestmark = pytest.mark.real


def test_exp_026_api_cli_parity_ci(tmp_path, monkeypatch):
    """Async API (cuda) matches CLI within 0.5 pp for hybrid_sandwich × breast_cancer."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_026 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")

    pairs = run_exp_026(profile="ci", db_path=tmp_path / "exp_026.db", verbose=False)
    assert pairs, "expected at least one seed pair"

    for pair in pairs:
        assert 0.55 <= pair.cli_accuracy <= 1.0
        assert 0.55 <= pair.api_accuracy <= 1.0
        assert pair.delta_pp <= MAX_DELTA_PP, (
            f"seed={pair.seed} delta {pair.delta_pp:.2f} pp > {MAX_DELTA_PP} pp"
        )
