"""Real gate — exp_031 curriculum beats random on breast cancer (RTX 4060)."""

from __future__ import annotations

import pytest

from experiments.exp_031_curriculum_clinical.run import run_exp_031

pytestmark = pytest.mark.real


def test_exp_031_curriculum_clinical_publication(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_031 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_031(profile="publication", verbose=False, require_cuda=True)

    assert result.n_seeds == 10
    assert result.applicable
    assert result.advantage_pp > result.min_advantage_pp
