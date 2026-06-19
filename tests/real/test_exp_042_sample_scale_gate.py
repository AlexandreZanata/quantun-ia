"""Real GPU gate for EXP 042 sample-scale precision."""

from __future__ import annotations

import os

import pytest
import torch

from experiments.exp_042_sample_scale_precision.run import run_exp_042


@pytest.mark.real
def test_exp_042_sample_scale_gate():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    os.environ["QML_DEVICE"] = "cuda"
    os.environ["MLFLOW_DISABLE"] = "1"
    result = run_exp_042(verbose=False, require_cuda=True)
    assert result.passed
    assert result.min_roc_auc >= 0.78
    assert result.predictions_n_rows == 100
    assert len(result.curve_points) == 20
    first = result.curve_points[0]
    assert first.n_rows == 100
    assert first.n_negatives >= 10
