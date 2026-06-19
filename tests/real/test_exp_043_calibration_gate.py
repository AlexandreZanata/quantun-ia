"""Real GPU gate for EXP 043 isotonic calibration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from experiments.exp_043_calibration_synthea.run import (
    MAX_ECE_AFTER,
    MIN_SPEARMAN,
    run_exp_043,
)


@pytest.mark.real
def test_exp_043_calibration_gate():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    os.environ["QML_DEVICE"] = "cuda"
    os.environ["MLFLOW_DISABLE"] = "1"
    result = run_exp_043(verbose=False, require_cuda=True)
    assert result.passed
    assert result.n_negatives >= 100
    assert result.ece_after <= MAX_ECE_AFTER
    assert result.spearman_rho >= MIN_SPEARMAN
    assert Path(result.artifact_path).is_file()
