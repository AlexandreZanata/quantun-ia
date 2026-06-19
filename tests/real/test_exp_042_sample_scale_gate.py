"""Real GPU gate for EXP 042 sample-scale precision."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch


def _load_run_module():
    path = Path("experiments/exp_042_sample_scale_precision/run.py")
    spec = importlib.util.spec_from_file_location("exp_042_run", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.real
def test_exp_042_sample_scale_gate():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    os.environ["QML_DEVICE"] = "cuda"
    os.environ["MLFLOW_DISABLE"] = "1"
    mod = _load_run_module()
    result = mod.run_exp_042(verbose=False, require_cuda=True)
    assert result.passed
    assert result.min_roc_auc >= 0.78
    assert result.predictions_n_rows == 100
    assert len(result.curve_points) == 20
