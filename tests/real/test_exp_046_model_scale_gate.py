"""Real gate — exp_046 model scale curve on RTX 4060 (CI subset)."""

from __future__ import annotations

import pytest

from experiments.exp_046_model_scale_curve.run import gate_passed, run_exp_046

pytestmark = pytest.mark.real


def test_exp_046_model_scale_ci_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_046 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_046(profile="ci", verbose=False, require_cuda=True)

    assert result.n_train_rows == 50_000
    assert result.n_val_rows == 10_000
    assert len(result.variants) == 3
    keys = {v.variant_key for v in result.variants}
    assert keys == {"nano_s", "nano_l", "nano_xl"}
    for variant in result.variants:
        if variant.oom:
            pytest.fail(f"{variant.variant_key} OOM on CI subset — unexpected")
        assert variant.n_params > 0
        assert 0.5 < variant.val_roc_auc < 1.0
        assert variant.peak_vram_mb is not None
    assert gate_passed(result)
