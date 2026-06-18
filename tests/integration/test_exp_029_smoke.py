"""Integration smoke test — exp_029 batch vs API parity."""

from __future__ import annotations

from pathlib import Path

from experiments.exp_029_batch_calc_parity.run import MAX_DELTA, run_exp_029

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "breast_cancer_holdout.csv"


def test_exp_029_ci_batch_api_parity_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_029(
        profile="ci",
        input_path=FIXTURE,
        verbose=False,
        require_cuda=False,
        bootstrap_checkpoint=True,
    )

    assert result.n_rows == 50
    assert result.max_delta < MAX_DELTA
