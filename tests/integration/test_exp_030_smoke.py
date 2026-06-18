"""Integration smoke test — exp_030 scale stability on circles."""

from __future__ import annotations

from experiments.exp_030_publication_large.run import run_exp_030


def test_exp_030_ci_scale_stability_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_030(profile="ci", verbose=False, require_cuda=False)

    assert result.n_samples == 100
    assert result.n_seeds == 3
    assert result.reference_seeds == 2
    assert result.delta_pp <= result.parity_max_delta_pp
    assert len(result.per_seed_accuracies) == 3
