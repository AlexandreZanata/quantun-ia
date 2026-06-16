"""Tests for Holm-Bonferroni and batch comparisons."""

from src.training.holdout import compare_conditions_batch
from src.training.statistics import holm_bonferroni


def test_holm_bonferroni_monotonic_adjustment():
    raw = [0.01, 0.04, 0.20]
    adjusted = holm_bonferroni(raw, alpha=0.05)
    assert adjusted[0]["p_adjusted"] <= adjusted[1]["p_adjusted"] <= adjusted[2]["p_adjusted"]
    assert adjusted[0]["significant_holm"] is True


def test_compare_conditions_batch_applies_holm(tmp_path, monkeypatch):
    import src.training.metrics as metrics_module

    log_file = tmp_path / "experiments.jsonl"
    monkeypatch.setattr(metrics_module, "LOGS_PATH", log_file)

    a = [0.9, 0.91, 0.89, 0.92, 0.88, 0.9, 0.91, 0.89]
    b = [0.5, 0.51, 0.49, 0.5, 0.52, 0.48, 0.5, 0.51]
    results = compare_conditions_batch(
        "exp_test",
        [{"label_a": "a", "label_b": "b", "condition_a": a, "condition_b": b}],
    )
    assert results[0]["p_value_holm"] is not None
    assert "paired_comparison_batch" in log_file.read_text()
