"""Unit tests for exp_040 full-scale ablation configuration."""

from src.training.config import load_experiment_config


def test_exp_040_full_scale_profile():
    cfg = load_experiment_config("exp_040_full_scale_ablation_higgs", profile="full_scale")
    assert cfg["exp_id"] == "exp_040"
    assert cfg["n_train_rows"] == 0
    assert cfg["n_val_rows"] == 0
    assert cfg["seeds"] == [42, 123, 456]
    assert cfg["batch_size"] == 2048
    assert cfg["baseline_epochs"] == 12
