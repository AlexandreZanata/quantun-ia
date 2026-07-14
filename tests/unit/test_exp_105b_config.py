"""Unit tests for exp_105b config."""

from src.training.config import load_experiment_config


def test_exp_105b_ci_config():
    cfg = load_experiment_config("exp_105b_gv_alr_image_ddpm", profile="ci")
    assert cfg["exp_id"] == "exp_105b"
    assert cfg["adaptive_epochs"] <= int(cfg["fixed_epochs"] * cfg["max_epoch_fraction"] + 1e-9)
    assert "adaptive_lr" in cfg
