"""Unit tests for exp_102 config."""

from src.training.config import load_experiment_config


def test_exp_102_ci_config():
    cfg = load_experiment_config("exp_102_nano_unet_cifar_i2i", profile="ci")
    assert cfg["exp_id"] == "exp_102"
    assert cfg["n_train"] == 32
    assert cfg["epochs"] == 2
    assert cfg["base_channels"] == 16
