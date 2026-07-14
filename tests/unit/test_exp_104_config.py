"""Unit tests for exp_104 config."""

from src.training.config import load_experiment_config


def test_exp_104_ci_config():
    cfg = load_experiment_config("exp_104_distill_image_nano", profile="ci")
    assert cfg["exp_id"] == "exp_104"
    assert cfg["n_train"] == 32
    assert cfg["teacher_base_channels"] >= cfg["student_base_channels"]
    assert 0.0 <= cfg["distill_alpha"] <= 1.0
