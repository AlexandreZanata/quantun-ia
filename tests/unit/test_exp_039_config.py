"""Unit tests for exp_039 regularized Synthea configuration."""

from src.training.config import load_experiment_config


def test_exp_039_ci_profile():
    cfg = load_experiment_config("exp_039_synthea_regularized", profile="ci")
    assert cfg["exp_id"] == "exp_039"
    assert cfg["dropout"] == 0.5
    assert cfg["weight_decay"] == 0.001
    assert cfg["n_train_rows"] == 50000


def test_exp_039_publication_profile():
    cfg = load_experiment_config("exp_039_synthea_regularized", profile="publication")
    assert cfg["epochs"] == 15
    assert cfg["save_checkpoints"] is True
    assert cfg["reference_nano_val_auc"] == 0.7867
