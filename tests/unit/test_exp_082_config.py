"""Config tests for exp_082 ACYD calibration."""

from src.training.config import load_experiment_config


def test_exp_082_ci_config():
    cfg = load_experiment_config("exp_082_calibration_acyd", profile="ci")
    assert cfg["dataset_id"] == "acyd_soy_brazil_v1"
    assert cfg["model_exp_id"] == "exp_060"
    assert cfg["n_rows"] == 800
    assert cfg["max_ece_after"] == 0.15
    assert cfg["force_balanced"] is False


def test_exp_082_publication_config():
    cfg = load_experiment_config("exp_082_calibration_acyd", profile="publication")
    assert cfg["n_rows"] == 0
    assert cfg["max_ece_after"] == 0.08
    assert cfg["min_spearman_rho"] == 0.85
    assert cfg["require_ece_improved"] is True
    assert float(cfg["min_auc_delta"]) == -0.005
