"""Unit tests for exp_078 agro clinical cases configuration."""

from src.training.config import load_experiment_config


def test_exp_078_config():
    cfg = load_experiment_config("exp_078_agro_clinical_cases")
    assert cfg["exp_id"] == "exp_078"
    assert cfg["dataset_id"] == "acyd_soy_brazil_v1"
    assert cfg["model_exp_id"] == "exp_060"
    assert int(cfg["n_cases"]) == 8
    assert float(cfg["min_spearman_rho"]) == 0.85
