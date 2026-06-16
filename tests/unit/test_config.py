"""Unit tests for experiment config loader."""

from src.training.config import load_config, load_experiment_config


def test_load_config_has_defaults():
    cfg = load_config()
    assert "defaults" in cfg
    assert cfg["defaults"]["epochs"] == 50
    assert cfg["defaults"]["dataset"] == "circles"


def test_load_experiment_config_merges_defaults():
    cfg = load_experiment_config("exp_004_data_poisoning")
    assert cfg["epochs"] == 50
    assert cfg["poison_rates"] == [0.0, 0.05, 0.1, 0.2, 0.3]


def test_publication_profile_defaults():
    cfg = load_config()["profiles"]["publication"]
    assert cfg["n_samples"] == 500
    assert len(cfg["seeds"]) == 10


def test_publication_large_profile():
    cfg = load_experiment_config("exp_001_quantum_vs_classical", profile="publication_large")
    assert cfg["n_samples"] == 1000


def test_publication_large_from_env(monkeypatch):
    monkeypatch.setenv("QML_PROFILE", "publication_large")
    cfg = load_experiment_config("exp_001_quantum_vs_classical")
    assert cfg["n_samples"] == 1000
    assert cfg["profile"] == "publication_large"


def test_exp_009_and_010_config():
    cfg9 = load_experiment_config("exp_009_entanglement_basic")
    assert cfg9["qnn_type"] == "basic"
    cfg10 = load_experiment_config("exp_010_poison_reupload_ablation")
    assert "reupload_2l" in cfg10["reupload_variants"]
