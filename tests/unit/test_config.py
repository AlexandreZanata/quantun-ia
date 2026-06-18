"""Unit tests for experiment config loader."""

from src.training.config import load_config, load_experiment_config


def test_load_config_has_defaults():
    cfg = load_config()
    assert "defaults" in cfg
    assert cfg["defaults"]["epochs"] == 50
    assert cfg["defaults"]["dataset"] == "circles"


def test_load_experiment_config_merges_defaults(monkeypatch):
    monkeypatch.delenv("QML_PROFILE", raising=False)
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


def test_ci_profile_fast_settings():
    cfg = load_experiment_config("exp_001_quantum_vs_classical", profile="ci")
    assert cfg["n_samples"] == 50
    assert cfg["seeds"] == [42, 123]
    assert cfg["epochs"] == 5
    assert cfg["profile"] == "ci"


def test_exp_009_and_010_config():
    cfg9 = load_experiment_config("exp_009_entanglement_basic")
    assert cfg9["qnn_type"] == "basic"
    cfg10 = load_experiment_config("exp_010_poison_reupload_ablation")
    assert "reupload_2l" in cfg10["reupload_variants"]


def test_exp_011_through_014_config(monkeypatch):
    monkeypatch.delenv("QML_PROFILE", raising=False)
    cfg11 = load_experiment_config("exp_011_uci_tabular_qml")
    assert cfg11["dataset"] == "breast_cancer"
    cfg12 = load_experiment_config("exp_012_mnist_pca_qml")
    assert cfg12["n_components"] == 8
    cfg13 = load_experiment_config("exp_013_augmentation_robustness")
    assert cfg13["augment_sigma"] == 0.15
    cfg14 = load_experiment_config("exp_014_sequence_baselines")
    assert cfg14["seq_len"] == 8
    cfg15 = load_experiment_config("exp_015_adaptive_qnn")
    assert cfg15["adaptive_lr"]["var_target"] == 0.015
    assert "quantum_6q_3l_adaptive" in cfg15["models"]
    cfg16 = load_experiment_config("exp_016_hybrid_nas")
    assert cfg16["exp_id"] == "exp_016"
    assert cfg16["hpo_trials"] == 20
    cfg16_ci = load_experiment_config("exp_016_hybrid_nas", profile="ci")
    assert cfg16_ci["hpo_trials"] == 3
    cfg17 = load_experiment_config("exp_017_poison_topology")
    assert cfg17["exp_id"] == "exp_017"
    assert "nas_preset" in cfg17["topologies"]
    assert 0.3 in cfg17["poison_rates"]
    cfg18 = load_experiment_config("exp_018_feature_fusion")
    assert cfg18["dataset"] == "sequential_phase"
    assert "transformer_qnn_fusion" in cfg18["models"]


def test_exp_019_and_020_infrastructure_config(monkeypatch):
    monkeypatch.delenv("QML_PROFILE", raising=False)
    cfg19 = load_experiment_config("exp_019_nanotrainer_smoke")
    assert cfg19["infrastructure"] is True
    assert cfg19["exp_id"] == "exp_019"
    cfg20 = load_experiment_config("exp_020_api_smoke")
    assert cfg20["infrastructure"] is True
    assert cfg20["api_model"] == "perceptron"


def test_exp_024_profile_overrides(monkeypatch):
    monkeypatch.delenv("QML_PROFILE", raising=False)
    cfg_ci = load_experiment_config("exp_024_quantum_nano_bc", profile="ci")
    assert cfg_ci["seeds"] == [42, 123]
    assert cfg_ci["epochs"] == 15
    cfg_pub = load_experiment_config("exp_024_quantum_nano_bc", profile="publication")
    assert len(cfg_pub["seeds"]) == 30
    assert cfg_pub["save_checkpoints"] is True
