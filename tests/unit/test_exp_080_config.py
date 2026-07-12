"""Config tests for exp_080 quantum champion fusion on ACYD."""

from src.training.config import load_experiment_config


def test_exp_080_ci_config():
    cfg = load_experiment_config("exp_080_quantum_champion_fusion_acyd", profile="ci")
    assert cfg["dataset_id"] == "acyd_soy_brazil_v1"
    assert cfg["checkpoint_exp_id"] == "exp_060"
    assert cfg["depolarizing_p"] == 0.03
    assert cfg["min_vs_classical_pp"] == -10.0
    assert cfg["min_vs_best_hybrid_pp"] == -10.0
    assert cfg["n_train_rows"] == 5000


def test_exp_080_publication_config():
    cfg = load_experiment_config("exp_080_quantum_champion_fusion_acyd", profile="publication")
    assert cfg["total_head_epochs"] == 8
    assert cfg["min_vs_classical_pp"] == -1.0
    assert cfg["min_vs_best_hybrid_pp"] == 0.5
    assert float(cfg["best_hybrid_ref_auc"]) == 0.6771
    assert cfg["save_checkpoints"] is True
