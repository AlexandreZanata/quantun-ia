"""Unit tests for exp_067 re-upload ladder ACYD configuration."""

from src.training.config import load_experiment_config


def test_exp_067_ci_profile():
    cfg = load_experiment_config("exp_067_reupload_ladder_acyd", profile="ci")
    assert cfg["exp_id"] == "exp_067"
    assert cfg["dataset_id"] == "acyd_soy_brazil_v1"
    assert cfg["min_wins"] == 0
    assert cfg["min_rung_advantage_pp"] == -10.0
    assert cfg["epochs_per_stage"] == 3
    rungs = cfg["rungs"]
    assert len(rungs) == 3
    assert rungs[0]["id"] == "temp_only"
    assert rungs[0]["feature_slice"] == [13, 21]
    assert rungs[0]["metric"] == "roc_auc"
    assert rungs[1]["feature_slice"] == [9, 21]
    assert rungs[2]["feature_slice"] == [0, 37]
    assert rungs[0]["n_train_rows"] == 2000


def test_exp_067_publication_profile():
    cfg = load_experiment_config("exp_067_reupload_ladder_acyd", profile="publication")
    assert cfg["min_wins"] == 2
    assert cfg["min_rung_advantage_pp"] == 0.3
    assert cfg["epochs_per_stage"] == 8
    assert cfg["save_checkpoints"] is True
    assert len(cfg["rungs"]) == 3
    assert cfg["rungs"][0]["n_train_rows"] == 20000
