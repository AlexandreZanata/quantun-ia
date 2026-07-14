"""Config presence for exp_108."""

from src.training.config import load_experiment_config


def test_exp_108_config_ci():
    cfg = load_experiment_config("exp_108_quantum_flow_coupling", profile="ci")
    assert cfg["exp_id"] == "exp_108"
    assert "min_relative_fid_improvement" in cfg
    assert int(cfg["depth"]) >= 2
    assert int(cfg["dim"]) >= 16

    pub = load_experiment_config("exp_108_quantum_flow_coupling", profile="publication")
    assert float(pub["min_relative_fid_improvement"]) == 0.05
