"""Config presence for exp_103."""

from src.training.config import load_experiment_config


def test_exp_103_config_ci_and_publication():
    cfg = load_experiment_config("exp_103_tiny_dit_flickr_t2i", profile="ci")
    assert cfg["exp_id"] == "exp_103"
    assert int(cfg["text_dim"]) >= 16

    pub = load_experiment_config("exp_103_tiny_dit_flickr_t2i", profile="publication")
    assert float(pub["min_clip_null_gap"]) == 0.5
