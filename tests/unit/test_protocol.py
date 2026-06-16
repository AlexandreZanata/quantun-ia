"""Unit tests for experiment protocol logging."""

from src.training.protocol import log_experiment_protocol, task_learnable


def test_task_learnable_above_threshold():
    assert task_learnable([0.7, 0.75, 0.8], threshold=0.55) is True


def test_task_learnable_below_threshold():
    assert task_learnable([0.48, 0.52, 0.50], threshold=0.55) is False


def test_log_experiment_protocol_includes_seeds():
    cfg = {
        "dataset": "circles",
        "n_samples": 500,
        "noise": 0.2,
        "test_size": 0.3,
        "epochs": 50,
        "learning_rate": 0.01,
        "seeds": [42, 123],
        "random_state": 42,
    }
    protocol = log_experiment_protocol("exp_test", cfg)
    assert protocol["n_seeds"] == 2
    assert protocol["dataset"] == "circles"
