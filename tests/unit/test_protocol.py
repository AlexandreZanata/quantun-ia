"""Unit tests for experiment protocol logging."""

from src.training.protocol import log_applicability_gate, log_experiment_protocol, task_learnable


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


def test_log_applicability_gate_writes_jsonl(tmp_path, monkeypatch):
    import src.training.metrics as metrics_module

    log_file = tmp_path / "experiments.jsonl"
    monkeypatch.setattr(metrics_module, "LOGS_PATH", log_file)
    gate = log_applicability_gate(
        "exp_005",
        "curriculum",
        False,
        threshold=0.55,
        mean_holdout=0.52,
    )
    assert gate["status"] == "not_applicable"
    content = log_file.read_text()
    assert "applicability_gate" in content
    assert "not_applicable" in content
