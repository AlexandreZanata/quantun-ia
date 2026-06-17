"""Unit tests for MLflow tracking wrapper."""


from src.training.tracking import RunTracker, mlflow_enabled


def test_mlflow_disabled_by_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    assert mlflow_enabled() is False
    tracker = RunTracker("exp_test", "model_a", seed=42, profile="ci")
    tracker.log_params({"lr": 0.01})
    tracker.log_metrics({"loss": 0.5}, step=0)
    tracker.end()


def test_tracker_noop_without_crash(monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    tracker = RunTracker("exp_test", "model_b")
    tracker.log_params({"epochs": 5})
    tracker.end()
