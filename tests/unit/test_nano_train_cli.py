"""CLI tests for qml-train."""

from unittest.mock import patch

from scripts.nano_train import main


def test_cli_json_output(monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")

    from src.application.dto import TrainNanomodelResult
    from src.shared.result import ok

    fake = TrainNanomodelResult(
        exp_id="nano_train",
        model_name="perceptron",
        dataset="breast_cancer",
        profile="ci",
        seed=42,
        accuracy=0.91,
        loss=0.2,
        elapsed_s=0.5,
        n_params=31,
        n_epochs=5,
    )

    with patch("scripts.nano_train.execute", return_value=ok(fake)):
        rc = main(
            [
                "--model",
                "perceptron",
                "--dataset",
                "breast_cancer",
                "--profile",
                "ci",
                "--json",
            ]
        )

    assert rc == 0


def test_cli_fail_exit_code(monkeypatch):
    from src.application.train_nanomodel import TrainNanomodelError
    from src.shared.result import fail

    with patch("scripts.nano_train.execute", return_value=fail(TrainNanomodelError("X", "bad"))):
        rc = main(["--model", "perceptron", "--dataset", "sequential_phase"])
    assert rc == 1
