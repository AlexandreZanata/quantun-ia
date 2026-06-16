"""Unit tests for ExperimentLogger."""

import json

from src.training.metrics import ExperimentLogger, load_all_experiments


def test_logger_writes_valid_json_line(temp_log_file):
    logger = ExperimentLogger("exp_test", "test_model")
    logger.log(0, loss=0.5, accuracy=0.75)
    logger.log(1, loss=0.3, accuracy=0.85)
    record = logger.finish(elapsed_seconds=1.5)

    assert record["exp_id"] == "exp_test"
    assert record["model_name"] == "test_model"
    assert record["final_acc"] == 0.85
    assert record["n_epochs"] == 2

    lines = temp_log_file.read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["exp_id"] == "exp_test"


def test_load_all_experiments(temp_log_file):
    logger = ExperimentLogger("exp_test", "model_a")
    logger.log(0, accuracy=0.9)
    logger.finish(1.0)

    records = load_all_experiments()
    assert len(records) == 1
    assert records[0]["model_name"] == "model_a"
