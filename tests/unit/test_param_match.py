"""Unit tests for parameter-matched baseline utilities."""

from src.classical.mlp import ClassicalNet
from src.quantum.qnn_basic import QuantumNetBasic
from src.training.param_match import (
    build_param_match_table,
    classical_n_params,
    nearest_classical_hidden,
)
from src.training.trainer import count_parameters


def test_nearest_classical_hidden_finds_close_match():
    target = count_parameters(QuantumNetBasic(n_qubits=4, n_layers=1, input_dim=2))
    hidden = nearest_classical_hidden(target)
    assert abs(classical_n_params(hidden) - target) <= 5


def test_build_param_match_table_includes_matched_hidden():
    n_params = count_parameters(ClassicalNet(hidden=8))
    records = [
        {
            "exp_id": "exp_test",
            "model_name": "classical_8_seed42",
            "n_params": n_params,
            "test_accuracy": 0.72,
            "eval_set": "holdout_test",
        }
    ]
    rows = build_param_match_table(records)
    assert len(rows) == 1
    assert rows[0]["n_params"] == n_params
    assert rows[0]["matched_classical_hidden"] >= 1
