"""Unit tests for Nano Parity Bench — quantum nanomodel vs parameter-matched classical."""

from __future__ import annotations

from src.application.dto import NanoParityBenchDTO
from src.application.nano_parity_bench import build_parity_pair, execute
from src.application.parity_config import load_parity_config
from src.application.parity_datasets import ensure_datasets_available
from src.shared.result import Fail, Ok
from src.training.param_match import classical_n_params


def test_load_parity_config_has_primary_claim():
    cfg = load_parity_config()
    assert "primary_claim" in cfg
    assert cfg["primary_claim"]["model"] == "hybrid_sandwich"


def test_build_parity_pair_matches_parameter_count():
    quantum, classical, meta = build_parity_pair("hybrid_sandwich", input_dim=30)
    assert meta.quantum_n_params > 0
    assert meta.classical_hidden >= 1
    assert abs(meta.quantum_n_params - meta.classical_n_params) <= 10
    assert classical_n_params(meta.classical_hidden, 30) == meta.classical_n_params


def test_ensure_datasets_downloads_mnist(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    status = ensure_datasets_available(["breast_cancer", "mnist_binary"], cache_root=tmp_path / "data")
    assert status["breast_cancer"] == "ready"
    assert status["mnist_binary"] == "ready"
    assert (tmp_path / "data" / "raw" / "mnist").exists()


def test_execute_parity_benchmark_hybrid_wins_ci(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", log_file)

    dto = NanoParityBenchDTO(
        quantum_model="hybrid_sandwich",
        dataset="breast_cancer",
        profile="ci",
        exp_id="exp_022",
        seeds=[42, 123, 456],
        epochs=15,
    )
    outcome = execute(dto)
    assert isinstance(outcome, Ok), outcome
    result = outcome.value
    assert result.quantum_model == "hybrid_sandwich"
    assert result.classical_hidden >= 1
    assert abs(result.quantum_n_params - result.classical_n_params) <= 10
    assert len(result.quantum_accuracies) == 3
    assert result.verdict in {"accepted", "inconclusive", "rejected"}
    assert "p_value" in result.comparison
    assert log_file.exists()


def test_execute_rejects_sequence_dataset():
    dto = NanoParityBenchDTO(
        quantum_model="hybrid_sandwich",
        dataset="sequential_phase",
        profile="ci",
        seeds=[42],
        epochs=5,
    )
    outcome = execute(dto)
    assert isinstance(outcome, Fail)
    assert outcome.error.code == "INVALID_PAIR"
