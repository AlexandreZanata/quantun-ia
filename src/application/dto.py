"""DTOs for nano trainer use cases."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainNanomodelDTO:
    model_name: str
    dataset: str
    profile: str = "mini"
    epochs: int | None = None
    seed: int | None = None
    exp_id: str = "nano_train"
    test_size: float = 0.3
    save_checkpoints: bool = False


@dataclass(frozen=True)
class TrainNanomodelResult:
    exp_id: str
    model_name: str
    dataset: str
    profile: str
    seed: int
    accuracy: float
    loss: float
    elapsed_s: float
    n_params: int
    n_epochs: int
    checkpoint_path: str | None = None
    record_source: str = field(default="nanotrainer")


@dataclass(frozen=True)
class PredictNanomodelDTO:
    exp_id: str
    model_name: str
    dataset: str
    seed: int
    features: list[list[float]]


@dataclass(frozen=True)
class PredictNanomodelResult:
    exp_id: str
    model_name: str
    dataset: str
    seed: int
    probabilities: list[float]
    labels: list[int]
    checkpoint_path: str
    record_source: str = field(default="nanotrainer_predict")


@dataclass(frozen=True)
class NanoParityBenchDTO:
    """Run quantum nanomodel vs parameter-matched classical on one dataset."""

    quantum_model: str
    dataset: str
    profile: str = "ci"
    exp_id: str = "exp_022"
    seeds: list[int] | None = None
    epochs: int | None = None
    test_size: float = 0.3
    classical_learning_rate: float = 0.01
    save_checkpoints: bool = False


@dataclass(frozen=True)
class ParityPairMeta:
    quantum_model: str
    classical_label: str
    quantum_n_params: int
    classical_n_params: int
    classical_hidden: int
    param_delta: int
    quantum_learning_rate: float
    classical_learning_rate: float


@dataclass(frozen=True)
class NanoParityBenchResult:
    exp_id: str
    quantum_model: str
    dataset: str
    profile: str
    classical_label: str
    quantum_n_params: int
    classical_n_params: int
    classical_hidden: int
    param_delta: int
    quantum_accuracies: list[float]
    classical_accuracies: list[float]
    quantum_mean: float
    classical_mean: float
    quantum_summary: dict
    classical_summary: dict
    comparison: dict
    quantum_wins: bool
    verdict: str
    datasets_status: dict[str, str]
    record_source: str = field(default="nano_parity_bench")
