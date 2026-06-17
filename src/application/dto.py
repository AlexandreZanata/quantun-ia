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
    record_source: str = field(default="nanotrainer")
