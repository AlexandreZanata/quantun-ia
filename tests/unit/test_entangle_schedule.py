"""Unit tests for dynamic entanglement schedule."""

import torch

from src.quantum.qnn_entangled import QuantumNetEntangled
from src.training.entangle_schedule import (
    DEFAULT_ENTANGLEMENT_LADDER,
    entanglement_for_stage,
    train_entangled_schedule,
    train_fixed_entangled,
)


def test_entanglement_for_stage_ladder_endpoints():
    assert entanglement_for_stage(0, 5) == "none"
    assert entanglement_for_stage(4, 5) == "ring"


def test_entanglement_for_stage_monotonic():
    stages = [entanglement_for_stage(i, 5) for i in range(5)]
    ladder = list(DEFAULT_ENTANGLEMENT_LADDER)
    indices = [ladder.index(s) for s in stages]
    assert indices == sorted(indices)


def test_quantum_net_entangled_set_entanglement_preserves_weights():
    model = QuantumNetEntangled(n_qubits=4, n_layers=2, entanglement="none", input_dim=8, reupload=True)
    before = model.qlayer.state_dict()["weights"].clone()
    model.set_entanglement("ring")
    after = model.qlayer.state_dict()["weights"]
    assert model.entanglement == "ring"
    assert torch.allclose(before, after)


def test_train_entangled_schedule_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    x = torch.randn(40, 8)
    y = torch.randint(0, 2, (40,)).float()
    x_h = torch.randn(20, 8)
    y_h = torch.randint(0, 2, (20,)).float()

    def build(ent: str) -> QuantumNetEntangled:
        return QuantumNetEntangled(
            n_qubits=4,
            n_layers=1,
            entanglement=ent,
            input_dim=8,
            reupload=True,
        )

    acc = train_entangled_schedule(
        build,
        x,
        y,
        x_h,
        y_h,
        "exp_test",
        "schedule_smoke",
        n_stages=2,
        epochs_per_stage=2,
        lr=0.05,
        seed=42,
    )
    assert 0.0 <= acc <= 1.0

    fixed = train_fixed_entangled(
        build,
        "ring",
        x,
        y,
        x_h,
        y_h,
        "exp_test",
        "fixed_ring",
        n_stages=2,
        epochs_per_stage=2,
        lr=0.05,
        seed=42,
    )
    assert 0.0 <= fixed <= 1.0
