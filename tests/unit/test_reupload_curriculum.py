"""Unit tests for re-upload depth curriculum."""

import numpy as np
import torch

from src.quantum.qnn_reupload import QuantumNetReupload
from src.training.reupload_curriculum import layers_for_stage, train_fixed_reupload, train_reupload_curriculum


def test_layers_for_stage_ladder():
    assert layers_for_stage(0, 3, (1, 2, 3)) == 1
    assert layers_for_stage(2, 3, (1, 2, 3)) == 3


def test_quantum_reupload_set_n_layers_grows_weights():
    model = QuantumNetReupload(n_qubits=4, n_layers=1, input_dim=8)
    before = model.qlayer.state_dict()["weights"].clone()
    model.set_n_layers(3)
    assert model.n_layers == 3
    after = model.qlayer.state_dict()["weights"]
    assert after.shape[0] == 3
    assert torch.allclose(after[0], before[0])


def test_train_reupload_curriculum_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(60, 8)).astype(np.float32)
    y = (rng.random(60) > 0.5).astype(np.float32)
    x_h = torch.tensor(x[:20])
    y_h = torch.tensor(y[:20])
    model = QuantumNetReupload(n_qubits=4, n_layers=1, input_dim=8)
    score = train_reupload_curriculum(
        model,
        x,
        y,
        x_h,
        y_h,
        "exp_test",
        "curriculum_smoke",
        n_stages=3,
        epochs_per_stage=2,
        layer_ladder=(1, 2, 3),
        lr=0.05,
        seed=42,
    )
    assert 0.0 <= score <= 1.0
    fixed = train_fixed_reupload(
        lambda n: QuantumNetReupload(n_qubits=4, n_layers=n, input_dim=8),
        3,
        x,
        y,
        x_h,
        y_h,
        "exp_test",
        "fixed_smoke",
        n_stages=3,
        epochs_per_stage=2,
        lr=0.05,
        seed=42,
    )
    assert 0.0 <= fixed <= 1.0
