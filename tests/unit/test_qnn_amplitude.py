"""Integration test: amplitude QNN learns beyond chance on binary data."""

import torch

from src.quantum.qnn_amplitude import QuantumNetAmplitude


def test_amplitude_model_learns_after_training(sample_binary_data):
    X, y = sample_binary_data
    model = QuantumNetAmplitude(n_qubits=4, n_layers=1, input_dim=2)

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    before = model.evaluate(X_t, y_t)["accuracy"]

    model.train(X_t, y_t, exp_id="exp_test", model_name="amp_learn", epochs=20, lr=0.05)

    after = model.evaluate(X_t, y_t)["accuracy"]
    assert after > before
    assert after > 0.55


def test_amplitude_accepts_qml_device(sample_binary_data):
    X, y = sample_binary_data
    model = QuantumNetAmplitude(
        n_qubits=4,
        n_layers=1,
        input_dim=2,
        qml_device="default.qubit",
    )
    assert model.qml_device == "default.qubit"
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    metrics = model.evaluate(X_t, y_t)
    assert 0.0 <= metrics["accuracy"] <= 1.0
