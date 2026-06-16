"""Smoke test for data re-uploading QNN."""

import torch

from src.quantum.qnn_reupload import QuantumNetReupload


def test_reupload_qnn_learns_synthetic_xor():
    """Re-upload QNN should exceed chance on a tiny separable task."""
    model = QuantumNetReupload(n_qubits=4, n_layers=2, input_dim=2)
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 8)
    y = torch.tensor([0.0, 1.0, 1.0, 0.0] * 8)
    model.train(X, y, exp_id="smoke", model_name="reupload_xor", epochs=80, lr=0.05)
    acc = model.evaluate(X, y)["accuracy"]
    assert acc > 0.55
