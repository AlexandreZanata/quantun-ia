"""Smoke tests for model forward passes."""

import torch

from src.classical.mlp import ClassicalNet
from src.classical.rnn_mini import RNNMini
from src.classical.transformer_mini import TransformerMini
from src.data.augmentation import add_gaussian_noise, random_flip_labels
from src.data.generators import make_binary_classification
from src.quantum.hybrid_model import ClassicalFirst, HybridSandwich, QuantumFirst
from src.quantum.qnn_basic import QuantumNetBasic
from src.quantum.qnn_entangled import QuantumNetEntangled
from src.training.trainer import evaluate, fine_tune, predict


def test_classical_models_forward(sample_binary_data):
    X, y = sample_binary_data
    X_t = torch.tensor(X)

    for model in [ClassicalNet(hidden=8), RNNMini(), TransformerMini()]:
        out = model(X_t)
        assert out.shape == (len(X),)


def test_quantum_models_forward(sample_binary_data):
    X, y = sample_binary_data
    X_t = torch.tensor(X)

    models = [
        QuantumNetBasic(n_qubits=4, n_layers=2, input_dim=2),
        QuantumNetEntangled(n_qubits=4, n_layers=2, entanglement="chain"),
        HybridSandwich(input_dim=2, n_qubits=4, n_layers=2),
        QuantumFirst(input_dim=2, n_qubits=4, n_layers=2),
        ClassicalFirst(input_dim=2, n_qubits=4, n_layers=2),
    ]
    for model in models:
        out = model(X_t)
        assert out.shape == (len(X),)


def test_data_augmentation(sample_binary_data):
    X, y = sample_binary_data
    X_aug = add_gaussian_noise(X, sigma=0.05)
    y_flip = random_flip_labels(y, flip_rate=0.1)
    assert X_aug.shape == X.shape
    assert y_flip.shape == y.shape


def test_generators_circles():
    X, y, _ = make_binary_classification(dataset="circles")
    assert X.shape[1] == 2
    assert set(y.tolist()) <= {0.0, 1.0}


def test_trainer_helpers(sample_binary_data):
    X, y = sample_binary_data
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    model = ClassicalNet(hidden=4)

    preds = predict(model, X_t)
    assert preds.shape == y_t.shape

    metrics = evaluate(model, X_t, y_t)
    assert "accuracy" in metrics

    acc = fine_tune(model, X_t, y_t, epochs=1)
    assert 0.0 <= acc <= 1.0
