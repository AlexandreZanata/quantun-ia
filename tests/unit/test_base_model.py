"""Unit tests for TrainableMixin model interface contract."""

import torch

from src.classical.mlp import ClassicalNet
from src.classical.perceptron import Perceptron
from src.training.base_model import TrainableMixin


def test_trainable_mixin_methods_exist():
    model = Perceptron()
    assert hasattr(model, "train")
    assert hasattr(model, "predict")
    assert hasattr(model, "evaluate")
    assert isinstance(model, TrainableMixin)


def test_model_predict_and_evaluate(sample_binary_data, temp_log_file):
    X, y = sample_binary_data
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    model = ClassicalNet(hidden=4)
    model.train(X_t, y_t, exp_id="exp_test", model_name="mlp", epochs=2, lr=0.05)

    preds = model.predict(X_t)
    assert preds.shape == y_t.shape

    metrics = model.evaluate(X_t, y_t)
    assert "accuracy" in metrics
    assert "loss" in metrics
