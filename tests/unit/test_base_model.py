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


def test_eval_propagates_to_dropout_layers():
    from src.classical.large_nano_mlp import LargeNanoMLP

    model = LargeNanoMLP(input_dim=28)
    model.eval()
    dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert dropout_layers
    assert all(not layer.training for layer in dropout_layers)
