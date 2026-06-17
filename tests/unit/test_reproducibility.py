"""Unit tests for global seed reproducibility."""

import torch

from src.classical.mlp import ClassicalNet
from src.training.reproducibility import set_global_seed
from src.training.trainer import train_model


def _first_epoch_loss(X_train, y_train, seed: int) -> float:
    set_global_seed(seed)
    model = ClassicalNet(hidden=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = torch.nn.BCELoss()
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()


def test_same_seed_same_first_epoch_loss(sample_binary_data):
    X, y = sample_binary_data
    split = int(len(X) * 0.7)
    X_train = torch.tensor(X[:split], dtype=torch.float32)
    y_train = torch.tensor(y[:split], dtype=torch.float32)

    loss_a = _first_epoch_loss(X_train, y_train, seed=42)
    loss_b = _first_epoch_loss(X_train, y_train, seed=42)
    assert loss_a == loss_b


def test_train_model_records_seed_in_jsonl(temp_log_file, sample_binary_data):
    import json

    X, y = sample_binary_data
    split = int(len(X) * 0.7)
    X_train = torch.tensor(X[:split], dtype=torch.float32)
    y_train = torch.tensor(y[:split], dtype=torch.float32)

    model = ClassicalNet(hidden=8)
    train_model(model, X_train, y_train, "exp_test", "seeded", epochs=2, lr=0.05, seed=99)

    record = json.loads(temp_log_file.read_text().strip().splitlines()[-1])
    assert record["seed"] == 99
