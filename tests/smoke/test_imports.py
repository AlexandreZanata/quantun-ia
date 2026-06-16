"""Smoke tests — verify all core modules import correctly."""

import importlib

MODULES = [
    "src.training.metrics",
    "src.training.trainer",
    "src.training.base_model",
    "src.training.config",
    "src.training.structured_log",
    "src.training.curriculum",
    "src.data.generators",
    "src.data.poisoning",
    "src.data.augmentation",
    "src.quantum.amplitude_encoding",
    "src.quantum.qnn_basic",
    "src.quantum.qnn_entangled",
    "src.quantum.hybrid_model",
    "src.classical.perceptron",
    "src.classical.mlp",
    "src.classical.transformer_mini",
    "src.classical.rnn_mini",
]


def test_all_modules_import():
    for module in MODULES:
        importlib.import_module(module)
