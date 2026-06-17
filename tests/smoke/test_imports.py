"""Smoke tests — verify all core modules import correctly."""

import importlib

MODULES = [
    "src.training.metrics",
    "src.training.trainer",
    "src.training.base_model",
    "src.training.config",
    "src.training.structured_log",
    "src.training.curriculum",
    "src.training.gradients",
    "src.training.param_match",
    "src.training.reproducibility",
    "src.training.tracking",
    "src.training.checkpoints",
    "src.training.ci_smoke",
    "src.training.device",
    "src.training.hpo",
    "src.data.generators",
    "src.data.poisoning",
    "src.data.scaling",
    "src.data.real_datasets",
    "src.data.dataset_registry",
    "src.quantum.qnn_entangled",
    "src.quantum.qnn_reupload",
    "src.quantum.qnn_factory",
    "src.quantum.circuit_utils",
    "src.data.splits",
    "src.training.holdout",
    "src.training.statistics",
    "src.training.protocol",
    "src.training.self_play",
    "src.data.augmentation",
    "src.quantum.amplitude_encoding",
    "src.quantum.qnn_basic",
    "src.quantum.qnn_amplitude",
    "src.quantum.hybrid_model",
    "src.classical.perceptron",
    "src.classical.mlp",
    "src.classical.transformer_mini",
    "src.classical.rnn_mini",
]


def test_all_modules_import():
    for module in MODULES:
        importlib.import_module(module)
