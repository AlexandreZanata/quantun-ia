"""Unit tests for training device resolution."""

import torch

from src.training.device import resolve_device


def test_resolve_device_cpu_explicit(monkeypatch):
    monkeypatch.delenv("QML_DEVICE", raising=False)
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_returns_cpu_without_cuda(monkeypatch):
    monkeypatch.setenv("QML_DEVICE", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device().type == "cpu"
