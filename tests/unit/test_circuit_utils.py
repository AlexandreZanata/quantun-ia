"""Tests for PennyLane circuit utilities."""

from src.quantum.circuit_utils import PARAMETER_SHIFT_MIN_LAYERS, qnode_diff_method


def test_parameter_shift_for_deep_circuits():
    assert qnode_diff_method(PARAMETER_SHIFT_MIN_LAYERS, for_training=True) == "backprop"
    assert qnode_diff_method(PARAMETER_SHIFT_MIN_LAYERS, for_training=False) == "parameter-shift"
    assert qnode_diff_method(PARAMETER_SHIFT_MIN_LAYERS - 1, for_training=False) == "backprop"
