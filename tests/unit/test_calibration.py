"""Unit tests for probability calibration helpers."""

from __future__ import annotations

import numpy as np

from src.application.calibration import (
    CalibrationArtifact,
    apply_isotonic,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
)


def test_isotonic_preserves_rank_order():
    y_true = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    raw = [0.05, 0.15, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.98]
    artifact = fit_isotonic_calibrator(raw, y_true)
    calibrated = apply_isotonic(raw, artifact)
    assert all(calibrated[i] <= calibrated[i + 1] + 1e-9 for i in range(len(calibrated) - 1))


def test_platt_returns_artifact_with_method():
    y_true = [0, 0, 1, 1, 1]
    raw = [0.2, 0.3, 0.7, 0.8, 0.9]
    artifact = fit_platt_calibrator(raw, y_true)
    assert artifact.method == "platt"
    calibrated = artifact.transform(raw)
    assert len(calibrated) == len(raw)
    assert all(0.0 <= p <= 1.0 for p in calibrated)


def test_calibration_artifact_roundtrip():
    y_true = [0, 0, 1, 1]
    raw = [0.1, 0.4, 0.6, 0.9]
    artifact = fit_isotonic_calibrator(raw, y_true)
    payload = artifact.to_dict()
    restored = CalibrationArtifact.from_dict(payload)
    assert restored.method == "isotonic"
    np.testing.assert_allclose(
        apply_isotonic(raw, restored),
        apply_isotonic(raw, artifact),
        rtol=1e-6,
    )
