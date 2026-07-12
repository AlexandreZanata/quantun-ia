"""Unit tests for Pauli/shadow feature encoder shapes."""

from __future__ import annotations

import numpy as np

from src.quantum.shadow_features import PauliShadowFeatureEncoder, build_pauli_observables


def test_build_pauli_observables_count():
    obs = build_pauli_observables(4, 64)
    assert len(obs) == 64


def test_pauli_shadow_encoder_shapes():
    encoder = PauliShadowFeatureEncoder(37, n_qubits=4, n_layers=1, n_features=64, seed=0)
    x = np.random.default_rng(0).normal(size=(5, 37)).astype(np.float64)
    features = encoder.transform(x)
    assert features.shape == (5, 64)
    assert np.isfinite(features).all()
    assert features.min() >= -1.0 - 1e-5
    assert features.max() <= 1.0 + 1e-5


def test_pauli_shadow_encoder_deterministic():
    encoder = PauliShadowFeatureEncoder(8, n_qubits=2, n_layers=1, n_features=12, seed=7)
    x = np.ones((3, 8), dtype=np.float64)
    a = encoder.transform(x)
    b = encoder.transform(x)
    np.testing.assert_allclose(a, b)
