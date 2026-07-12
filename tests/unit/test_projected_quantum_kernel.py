"""Unit tests for projected quantum kernel helpers."""

from __future__ import annotations

import numpy as np

from src.quantum.projected_quantum_kernel import (
    ProjectedQuantumFeatureEncoder,
    build_local_pauli_observables,
    fit_kernel_ridge_scores,
    fit_nystroem_logistic_proba,
)


def test_local_pauli_count():
    assert len(build_local_pauli_observables(4)) == 12


def test_projected_encoder_shapes():
    encoder = ProjectedQuantumFeatureEncoder(37, n_qubits=4, n_layers=1, seed=0)
    x = np.random.default_rng(0).normal(size=(4, 37)).astype(np.float64)
    phi = encoder.transform(x)
    assert phi.shape == (4, 12)
    assert np.isfinite(phi).all()
    assert phi.min() >= -1.0 - 1e-5
    assert phi.max() <= 1.0 + 1e-5


def test_kernel_ridge_and_nystroem_smoke():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 8)).astype(np.float64)
    y = (x[:, 0] > 0).astype(np.float64)
    encoder = ProjectedQuantumFeatureEncoder(8, n_qubits=2, n_layers=1, seed=1)
    phi = encoder.transform(x)
    scores = fit_kernel_ridge_scores(phi[:30], y[:30], phi[30:], alpha=1.0)
    assert scores.shape == (10,)
    proba = fit_nystroem_logistic_proba(
        phi[:30],
        y[:30],
        phi[30:],
        n_components=16,
        seed=1,
    )
    assert proba.shape == (10,)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
