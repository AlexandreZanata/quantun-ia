"""
EXP 006 — Barren Plateau Explorer
Write your hypothesis in hypothesis.md BEFORE running this script.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time

import numpy as np
import torch

from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_006_barren_plateau"
EXP_ID = "exp_006"


def measure_gradient_variance(n_qubits_list, n_samples=20):
    variances = {}
    for n_q in n_qubits_list:
        grads = []
        for _ in range(n_samples):
            model = QuantumNetBasic(n_qubits=n_q, n_layers=2, input_dim=n_q)
            X_dummy = torch.randn(10, n_q, requires_grad=False)
            y_dummy = torch.randint(0, 2, (10,)).float()

            model.training = True
            model.zero_grad()
            pred = model(X_dummy)
            loss = torch.nn.functional.binary_cross_entropy(pred, y_dummy)
            loss.backward()

            grad_vars = [p.grad.var().item() for p in model.parameters() if p.grad is not None]
            grads.append(np.mean(grad_vars) if grad_vars else 0.0)

        variances[n_q] = float(np.mean(grads))
        log_event("info", "gradient variance measured", exp_id=EXP_ID, n_qubits=n_q, variance=variances[n_q])

    return variances


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    log_event("info", "experiment run started", exp_id=EXP_ID)

    t0 = time.time()
    variances = measure_gradient_variance(cfg["n_qubits_list"])

    log = ExperimentLogger(EXP_ID, "barren_plateau_scan")
    for n_q, variance in variances.items():
        log.log(n_q, grad_variance=variance)
    log.finish(time.time() - t0, gradient_variances=variances)

    log_event("info", "experiment run finished", exp_id=EXP_ID)
