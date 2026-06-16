"""
EXP 006 — Barren Plateau Explorer
Measures mean gradient variance across random initializations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time

from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.gradients import measure_gradient_variance
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_006_barren_plateau"
EXP_ID = "exp_006"


def _model_factory(n_qubits: int):
    return QuantumNetBasic(n_qubits=n_qubits, n_layers=2, input_dim=2)


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    log_event("info", "experiment run started", exp_id=EXP_ID)

    t0 = time.time()
    variances = measure_gradient_variance(
        _model_factory,
        cfg["n_qubits_list"],
        n_samples=cfg.get("grad_samples", 20),
        input_dim=2,
    )

    log = ExperimentLogger(EXP_ID, "barren_plateau_scan")
    for n_q, variance in variances.items():
        log.log(n_q, grad_variance=variance)
        log_event("info", "gradient variance measured", exp_id=EXP_ID, n_qubits=n_q, variance=variance)
    log.finish(time.time() - t0, gradient_variances=variances)

    log_event("info", "experiment run finished", exp_id=EXP_ID)
