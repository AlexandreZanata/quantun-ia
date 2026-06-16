"""
EXP 006 — Barren Plateau Explorer
Gradient variance with bootstrap 95% CI across random initializations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time

from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.gradients import measure_gradient_variance
from src.training.metrics import ExperimentLogger
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_006_barren_plateau"
EXP_ID = "exp_006"


def _model_factory(n_qubits: int):
    return QuantumNetBasic(n_qubits=n_qubits, n_layers=2, input_dim=2)


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID)

    t0 = time.time()
    variance_stats = measure_gradient_variance(
        _model_factory,
        cfg["n_qubits_list"],
        n_samples=cfg.get("grad_samples", 50),
        input_dim=2,
        batch_size=1,
        use_parameter_shift=True,
    )

    log = ExperimentLogger(EXP_ID, "barren_plateau_scan")
    flat_variances = {}
    for n_q, stats in variance_stats.items():
        log.log(
            n_q,
            grad_variance=stats["mean"],
            grad_variance_ci_low=stats["ci_low"],
            grad_variance_ci_high=stats["ci_high"],
        )
        flat_variances[n_q] = stats["mean"]
        log_event(
            "info",
            "gradient variance measured",
            exp_id=EXP_ID,
            n_qubits=n_q,
            variance=stats["mean"],
            ci_low=stats["ci_low"],
            ci_high=stats["ci_high"],
            n_samples=stats["n_samples"],
        )
    log.finish(time.time() - t0, gradient_variances=flat_variances, gradient_stats=variance_stats)

    log_event("info", "experiment run finished", exp_id=EXP_ID)
