"""
EXP 022 — Nano Quantum Parity Bench
Prove hybrid_sandwich quantum nanomodel vs parameter-matched classical on UCI tabular.
Uses qml-bench-parity application (downloads datasets + builds matched baseline).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.application.dto import NanoParityBenchDTO
from src.application.nano_parity_bench import execute
from src.application.parity_config import load_parity_config, profile_settings
from src.shared.result import Fail, Ok
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_022_nano_quantum_parity"
EXP_ID = "exp_022"


def _primary_runs(cfg: dict, profile: str) -> list[NanoParityBenchDTO]:
    primary = cfg["primary_claim"]
    prof = profile_settings(cfg, profile)
    seeds = prof.get("seeds")
    epochs = prof.get("epochs")
    return [
        NanoParityBenchDTO(
            quantum_model=primary["model"],
            dataset=dataset,
            profile=profile,
            exp_id=EXP_ID,
            seeds=list(seeds) if seeds else None,
            epochs=int(epochs) if epochs else None,
            save_checkpoints=True,
        )
        for dataset in primary["datasets"]
    ]


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_parity_config()
    profile = cfg.get("defaults", {}).get("profile", "ci")
    log_event("info", "experiment run started", exp_id=EXP_ID, profile=profile)

    outcomes = []
    for dto in _primary_runs(cfg, profile):
        result = execute(dto)
        if isinstance(result, Fail):
            raise SystemExit(f"{dto.dataset}: {result.error.message}")
        assert isinstance(result, Ok)
        outcomes.append(result.value)
        print(
            f"{dto.dataset}: quantum={result.value.quantum_mean:.3f} "
            f"classical={result.value.classical_mean:.3f} "
            f"verdict={result.value.verdict}"
        )

    wins = sum(1 for o in outcomes if o.quantum_wins)
    log_event(
        "info",
        "experiment run finished",
        exp_id=EXP_ID,
        datasets_tested=len(outcomes),
        quantum_wins=wins,
    )
    print(f"Done — quantum wins on {wins}/{len(outcomes)} primary datasets.")
