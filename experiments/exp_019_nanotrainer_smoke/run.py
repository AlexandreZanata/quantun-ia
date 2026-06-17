"""
EXP 019 — Nano Trainer smoke: all registry models via train_nanomodel.execute.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.application.dto import TrainNanomodelDTO
from src.application.model_registry import list_models
from src.application.nanotrainer_config import load_nanotrainer_config
from src.application.train_nanomodel import execute
from src.shared.result import Ok
from src.training.structured_log import init_correlation_id, log_event

EXP_ID = "exp_019"
NANO_EXP_ID = "nano_train"
PROFILE = "ci"
ACC_MIN = 0.35
ACC_MAX = 1.0


def _dataset_for_model(model_name: str) -> str:
    cfg = load_nanotrainer_config()
    for pair in cfg.get("pairs", []):
        if pair["model"] == model_name:
            return pair["dataset"]
    raise ValueError(f"No dataset pair for {model_name}")


def main() -> None:
    init_correlation_id()
    log_event("info", "exp_019 nanotrainer smoke started", exp_id=EXP_ID)

    results: list[dict] = []
    for model_name in list_models():
        dataset = _dataset_for_model(model_name)
        dto = TrainNanomodelDTO(
            model_name=model_name,
            dataset=dataset,
            profile=PROFILE,
            exp_id=NANO_EXP_ID,
        )
        outcome = execute(dto)
        if not isinstance(outcome, Ok):
            raise RuntimeError(f"{model_name}: {outcome.error.code} — {outcome.error.message}")
        r = outcome.value
        if not (ACC_MIN <= r.accuracy <= ACC_MAX):
            raise RuntimeError(
                f"{model_name}: accuracy {r.accuracy:.3f} outside [{ACC_MIN}, {ACC_MAX}]"
            )
        results.append(
            {
                "model": model_name,
                "dataset": dataset,
                "accuracy": round(r.accuracy, 4),
                "elapsed_s": r.elapsed_s,
            }
        )
        print(f"  OK {model_name:25s} {dataset:18s} acc={r.accuracy * 100:.1f}%")

    log_event(
        "info",
        "exp_019 nanotrainer smoke finished",
        exp_id=EXP_ID,
        n_models=len(results),
        record_source="experiment",
    )
    print(f"\nexp_019 complete — {len(results)} models validated via Nano Trainer path")


if __name__ == "__main__":
    main()
