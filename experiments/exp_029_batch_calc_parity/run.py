"""
EXP 029 — Batch calculation vs REST API probability parity (569 rows).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_029_batch_calc_parity/run.py --profile ci
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient

from src.application.batch_predict import (
    BatchPredictDTO,
    load_input_rows,
    predict_request_payload_all_rows,
    run_batch_predict,
)
from src.application.dto import TrainNanomodelDTO
from src.application.train_nanomodel import execute as train_execute
from src.presentation.http.app import create_app
from src.shared.result import Ok
from src.training.config import load_experiment_config

EXP_KEY = "exp_029_batch_calc_parity"
MAX_DELTA = 1e-5
FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "breast_cancer_holdout.csv"


@dataclass(frozen=True)
class BatchParityResult:
    n_rows: int
    max_delta: float
    mean_delta: float
    batch_elapsed_s: float
    api_elapsed_s: float


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _ensure_checkpoint(*, exp_id: str, seed: int, profile: str, epochs: int) -> None:
    dto = TrainNanomodelDTO(
        model_name="hybrid_sandwich",
        dataset="breast_cancer",
        profile=profile,
        epochs=epochs,
        seed=seed,
        exp_id=exp_id,
        save_checkpoints=True,
    )
    outcome = train_execute(dto)
    if not isinstance(outcome, Ok):
        raise RuntimeError(f"checkpoint bootstrap failed: {outcome.error.message}")


def run_exp_029(
    *,
    profile: str = "ci",
    input_path: Path | None = None,
    verbose: bool = True,
    require_cuda: bool = True,
    bootstrap_checkpoint: bool = True,
) -> BatchParityResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    epochs = int(cfg["epochs"])
    checkpoint_exp_id = str(cfg.get("checkpoint_exp_id", "quantum_nano_bc_app"))
    n_rows_limit = int(cfg.get("n_rows", 569))
    chunk_size = int(cfg.get("chunk_size", 64))

    csv_path = input_path or FIXTURE
    rows, _ = load_input_rows(csv_path)
    if n_rows_limit < len(rows):
        rows = rows[:n_rows_limit]

    if bootstrap_checkpoint:
        if verbose:
            print(f"Bootstrapping checkpoint exp_id={checkpoint_exp_id} seed={seed}...", flush=True)
        _ensure_checkpoint(exp_id=checkpoint_exp_id, seed=seed, profile=profile, epochs=epochs)

    dto = BatchPredictDTO(
        features=rows,
        exp_id=checkpoint_exp_id,
        model_name="hybrid_sandwich",
        dataset="breast_cancer",
        seed=seed,
        chunk_size=chunk_size,
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXP 029 — Batch vs API | profile={profile} | rows={len(rows)}")
        print(f"Checkpoint: {checkpoint_exp_id} seed={seed} | max_delta={MAX_DELTA}")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    batch_outcome = run_batch_predict(dto)
    batch_elapsed = time.perf_counter() - t0
    if not isinstance(batch_outcome, Ok):
        raise RuntimeError(f"batch predict failed: {batch_outcome.error.message}")

    app = create_app()
    t1 = time.perf_counter()
    with TestClient(app) as client:
        api_res = client.post(
            "/api/v1/predictions",
            json=predict_request_payload_all_rows(dto),
            headers={"X-Tenant-ID": "local"},
        )
    api_elapsed = time.perf_counter() - t1
    if api_res.status_code != 200:
        raise RuntimeError(f"API failed: {api_res.status_code} {api_res.text}")

    api_probs = api_res.json()["probabilities"]
    batch_probs = batch_outcome.value.probabilities
    if len(api_probs) != len(batch_probs):
        raise RuntimeError(f"row mismatch: api={len(api_probs)} batch={len(batch_probs)}")

    deltas = [abs(a - b) for a, b in zip(batch_probs, api_probs, strict=True)]
    max_delta = max(deltas)
    mean_delta = sum(deltas) / len(deltas)

    if verbose:
        status = "OK" if max_delta < MAX_DELTA else "FAIL"
        print(
            f"Rows={len(rows)} max|Δp|={max_delta:.2e} mean|Δp|={mean_delta:.2e} "
            f"batch={batch_elapsed:.2f}s api={api_elapsed:.2f}s [{status}]",
            flush=True,
        )

    return BatchParityResult(
        n_rows=len(rows),
        max_delta=max_delta,
        mean_delta=mean_delta,
        batch_elapsed_s=round(batch_elapsed, 3),
        api_elapsed_s=round(api_elapsed, 3),
    )


def _summarize(result: BatchParityResult) -> str:
    accepted = result.max_delta < MAX_DELTA
    verdict = "accepted" if accepted else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 029 SUMMARY",
            f"{'=' * 60}",
            f"Rows: {result.n_rows}",
            f"Max |Δp|: {result.max_delta:.2e}",
            f"Mean |Δp|: {result.mean_delta:.2e}",
            f"Batch time: {result.batch_elapsed_s}s | API time: {result.api_elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 029 — batch vs API parity")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_029(
        profile=args.profile,
        input_path=args.input,
        verbose=not args.quiet,
        bootstrap_checkpoint=not args.no_bootstrap,
    )
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if result.max_delta < MAX_DELTA else 1


def _build_results_md(result: BatchParityResult) -> str:
    from datetime import date

    verdict = "accepted" if result.max_delta < MAX_DELTA else "rejected"
    return "\n".join(
        [
            "# Results — EXP 029: Batch Calculation vs API Parity",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Parity",
            "",
            f"- Rows scored: **{result.n_rows}**",
            f"- Max |Δp|: **{result.max_delta:.2e}**",
            f"- Mean |Δp|: **{result.mean_delta:.2e}**",
            f"- Batch elapsed: **{result.batch_elapsed_s}s**",
            f"- API elapsed: **{result.api_elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — batch script matches API within 1e-5 per row.",
            "",
            "## Limitations",
            "- Wisconsin Breast Cancer only; research prototype.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
