"""
EXP 035 — LargeNanoMLP serve parity: batch vs API vs chatbot on Synthea holdout.

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_035_synthea_serve_parity/run.py --profile publication
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
    max_probability_delta,
    predict_request_payload_all_rows,
    run_batch_predict,
)
from src.application.chatbot_tool import (
    TOOL_SCORE_SYNTHEA_CV,
    ChatbotToolCallDTO,
    execute_tool_call,
)
from src.application.open_serve import (
    DEFAULT_SERVE_MODEL,
    ensure_large_nano_serve_artifact,
    load_open_holdout_rows,
)
from src.presentation.http.app import create_app
from src.shared.result import Ok
from src.training.config import load_experiment_config

EXP_KEY = "exp_035_synthea_serve_parity"
MAX_DELTA = 1e-5
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_EXP = "exp_034"
DEFAULT_DATASET = "synthea_cv_risk_v1"


@dataclass(frozen=True)
class ServeParityResult:
    n_rows: int
    max_delta_batch_api: float
    max_delta_tool_api: float
    max_delta_batch_tool: float
    batch_elapsed_s: float
    api_elapsed_s: float
    tool_elapsed_s: float


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def run_exp_035(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
    publish_serve: bool = True,
) -> ServeParityResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    n_rows = int(cfg["n_rows"])
    chunk_size = int(cfg.get("chunk_size", 256))
    exp_id = str(cfg.get("checkpoint_exp_id", DEFAULT_CHECKPOINT_EXP))
    model_name = str(cfg.get("model_name", DEFAULT_SERVE_MODEL))
    dataset_id = str(cfg.get("dataset_id", DEFAULT_DATASET))

    if publish_serve:
        if verbose:
            print(f"Publishing serve artifact {exp_id}/{model_name}/{dataset_id}...", flush=True)
        ensure_large_nano_serve_artifact(
            ROOT, exp_id=exp_id, model_name=model_name, dataset_id=dataset_id, seed=seed
        )

    rows = load_open_holdout_rows(dataset_id, ROOT, n_rows=n_rows, random_state=seed)
    dto = BatchPredictDTO(
        features=rows,
        exp_id=exp_id,
        model_name=model_name,
        dataset=dataset_id,
        seed=seed,
        chunk_size=chunk_size,
    )
    tool_dto = ChatbotToolCallDTO(
        tool_name=TOOL_SCORE_SYNTHEA_CV,
        arguments={"features": rows},
        exp_id=exp_id,
        model_name=model_name,
        dataset=dataset_id,
        seed=seed,
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXP 035 — Serve Parity | profile={profile} | rows={len(rows)}")
        print(f"Checkpoint: {exp_id} {model_name} × {dataset_id} seed={seed}")
        print(f"Gate: max |Δp| < {MAX_DELTA}")
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

    t2 = time.perf_counter()
    tool_outcome = execute_tool_call(tool_dto)
    tool_elapsed = time.perf_counter() - t2
    if not isinstance(tool_outcome, Ok):
        raise RuntimeError(f"chatbot tool failed: {tool_outcome.error.message}")

    batch_probs = batch_outcome.value.probabilities
    api_probs = api_res.json()["probabilities"]
    tool_probs = tool_outcome.value.probabilities

    delta_batch_api = max_probability_delta(batch_probs, api_probs)
    delta_tool_api = max_probability_delta(tool_probs, api_probs)
    delta_batch_tool = max_probability_delta(batch_probs, tool_probs)

    if verbose:
        ok = max(delta_batch_api, delta_tool_api, delta_batch_tool) < MAX_DELTA
        status = "OK" if ok else "FAIL"
        print(
            f"batch↔api={delta_batch_api:.2e} tool↔api={delta_tool_api:.2e} "
            f"batch↔tool={delta_batch_tool:.2e} "
            f"batch={batch_elapsed:.2f}s api={api_elapsed:.2f}s tool={tool_elapsed:.2f}s [{status}]",
            flush=True,
        )

    return ServeParityResult(
        n_rows=len(rows),
        max_delta_batch_api=delta_batch_api,
        max_delta_tool_api=delta_tool_api,
        max_delta_batch_tool=delta_batch_tool,
        batch_elapsed_s=round(batch_elapsed, 3),
        api_elapsed_s=round(api_elapsed, 3),
        tool_elapsed_s=round(tool_elapsed, 3),
    )


def _passed(result: ServeParityResult) -> bool:
    return (
        result.max_delta_batch_api < MAX_DELTA
        and result.max_delta_tool_api < MAX_DELTA
        and result.max_delta_batch_tool < MAX_DELTA
    )


def _summarize(result: ServeParityResult) -> str:
    verdict = "accepted" if _passed(result) else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 035 SUMMARY",
            f"{'=' * 60}",
            f"Rows: {result.n_rows}",
            f"Max |Δp| batch↔api: {result.max_delta_batch_api:.2e}",
            f"Max |Δp| tool↔api: {result.max_delta_tool_api:.2e}",
            f"Max |Δp| batch↔tool: {result.max_delta_batch_tool:.2e}",
            f"Batch: {result.batch_elapsed_s}s | API: {result.api_elapsed_s}s | Tool: {result.tool_elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 035 — Synthea serve parity")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_035(
        profile=args.profile,
        verbose=not args.quiet,
        publish_serve=not args.no_publish,
    )
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if _passed(result) else 1


def _build_results_md(result: ServeParityResult) -> str:
    from datetime import date

    verdict = "accepted" if _passed(result) else "rejected"
    return "\n".join(
        [
            "# Results — EXP 035: Synthea Serve Parity",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Parity",
            "",
            f"- Rows scored: **{result.n_rows}**",
            f"- Max |Δp| batch↔api: **{result.max_delta_batch_api:.2e}**",
            f"- Max |Δp| tool↔api: **{result.max_delta_tool_api:.2e}**",
            f"- Max |Δp| batch↔tool: **{result.max_delta_batch_tool:.2e}**",
            "",
            "## Verdict",
            f"**{verdict}** — batch, API, and chatbot tool match within 1e-5.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
