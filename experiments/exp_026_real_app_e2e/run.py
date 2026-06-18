"""
EXP 026 — Real application E2E: async API (CUDA) vs CLI holdout parity.

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_026_real_app_e2e/run.py --profile ci
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

from src.application.dto import TrainNanomodelDTO
from src.application.train_nanomodel import execute as train_execute
from src.presentation.http.app import create_app
from src.shared.result import Ok
from src.training.config import load_experiment_config

EXP_KEY = "exp_026_real_app_e2e"
EXP_ID = "exp_026"
MODEL = "hybrid_sandwich"
DATASET = "breast_cancer"
MAX_DELTA_PP = 0.5
POLL_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class SeedPairResult:
    seed: int
    cli_accuracy: float
    api_accuracy: float
    delta_pp: float
    cli_elapsed_s: float
    api_elapsed_s: float


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _run_cli(
    *,
    seed: int,
    epochs: int,
    profile: str,
) -> tuple[float, float]:
    dto = TrainNanomodelDTO(
        model_name=MODEL,
        dataset=DATASET,
        profile=profile,
        epochs=epochs,
        seed=seed,
        exp_id=EXP_ID,
    )
    t0 = time.perf_counter()
    outcome = train_execute(dto)
    elapsed = time.perf_counter() - t0
    if not isinstance(outcome, Ok):
        raise RuntimeError(f"CLI failed seed={seed}: {outcome.error.message}")
    return outcome.value.accuracy, elapsed


def _run_api_async(
    client: TestClient,
    *,
    seed: int,
    epochs: int,
    profile: str,
) -> tuple[float, float]:
    t0 = time.perf_counter()
    res = client.post(
        "/api/v1/training-jobs",
        json={
            "model_name": MODEL,
            "dataset": DATASET,
            "profile": profile,
            "epochs": epochs,
            "seed": seed,
            "exp_id": EXP_ID,
            "device": "cuda",
            "async_mode": True,
        },
        headers={"X-Tenant-ID": "local"},
    )
    if res.status_code != 202:
        raise RuntimeError(f"API enqueue failed seed={seed}: {res.status_code} {res.text}")

    job_id = res.json()["id"]
    deadline = time.time() + POLL_TIMEOUT_S
    status = "PENDING"
    body: dict = res.json()

    while time.time() < deadline:
        poll = client.get(
            f"/api/v1/training-jobs/{job_id}",
            headers={"X-Tenant-ID": "local"},
        )
        if poll.status_code != 200:
            raise RuntimeError(f"API poll failed seed={seed}: {poll.text}")
        body = poll.json()
        status = body["status"]
        if status in {"COMPLETED", "FAILED"}:
            break
        time.sleep(0.25)

    elapsed = time.perf_counter() - t0
    if status != "COMPLETED":
        detail = body.get("error_message") or body
        raise RuntimeError(f"API job not completed seed={seed}: status={status} {detail}")

    result = body.get("result") or {}
    accuracy = float(result["accuracy"])
    return accuracy, elapsed


def run_exp_026(
    *,
    profile: str = "ci",
    db_path: Path | None = None,
    verbose: bool = True,
) -> list[SeedPairResult]:
    """Compare CLI vs async API holdout accuracy for each configured seed."""
    _require_cuda()
    os.environ.setdefault("MLFLOW_DISABLE", "1")
    os.environ["QML_DEVICE"] = "cuda"

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seeds: list[int] = list(cfg["seeds"])
    epochs: int = int(cfg["epochs"])

    db = db_path or Path("data") / "exp_026_api.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    os.environ["DATABASE_PATH"] = str(db)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXP 026 — Real App E2E | profile={profile} | seeds={len(seeds)} | epochs={epochs}")
        print(f"Model: {MODEL} × {DATASET} | device=cuda | max_delta={MAX_DELTA_PP} pp")
        print(f"{'=' * 60}\n")

    pairs: list[SeedPairResult] = []
    app = create_app()

    with TestClient(app) as client:
        health = client.get("/health")
        if health.status_code != 200:
            raise RuntimeError(f"API health failed: {health.text}")

        for idx, seed in enumerate(seeds, start=1):
            if verbose:
                print(f"[{idx}/{len(seeds)}] seed={seed} — CLI training...", flush=True)
            cli_acc, cli_elapsed = _run_cli(seed=seed, epochs=epochs, profile=profile)
            if verbose:
                print(
                    f"         CLI done — holdout={cli_acc * 100:.2f}% ({cli_elapsed:.1f}s)",
                    flush=True,
                )
                print(f"         seed={seed} — API async (cuda)...", flush=True)

            api_acc, api_elapsed = _run_api_async(
                client, seed=seed, epochs=epochs, profile=profile
            )
            delta_pp = abs(cli_acc - api_acc) * 100.0
            pair = SeedPairResult(
                seed=seed,
                cli_accuracy=cli_acc,
                api_accuracy=api_acc,
                delta_pp=delta_pp,
                cli_elapsed_s=round(cli_elapsed, 2),
                api_elapsed_s=round(api_elapsed, 2),
            )
            pairs.append(pair)

            if verbose:
                verdict = "OK" if delta_pp <= MAX_DELTA_PP else "FAIL"
                print(
                    f"         API done — holdout={api_acc * 100:.2f}% ({api_elapsed:.1f}s) "
                    f"| Δ={delta_pp:.2f} pp [{verdict}]",
                    flush=True,
                )
                print(flush=True)

    return pairs


def _summarize(pairs: list[SeedPairResult]) -> str:
    mean_delta = sum(p.delta_pp for p in pairs) / len(pairs)
    max_delta = max(p.delta_pp for p in pairs)
    accepted = max_delta <= MAX_DELTA_PP
    verdict = "accepted" if accepted else "rejected"

    lines = [
        f"\n{'=' * 60}",
        "EXP 026 SUMMARY",
        f"{'=' * 60}",
        f"{'Seed':>6}  {'CLI %':>8}  {'API %':>8}  {'Δ pp':>7}  {'CLI s':>6}  {'API s':>6}",
        "-" * 60,
    ]
    for p in pairs:
        lines.append(
            f"{p.seed:>6}  {p.cli_accuracy * 100:>7.2f}%  {p.api_accuracy * 100:>7.2f}%  "
            f"{p.delta_pp:>6.2f}  {p.cli_elapsed_s:>6.1f}  {p.api_elapsed_s:>6.1f}"
        )
    lines.extend(
        [
            "-" * 60,
            f"Mean |Δ|: {mean_delta:.3f} pp  |  Max |Δ|: {max_delta:.3f} pp  |  Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 026 — API vs CLI real-app parity")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    pairs = run_exp_026(profile=args.profile, verbose=not args.quiet)
    summary = _summarize(pairs)
    print(summary)

    max_delta = max(p.delta_pp for p in pairs)
    return 0 if max_delta <= MAX_DELTA_PP else 1


if __name__ == "__main__":
    raise SystemExit(main())
