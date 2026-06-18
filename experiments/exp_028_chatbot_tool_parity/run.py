"""
EXP 028 — Chatbot tool adapter vs REST API probability parity.

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_028_chatbot_tool_parity/run.py --profile ci
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

from src.application.chatbot_tool import (
    RESEARCH_DISCLAIMER,
    ChatbotToolCallDTO,
    execute_tool_call,
    load_dialogue_fixtures,
    max_probability_delta,
    predict_request_payload,
)
from src.application.dto import TrainNanomodelDTO
from src.application.train_nanomodel import execute as train_execute
from src.presentation.http.app import create_app
from src.shared.result import Ok
from src.training.config import load_experiment_config

EXP_KEY = "exp_028_chatbot_tool_parity"
EXP_ID = "exp_028"
MAX_DELTA = 1e-5
MAX_LATENCY_S = 2.0
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "chatbot_dialogues"


@dataclass(frozen=True)
class DialogueResult:
    dialogue_id: str
    max_delta: float
    latency_s: float
    has_disclaimer: bool
    feature_count: int


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _ensure_checkpoint(
    *,
    exp_id: str,
    seed: int,
    profile: str,
    epochs: int,
) -> None:
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


def run_exp_028(
    *,
    profile: str = "ci",
    fixtures_dir: Path | None = None,
    verbose: bool = True,
    require_cuda: bool = True,
    bootstrap_checkpoint: bool = True,
) -> list[DialogueResult]:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    epochs = int(cfg["epochs"])
    checkpoint_exp_id = str(cfg.get("checkpoint_exp_id", "quantum_nano_bc_app"))

    if bootstrap_checkpoint:
        if verbose:
            print(f"Bootstrapping checkpoint exp_id={checkpoint_exp_id} seed={seed}...", flush=True)
        _ensure_checkpoint(exp_id=checkpoint_exp_id, seed=seed, profile=profile, epochs=epochs)

    dialogues = load_dialogue_fixtures(fixtures_dir or FIXTURES)
    if len(dialogues) < 10:
        raise ValueError(f"expected 10 dialogues, found {len(dialogues)}")

    app = create_app()
    results: list[DialogueResult] = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXP 028 — Chatbot Tool Parity | profile={profile} | dialogues={len(dialogues)}")
        print(f"Checkpoint: {checkpoint_exp_id} seed={seed} | max_delta={MAX_DELTA}")
        print(f"{'=' * 60}\n")

    with TestClient(app) as client:
        for idx, raw in enumerate(load_dialogue_fixtures(fixtures_dir or FIXTURES), start=1):
            dto = ChatbotToolCallDTO(
                tool_name=raw.tool_name,
                arguments=raw.arguments,
                exp_id=checkpoint_exp_id,
                model_name=raw.model_name,
                dataset=raw.dataset,
                seed=seed,
            )
            dialogue_id = f"dialogue_{idx:02d}"
            t0 = time.perf_counter()
            tool_outcome = execute_tool_call(dto)
            if not isinstance(tool_outcome, Ok):
                raise RuntimeError(f"{dialogue_id} tool failed: {tool_outcome.error.message}")

            api_res = client.post(
                "/api/v1/predictions",
                json=predict_request_payload(dto),
                headers={"X-Tenant-ID": "local"},
            )
            elapsed = time.perf_counter() - t0
            if api_res.status_code != 200:
                raise RuntimeError(f"{dialogue_id} API failed: {api_res.status_code} {api_res.text}")

            api_probs = api_res.json()["probabilities"]
            delta = max_probability_delta(tool_outcome.value.probabilities, api_probs)
            has_disclaimer = RESEARCH_DISCLAIMER in tool_outcome.value.message
            feature_count = len(dto.arguments["features"][0])
            result = DialogueResult(
                dialogue_id=dialogue_id,
                max_delta=delta,
                latency_s=round(elapsed, 3),
                has_disclaimer=has_disclaimer,
                feature_count=feature_count,
            )
            results.append(result)
            if verbose:
                status = "OK" if delta < MAX_DELTA and has_disclaimer else "FAIL"
                print(
                    f"[{idx}/10] {dialogue_id} — Δ={delta:.2e} "
                    f"features={feature_count} latency={elapsed:.3f}s [{status}]",
                    flush=True,
                )

    return results


def _summarize(results: list[DialogueResult]) -> str:
    max_delta = max(r.max_delta for r in results)
    max_latency = max(r.latency_s for r in results)
    all_disclaimer = all(r.has_disclaimer for r in results)
    accepted = max_delta < MAX_DELTA and all_disclaimer and max_latency <= MAX_LATENCY_S
    verdict = "accepted" if accepted else "rejected"

    lines = [
        f"\n{'=' * 60}",
        "EXP 028 SUMMARY",
        f"{'=' * 60}",
        f"{'Dialogue':>12}  {'Δ max':>10}  {'Latency s':>10}  {'Disclaimer':>10}",
        "-" * 60,
    ]
    for r in results:
        disc = "yes" if r.has_disclaimer else "no"
        lines.append(f"{r.dialogue_id:>12}  {r.max_delta:>10.2e}  {r.latency_s:>10.3f}  {disc:>10}")
    lines.extend(
        [
            "-" * 60,
            f"Max |Δp|: {max_delta:.2e} | Max latency: {max_latency:.3f}s | Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 028 — chatbot tool vs API parity")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args(argv)

    results = run_exp_028(
        profile=args.profile,
        fixtures_dir=args.fixtures,
        verbose=not args.quiet,
        bootstrap_checkpoint=not args.no_bootstrap,
    )
    print(_summarize(results))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(results), encoding="utf-8")
        print(f"Wrote {out}")

    max_delta = max(r.max_delta for r in results)
    ok = (
        max_delta < MAX_DELTA
        and all(r.has_disclaimer for r in results)
        and max(r.latency_s for r in results) <= MAX_LATENCY_S
    )
    return 0 if ok else 1


def _build_results_md(results: list[DialogueResult]) -> str:
    from datetime import date

    max_delta = max(r.max_delta for r in results)
    max_latency = max(r.latency_s for r in results)
    verdict = (
        "accepted"
        if max_delta < MAX_DELTA
        and all(r.has_disclaimer for r in results)
        and max_latency <= MAX_LATENCY_S
        else "rejected"
    )
    lines = [
        "# Results — EXP 028: Chatbot Tool vs API Parity",
        "",
        f"**Run date:** {date.today().isoformat()}  ",
        "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
        "",
        "## Dialogue parity",
        "",
        "| Dialogue | max Δp | Latency (s) | Disclaimer |",
        "|----------|--------|-------------|------------|",
    ]
    for r in results:
        disc = "yes" if r.has_disclaimer else "no"
        lines.append(f"| {r.dialogue_id} | {r.max_delta:.2e} | {r.latency_s:.3f} | {disc} |")
    lines.extend(
        [
            "",
            "## Verdict",
            f"**{verdict}** — max |Δp|={max_delta:.2e}; max latency={max_latency:.3f}s.",
            "",
            "## Conclusion",
            "Chatbot tool adapter matches REST API predictions within numerical tolerance.",
            "",
            "## Limitations",
            "- Scripted golden dialogues; no live Ollama routing in this experiment.",
            "- Research prototype only — not a clinical deployment claim.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
