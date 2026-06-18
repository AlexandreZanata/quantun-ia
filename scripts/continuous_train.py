#!/usr/bin/env python3
"""Continuous training — retrain challenger, compare vs champion, promote if within bounds."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

from src.application.dto import TrainNanomodelDTO
from src.application.train_nanomodel import execute as train_execute
from src.shared.result import Fail, Ok
from src.training.champion import (
    holdout_delta_pp,
    load_champion_manifest,
    manifest_from_training,
    promote_champion,
    should_promote,
    should_rollback,
)
from src.training.structured_log import init_correlation_id, log_event


@dataclass(frozen=True)
class ContinuousTrainResult:
    champion_accuracy: float
    challenger_accuracy: float
    delta_pp: float
    promoted: bool
    blocked: bool
    checkpoint_path: str | None


def run_continuous_train(
    *,
    model_name: str = "hybrid_sandwich",
    dataset: str = "breast_cancer",
    profile: str = "publication",
    epochs: int | None = None,
    seed: int = 42,
    challenger_exp_id: str = "exp_027",
    champion_exp_id: str | None = None,
    champion_seed: int | None = None,
    promote_max_delta_pp: float = 0.5,
    rollback_regression_pp: float = 1.0,
    force_promote: bool = False,
    bootstrap_only: bool = False,
) -> ContinuousTrainResult:
    """Train challenger and optionally promote to champion."""
    init_correlation_id()
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    champion_exp = champion_exp_id or os.environ.get("CHAMPION_EXP_ID", "quantum_nano_bc_app")
    champion_seed_val = champion_seed if champion_seed is not None else int(
        os.environ.get("CHAMPION_SEED", "42")
    )

    manifest = load_champion_manifest()
    if manifest is None:
        bootstrap_dto = TrainNanomodelDTO(
            model_name=model_name,
            dataset=dataset,
            profile=profile,
            epochs=epochs,
            seed=champion_seed_val,
            exp_id=champion_exp,
            save_checkpoints=True,
        )
        bootstrap = train_execute(bootstrap_dto)
        if isinstance(bootstrap, Fail):
            raise RuntimeError(f"champion bootstrap failed: {bootstrap.error.message}")
        assert isinstance(bootstrap, Ok)
        br = bootstrap.value
        manifest = manifest_from_training(
            exp_id=champion_exp,
            model_name=model_name,
            dataset=dataset,
            seed=champion_seed_val,
            holdout_accuracy=br.accuracy,
            checkpoint_path=br.checkpoint_path,
        )
        promote_champion(manifest)
        log_event(
            "info",
            "champion bootstrapped",
            exp_id=champion_exp,
            holdout_accuracy=br.accuracy,
            record_source="continuous_train",
        )
        if bootstrap_only:
            return ContinuousTrainResult(
                champion_accuracy=br.accuracy,
                challenger_accuracy=br.accuracy,
                delta_pp=0.0,
                promoted=True,
                blocked=False,
                checkpoint_path=br.checkpoint_path,
            )

    champion_accuracy = manifest.holdout_accuracy

    train_dto = TrainNanomodelDTO(
        model_name=model_name,
        dataset=dataset,
        profile=profile,
        epochs=epochs,
        seed=seed,
        exp_id=challenger_exp_id,
        save_checkpoints=True,
    )
    outcome = train_execute(train_dto)
    if isinstance(outcome, Fail):
        raise RuntimeError(f"challenger train failed: {outcome.error.message}")
    assert isinstance(outcome, Ok)
    challenger = outcome.value
    challenger_accuracy = challenger.accuracy
    delta = holdout_delta_pp(challenger_accuracy, champion_accuracy)

    blocked = should_rollback(
        challenger_accuracy,
        champion_accuracy,
        max_regression_pp=rollback_regression_pp,
    )
    can_promote = should_promote(
        challenger_accuracy,
        champion_accuracy,
        max_delta_pp=promote_max_delta_pp,
    )
    promoted = False

    if blocked:
        log_event(
            "warn",
            "challenger blocked — holdout regression exceeds rollback threshold",
            champion_accuracy=champion_accuracy,
            challenger_accuracy=challenger_accuracy,
            delta_pp=delta,
            record_source="continuous_train",
        )
    elif can_promote or force_promote:
        new_manifest = manifest_from_training(
            exp_id=challenger_exp_id,
            model_name=model_name,
            dataset=dataset,
            seed=seed,
            holdout_accuracy=challenger_accuracy,
            checkpoint_path=challenger.checkpoint_path,
        )
        promote_champion(new_manifest)
        promoted = True
        log_event(
            "info",
            "challenger promoted to champion",
            exp_id=challenger_exp_id,
            holdout_accuracy=challenger_accuracy,
            delta_pp=delta,
            record_source="continuous_train",
        )
    else:
        log_event(
            "info",
            "challenger kept — outside promotion tolerance",
            champion_accuracy=champion_accuracy,
            challenger_accuracy=challenger_accuracy,
            delta_pp=delta,
            record_source="continuous_train",
        )

    return ContinuousTrainResult(
        champion_accuracy=champion_accuracy,
        challenger_accuracy=challenger_accuracy,
        delta_pp=delta,
        promoted=promoted,
        blocked=blocked,
        checkpoint_path=challenger.checkpoint_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Continuous training champion/challenger gate")
    parser.add_argument("--model", default="hybrid_sandwich")
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--profile", default="publication")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--challenger-exp-id", default="exp_027", dest="challenger_exp_id")
    parser.add_argument("--champion-exp-id", default=None, dest="champion_exp_id")
    parser.add_argument("--champion-seed", type=int, default=None, dest="champion_seed")
    parser.add_argument("--promote-max-delta-pp", type=float, default=0.5)
    parser.add_argument("--rollback-regression-pp", type=float, default=1.0)
    parser.add_argument("--force-promote", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = run_continuous_train(
        model_name=args.model,
        dataset=args.dataset,
        profile=args.profile,
        epochs=args.epochs,
        seed=args.seed,
        challenger_exp_id=args.challenger_exp_id,
        champion_exp_id=args.champion_exp_id,
        champion_seed=args.champion_seed,
        promote_max_delta_pp=args.promote_max_delta_pp,
        rollback_regression_pp=args.rollback_regression_pp,
        force_promote=args.force_promote,
    )

    payload = {
        "champion_accuracy": result.champion_accuracy,
        "challenger_accuracy": result.challenger_accuracy,
        "delta_pp": result.delta_pp,
        "promoted": result.promoted,
        "blocked": result.blocked,
        "checkpoint_path": result.checkpoint_path,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        status = "PROMOTED" if result.promoted else ("BLOCKED" if result.blocked else "KEPT")
        print(
            f"Continuous train [{status}] — "
            f"champion={result.champion_accuracy * 100:.2f}% "
            f"challenger={result.challenger_accuracy * 100:.2f}% "
            f"Δ={result.delta_pp:.2f} pp"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
