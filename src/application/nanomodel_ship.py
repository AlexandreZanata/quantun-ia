"""Unified train → gate → publish → export pipeline for shippable nanomodels."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.application.dto import TrainNanomodelDTO
from src.application.model_export import export_bundle
from src.application.nanomodel_registry import NanomodelSpec, get_nanomodel_spec
from src.application.open_serve import (
    ensure_large_nano_hybrid_serve_artifact,
    ensure_large_nano_serve_artifact,
    ensure_residual_nano_serve_artifact,
    verify_serve_artifact,
)
from src.application.train_nanomodel import execute as train_execute
from src.shared.result import Fail, Ok, Result, fail, ok
from src.training.checkpoints import checkpoint_path, load_checkpoint_bundle, resolve_checkpoint_dir
from src.training.structured_log import init_correlation_id, log_event


class ShipNanomodelError:
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ShipNanomodelDTO:
    registry_key: str
    root: Path | None = None
    profile: str | None = None
    retrain: bool = False
    skip_gate: bool = False
    skip_train: bool = False


@dataclass(frozen=True)
class ShipNanomodelResult:
    registry_key: str
    bundle_dir: str
    serve_dir: str
    stages: dict[str, Any]
    elapsed_s: float


def _training_checkpoint_exists(spec: NanomodelSpec) -> bool:
    if spec.train_kind == "none":
        return True
    if spec.serve_kind in {"open_large_nano", "open_hybrid", "open_residual_nano"}:
        raw = checkpoint_path(spec.exp_id, spec.train_model, spec.seed) / "best.pt"
        return raw.is_file()
    serve = resolve_checkpoint_dir(spec.exp_id, spec.train_model, spec.dataset, seed=spec.seed)
    return (serve / "best.pt").is_file()


def _serve_dir_exists(spec: NanomodelSpec) -> bool:
    serve = resolve_checkpoint_dir(spec.exp_id, spec.train_model, spec.dataset, seed=spec.seed)
    return (serve / "best.pt").is_file() and (serve / "scaler.joblib").is_file()



def _resolve_run_fn(spec: NanomodelSpec):
    """Map experiment_key to run function (exp_034 -> run_exp_034)."""
    if not spec.experiment_key:
        return None
    module = importlib.import_module(f"experiments.{spec.experiment_key}.run")
    suffix = spec.experiment_key.removeprefix("exp_").split("_", 1)[0]
    fn_name = f"run_exp_{suffix}"
    if hasattr(module, fn_name):
        return getattr(module, fn_name)
    for attr in dir(module):
        if attr.startswith("run_exp_"):
            return getattr(module, attr)
    return None


def _run_train(spec: NanomodelSpec, profile: str, *, require_cuda: bool) -> dict[str, Any]:
    if spec.train_kind == "none":
        return {"skipped": True, "reason": "depends_on_existing_serve"}
    if spec.train_kind == "experiment":
        run_fn = _resolve_run_fn(spec)
        if run_fn is None:
            msg = f"cannot resolve train runner for {spec.experiment_key}"
            raise RuntimeError(msg)
        run_fn(profile=profile, verbose=True, require_cuda=require_cuda)
        return {"kind": "experiment", "experiment_key": spec.experiment_key, "profile": profile}
    if spec.train_kind == "nanotrainer":
        outcome = train_execute(
            TrainNanomodelDTO(
                model_name=spec.train_model,
                dataset=spec.dataset,
                profile=profile,
                seed=spec.seed,
                exp_id=spec.exp_id,
                save_checkpoints=True,
            )
        )
        if isinstance(outcome, Fail):
            msg = f"nanotrainer train failed: {outcome.error.message}"
            raise RuntimeError(msg)
        assert isinstance(outcome, Ok)
        return {
            "kind": "nanotrainer",
            "accuracy": outcome.value.accuracy,
            "checkpoint_path": outcome.value.checkpoint_path,
        }
    msg = f"unknown train kind: {spec.train_kind}"
    raise RuntimeError(msg)


def _run_gate(spec: NanomodelSpec, root: Path) -> dict[str, Any]:
    if not spec.gate_test:
        return {"skipped": True, "reason": "no gate_test configured"}
    gate_path = root / spec.gate_test
    if not gate_path.is_file():
        msg = f"gate test missing: {gate_path}"
        raise FileNotFoundError(msg)
    env = {**os.environ, "MLFLOW_DISABLE": "1", "QML_DEVICE": "cuda"}
    cmd = [sys.executable, "-m", "pytest", str(gate_path), "-m", "real", "-q"]
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        msg = proc.stdout + proc.stderr
        raise RuntimeError(f"gate test failed ({spec.gate_test}): {msg}")
    return {"gate_test": spec.gate_test, "passed": True}


def _verify_serve(spec: NanomodelSpec) -> None:
    if spec.serve_kind == "nanotrainer":
        bundle = load_checkpoint_bundle(
            spec.exp_id,
            spec.train_model,
            spec.dataset,
            seed=spec.seed,
        )
        if not (bundle.directory / "scaler.joblib").is_file():
            msg = f"scaler missing for {spec.registry_key}"
            raise FileNotFoundError(msg)
        return
    verify_serve_artifact(spec.exp_id, spec.train_model, spec.dataset, seed=spec.seed)


def _publish_serve(spec: NanomodelSpec, root: Path) -> Path:
    if spec.serve_kind == "open_large_nano":
        return ensure_large_nano_serve_artifact(
            root,
            exp_id=spec.exp_id,
            model_name=spec.train_model,
            dataset_id=spec.dataset,
            seed=spec.seed,
        )
    if spec.serve_kind == "open_hybrid":
        return ensure_large_nano_hybrid_serve_artifact(
            root,
            exp_id=spec.exp_id,
            model_name=spec.train_model,
            dataset_id=spec.dataset,
            seed=spec.seed,
        )
    if spec.serve_kind == "open_residual_nano":
        return ensure_residual_nano_serve_artifact(
            root,
            exp_id=spec.exp_id,
            model_name=spec.train_model,
            dataset_id=spec.dataset,
            seed=spec.seed,
        )
    serve = resolve_checkpoint_dir(spec.exp_id, spec.train_model, spec.dataset, seed=spec.seed)
    if not (serve / "best.pt").is_file():
        msg = f"nanotrainer serve checkpoint missing: {serve}"
        raise FileNotFoundError(msg)
    return serve


def execute(dto: ShipNanomodelDTO) -> Result[ShipNanomodelResult, ShipNanomodelError]:
    """Run the full ship pipeline for a registry model key."""
    init_correlation_id()
    root = dto.root or Path(".")
    try:
        spec = get_nanomodel_spec(dto.registry_key)
    except KeyError as exc:
        return fail(ShipNanomodelError("UNKNOWN_MODEL", str(exc)))

    profile = dto.profile or spec.profile
    if profile == "publication" and dto.skip_gate:
        return fail(ShipNanomodelError("INVALID_FLAGS", "skip_gate forbidden for publication profile"))

    stages: dict[str, Any] = {}
    t0 = time.perf_counter()

    try:
        import torch

        require_cuda = profile in {"publication", "publication_large"} and torch.cuda.is_available()
        if profile in {"publication", "publication_large"} and not torch.cuda.is_available():
            return fail(ShipNanomodelError("CUDA_REQUIRED", "CUDA required for publication ship profile"))

        if spec.depends_on:
            dep = get_nanomodel_spec(spec.depends_on)
            if not _serve_dir_exists(dep):
                dep_outcome = execute(
                    ShipNanomodelDTO(
                        registry_key=dep.registry_key,
                        root=root,
                        profile=profile,
                        retrain=dto.retrain,
                        skip_gate=dto.skip_gate,
                        skip_train=dto.skip_train,
                    )
                )
                if isinstance(dep_outcome, Fail):
                    return fail(ShipNanomodelError("DEPENDENCY_FAILED", dep_outcome.error.message))
                stages["dependency"] = dep.registry_key

        need_train = dto.retrain or (not dto.skip_train and not _training_checkpoint_exists(spec))
        if need_train and spec.train_kind != "none":
            stages["train"] = _run_train(spec, profile, require_cuda=require_cuda)
        else:
            stages["train"] = {"skipped": True, "reason": "checkpoint exists or skip_train"}

        if not dto.skip_gate:
            stages["gate"] = _run_gate(spec, root)
        else:
            stages["gate"] = {"skipped": True}

        serve_dir = _publish_serve(spec, root)
        _verify_serve(spec)
        stages["publish"] = {"serve_dir": str(serve_dir)}

        bundle_dir = export_bundle(spec, serve_dir, root=root, stages=stages)
        stages["bundle"] = str(bundle_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return fail(ShipNanomodelError("SHIP_FAILED", str(exc)))

    elapsed = time.perf_counter() - t0
    log_event(
        "info",
        "nanomodel ship complete",
        registry_key=spec.registry_key,
        bundle_dir=str(bundle_dir),
        elapsed_s=round(elapsed, 3),
        record_source="nanomodel_ship",
    )
    return ok(
        ShipNanomodelResult(
            registry_key=spec.registry_key,
            bundle_dir=str(bundle_dir),
            serve_dir=str(serve_dir),
            stages=stages,
            elapsed_s=elapsed,
        )
    )
