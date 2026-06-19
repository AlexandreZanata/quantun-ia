"""Integration — download bundled nanomodel and run CPU inference."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.application.dto import PredictNanomodelDTO
from src.application.model_export import install_bundle_to_artifacts
from src.application.nanomodel_registry import get_nanomodel_spec
from src.application.nanomodel_ship import ShipNanomodelDTO, execute
from src.application.predict_nanomodel import execute as predict_execute
from src.shared.result import Ok
from src.training.checkpoints import resolve_checkpoint_dir


def test_ship_install_and_predict_via_api_path(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = get_nanomodel_spec("large_nano_mlp_synthea")
    serve = resolve_checkpoint_dir(spec.exp_id, spec.train_model, spec.dataset, seed=spec.seed)
    if not (serve / "best.pt").is_file():
        pytest.skip("exp_034 serve artifact not available — run exp-034-publication first")

    monkeypatch.chdir(root)
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")

    outcome = execute(
        ShipNanomodelDTO(
            registry_key="large_nano_mlp_synthea",
            root=root,
            profile="ci",
            skip_train=True,
            skip_gate=True,
        )
    )
    assert isinstance(outcome, Ok)
    bundle_dir = Path(outcome.value.bundle_dir)
    install_bundle_to_artifacts(spec, bundle_dir)

    fixture = root / "tests" / "fixtures" / "synthea_patient_row.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    pred = predict_execute(
        PredictNanomodelDTO(
            exp_id=spec.exp_id,
            model_name=spec.train_model,
            dataset=spec.dataset,
            seed=spec.seed,
            features=payload["features"],
        )
    )
    assert isinstance(pred, Ok)
    assert len(pred.value.probabilities) == 1


def test_bundled_predict_script_cpu(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = get_nanomodel_spec("large_nano_mlp_synthea")
    bundle_dir = root / spec.bundle_dir
    if not (bundle_dir / "inference" / "predict.py").is_file():
        pytest.skip("bundle not shipped — run qml-ship --model large_nano_mlp_synthea --skip-train")

    fixture = root / "tests" / "fixtures" / "synthea_patient_row.json"
    proc = subprocess.run(
        [sys.executable, str(bundle_dir / "inference" / "predict.py"), "--input", str(fixture)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    body = json.loads(proc.stdout)
    assert "probabilities" in body
