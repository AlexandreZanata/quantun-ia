"""Real gate — ship P0 nanomodel bundles on RTX 4060 (skip retrain when checkpoint exists)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.nanomodel_ship import ShipNanomodelDTO, execute
from src.shared.result import Ok

pytestmark = pytest.mark.real


@pytest.mark.parametrize(
    "registry_key",
    [
        "large_nano_mlp_synthea",
        "large_nano_mlp_higgs",
        "quantum_nano_bc",
    ],
)
def test_ship_p0_bundle_on_cuda(registry_key: str, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for nanomodel ship real gate")

    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")

    outcome = execute(
        ShipNanomodelDTO(
            registry_key=registry_key,
            root=root,
            profile="ci",
            skip_train=True,
            skip_gate=True,
        )
    )
    assert isinstance(outcome, Ok), getattr(outcome, "error", outcome)
    bundle = Path(outcome.value.bundle_dir)
    assert (bundle / "best.pt").is_file()
    assert (bundle / "MANIFEST.sha256").is_file()
    assert (bundle / "inference" / "predict.py").is_file()
