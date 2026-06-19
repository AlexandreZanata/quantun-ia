"""Export shippable nanomodel artifacts — ONNX, TorchScript, inference helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.application.model_registry import build_model
from src.application.nanomodel_registry import NanomodelSpec
from src.application.open_serve import build_large_nano_hybrid_for_inference
from src.training.checkpoints import load_checkpoint_bundle


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest_sha256(bundle_dir: Path) -> Path:
    """Write MANIFEST.sha256 for all files under bundle_dir."""
    lines: list[str] = []
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rel = path.relative_to(bundle_dir)
            lines.append(f"{rel} sha256:{_sha256_file(path)}")
    manifest = bundle_dir / "MANIFEST.sha256"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def write_schema_json(spec: NanomodelSpec, serve_dir: Path, out_path: Path) -> None:
    bundle = load_checkpoint_bundle(spec.exp_id, spec.train_model, spec.dataset, seed=spec.seed)
    input_dim = int(bundle.config.get("input_dim", 0))
    payload = {
        "registry_key": spec.registry_key,
        "train_model": spec.train_model,
        "dataset": spec.dataset,
        "exp_id": spec.exp_id,
        "seed": spec.seed,
        "input_dim": input_dim,
        "feature_names": [f"feature_{i}" for i in range(input_dim)],
        "label_semantics": {"0": "negative", "1": "positive"},
        "serve_checkpoint": str(serve_dir),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_predict_script(out_path: Path) -> None:
    """Write standalone CPU inference script for bundled models."""
    script = '''#!/usr/bin/env python3
"""Run inference on a bundled shippable nanomodel (CPU)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "application").is_dir():
            return parent
    msg = "quantun-ia source tree not found — pip install -e . from repo root"
    raise RuntimeError(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict with a shippable nanomodel bundle")
    parser.add_argument("--input", required=True, help="JSON file with {features: [[...]]}")
    args = parser.parse_args()

    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    schema = json.loads((BUNDLE_DIR / "inference" / "schema.json").read_text(encoding="utf-8"))
    config = json.loads((BUNDLE_DIR / "config.json").read_text(encoding="utf-8"))["config"]
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload.get("features") or payload.get("feature_rows") or []
    if not rows:
        print("No features in input JSON", file=sys.stderr)
        return 1

    import joblib

    scaler = joblib.load(BUNDLE_DIR / "scaler.joblib")
    scaled = scaler.transform(np.asarray(rows, dtype=np.float32))
    x = torch.tensor(scaled, dtype=torch.float32)

    train_model = schema["train_model"]
    input_dim = int(schema["input_dim"])
    if train_model == "large_nano_hybrid":
        from src.application.open_serve import build_large_nano_hybrid_for_inference

        state = torch.load(BUNDLE_DIR / "best.pt", map_location="cpu", weights_only=True)
        model = build_large_nano_hybrid_for_inference(state, n_features=input_dim, config=config)
    else:
        from src.application.model_registry import build_model

        model, _ = build_model(train_model, input_dim=input_dim)
        model.load_state_dict(torch.load(BUNDLE_DIR / "best.pt", map_location="cpu", weights_only=True))

    model.eval()
    with torch.no_grad():
        probs = model(x).cpu().numpy().reshape(-1).tolist()
    labels = [1 if p >= 0.5 else 0 for p in probs]
    print(json.dumps({"probabilities": probs, "labels": labels}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script, encoding="utf-8")
    out_path.chmod(out_path.stat().st_mode | 0o111)


def _build_inference_model(spec: NanomodelSpec, bundle: Any) -> nn.Module:
    input_dim = int(bundle.config.get("input_dim", 0))
    if spec.train_model == "large_nano_hybrid":
        return build_large_nano_hybrid_for_inference(
            bundle.state_dict,
            n_features=input_dim,
            config=bundle.config,
        )
    model, _ = build_model(spec.train_model, input_dim=input_dim)
    model.load_state_dict(bundle.state_dict)
    model.eval()
    return model


def export_torchscript(spec: NanomodelSpec, serve_dir: Path, out_path: Path) -> Path | None:
    bundle = load_checkpoint_bundle(spec.exp_id, spec.train_model, spec.dataset, seed=spec.seed)
    model = _build_inference_model(spec, bundle)
    input_dim = int(bundle.config.get("input_dim", 0))
    example = torch.zeros(1, input_dim, dtype=torch.float32)
    try:
        traced = torch.jit.trace(model, example)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(out_path))
        return out_path
    except Exception:
        return None


def export_onnx(spec: NanomodelSpec, serve_dir: Path, out_path: Path) -> Path | None:
    if spec.train_model in {"large_nano_hybrid", "hybrid_sandwich", "quantum_angle", "quantum_amplitude"}:
        return None
    bundle = load_checkpoint_bundle(spec.exp_id, spec.train_model, spec.dataset, seed=spec.seed)
    model = _build_inference_model(spec, bundle)
    input_dim = int(bundle.config.get("input_dim", 0))
    example = torch.zeros(1, input_dim, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            example,
            str(out_path),
            input_names=["features"],
            output_names=["probability"],
            dynamic_axes={"features": {0: "batch"}, "probability": {0: "batch"}},
            opset_version=17,
        )
        return out_path
    except Exception:
        return None


def copy_native_bundle(serve_dir: Path, native_dir: Path) -> None:
    native_dir.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "config.json", "scaler.joblib", "calibration_isotonic.json"):
        src = serve_dir / name
        if src.is_file():
            shutil.copy2(src, native_dir / name)


def copy_calibration_artifact(
    calibration_exp_id: str,
    spec: NanomodelSpec,
    bundle_dir: Path,
) -> Path | None:
    src = (
        Path("artifacts")
        / calibration_exp_id
        / f"{spec.train_model}_{spec.dataset}"
        / f"seed_{spec.seed}"
        / "calibration_isotonic.json"
    )
    if not src.is_file():
        return None
    dest = bundle_dir / "calibration_isotonic.json"
    shutil.copy2(src, dest)
    return dest


def write_metrics_json(
    bundle_dir: Path,
    *,
    spec: NanomodelSpec,
    serve_dir: Path,
    stages: dict[str, Any],
) -> Path:
    bundle = load_checkpoint_bundle(spec.exp_id, spec.train_model, spec.dataset, seed=spec.seed)
    payload = {
        "registry_key": spec.registry_key,
        "description": spec.description,
        "exp_id": spec.exp_id,
        "train_model": spec.train_model,
        "dataset": spec.dataset,
        "seed": spec.seed,
        "profile": spec.profile,
        "metadata": bundle.metadata,
        "exports": list(spec.exports),
        "stages": stages,
    }
    path = bundle_dir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_model_card(spec: NanomodelSpec, bundle_dir: Path, root: Path) -> Path:
    src = root / spec.model_card
    dest = bundle_dir / "model_card.md"
    if src.is_file():
        shutil.copy2(src, dest)
    else:
        dest.write_text(
            f"# {spec.registry_key}\n\n{spec.description}\n\n"
            f"- exp_id: `{spec.exp_id}`\n"
            f"- model: `{spec.train_model}`\n"
            f"- dataset: `{spec.dataset}`\n",
            encoding="utf-8",
        )
    return dest


def export_bundle(
    spec: NanomodelSpec,
    serve_dir: Path,
    *,
    root: Path,
    stages: dict[str, Any],
) -> Path:
    """Assemble dist/serve_models/{registry_key}/ from a published serve checkpoint."""
    bundle_dir = root / spec.bundle_dir
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    for name in ("best.pt", "config.json", "scaler.joblib"):
        src = serve_dir / name
        if not src.is_file():
            msg = f"missing serve artifact: {src}"
            raise FileNotFoundError(msg)
        shutil.copy2(src, bundle_dir / name)

    copy_calibration_artifact(spec.calibration_exp_id or "", spec, bundle_dir)

    exports_dir = bundle_dir / "exports"
    export_paths: dict[str, str | None] = {}
    if "native" in spec.exports:
        native_dir = exports_dir / "native"
        copy_native_bundle(serve_dir, native_dir)
        export_paths["native"] = str(native_dir.relative_to(bundle_dir))
    if "torchscript" in spec.exports:
        ts_path = exports_dir / "torchscript" / "model.pt"
        result = export_torchscript(spec, serve_dir, ts_path)
        export_paths["torchscript"] = str(result.relative_to(bundle_dir)) if result else None
    if "onnx" in spec.exports:
        onnx_path = exports_dir / "onnx" / "model.onnx"
        result = export_onnx(spec, serve_dir, onnx_path)
        export_paths["onnx"] = str(result.relative_to(bundle_dir)) if result else None

    inference_dir = bundle_dir / "inference"
    write_schema_json(spec, serve_dir, inference_dir / "schema.json")
    write_predict_script(inference_dir / "predict.py")

    stages = {**stages, "export_paths": export_paths}
    write_metrics_json(bundle_dir, spec=spec, serve_dir=serve_dir, stages=stages)
    write_model_card(spec, bundle_dir, root)
    write_manifest_sha256(bundle_dir)
    return bundle_dir


def install_bundle_to_artifacts(spec: NanomodelSpec, bundle_dir: Path) -> Path:
    """Copy a dist bundle back into artifacts/ for API and Streamlit inference."""
    from src.training.checkpoints import resolve_checkpoint_dir

    target = resolve_checkpoint_dir(spec.exp_id, spec.train_model, spec.dataset, seed=spec.seed)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "config.json", "scaler.joblib", "calibration_isotonic.json"):
        src = bundle_dir / name
        if src.is_file():
            shutil.copy2(src, target / name)
    return target
