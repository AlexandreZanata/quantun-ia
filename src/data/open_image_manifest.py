"""Helpers to upsert Cycle v4 image packs into data/open/manifest.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.open_higgs import sha256_file

ROOT_DEFAULT = Path(__file__).resolve().parents[2]
MANIFEST_DEFAULT = ROOT_DEFAULT / "data" / "open" / "manifest.json"


def upsert_manifest_dataset(manifest_path: Path, entry: dict[str, Any]) -> None:
    """Insert or replace a dataset entry by id; rewrite manifest JSON."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = list(payload.get("datasets", []))
    did = entry["id"]
    replaced = False
    for i, item in enumerate(datasets):
        if item.get("id") == did:
            datasets[i] = entry
            replaced = True
            break
    if not replaced:
        datasets.append(entry)
    payload["datasets"] = datasets
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_image_pack_manifest_entry(
    *,
    dataset_id: str,
    description: str,
    license_name: str,
    source_url: str,
    spatial_shape: list[Any],
    n_classes: int | None,
    modality: str,
    row_counts: dict[str, int],
    processed_rel: str,
    raw_rel: str,
    files: dict[str, str],
    build_script: str,
    root: Path = ROOT_DEFAULT,
) -> dict[str, Any]:
    """Build a ready image / image_text manifest row with checksums."""
    processed = root / "data" / "open" / processed_rel
    checksums: dict[str, str] = {}
    for key, filename in files.items():
        path = processed / filename
        if not path.is_file():
            msg = f"missing processed artifact for checksum: {path}"
            raise FileNotFoundError(msg)
        checksums[key] = sha256_file(path)
    if modality == "image_text":
        n_features = 0
    else:
        dims = [int(s) for s in spatial_shape if s is not None]
        n_features = int(np.prod(dims)) if dims else 0
    entry: dict[str, Any] = {
        "id": dataset_id,
        "path": processed_rel,
        "description": description,
        "license": license_name,
        "source_url": source_url,
        "spatial_shape": spatial_shape,
        "n_features": n_features,
        "modality": modality,
        "ready": True,
        "row_counts": row_counts,
        "files": files,
        "checksums": checksums,
        "build_script": build_script,
        "raw_path": raw_rel,
    }
    if n_classes is not None:
        entry["n_classes"] = n_classes
    return entry
