#!/usr/bin/env python3
"""Gera o manifesto de aquisição de pesos do Hugging Face.

Registra revisão imutável, licença, checksums LFS, tamanho de cada arquivo e uma
recontagem de parâmetros lida do header `safetensors`, sem carregar tensores.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_safetensors(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_length).decode("utf-8"))
    by_dtype: dict[str, int] = {}
    stored_total = 0
    tensor_count = 0
    data_end = 0
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        numel = 1
        for dimension in spec["shape"]:
            numel *= int(dimension)
        stored_total += numel
        tensor_count += 1
        by_dtype[spec["dtype"]] = by_dtype.get(spec["dtype"], 0) + numel
        data_end = max(data_end, int(spec["data_offsets"][1]))
    return {
        "stored_total": stored_total,
        "active_total": stored_total,
        "tensor_count": tensor_count,
        "by_dtype": dict(sorted(by_dtype.items())),
        "layout_consistent": 8 + header_length + data_end == file_size,
        "method": "soma de shapes do header safetensors, sem carregar tensores",
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_summary(root: Path) -> dict[str, Any] | None:
    config_path = root / "config.json"
    if not config_path.exists():
        return None
    config = read_json(config_path)
    keys = (
        "architectures",
        "model_type",
        "d_model",
        "d_ff",
        "num_layers",
        "num_decoder_layers",
        "num_heads",
        "vocab_size",
        "tie_word_embeddings",
        "torch_dtype",
    )
    return {key: config[key] for key in keys if key in config}


def lfs_checksums(api_snapshot: dict[str, Any]) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for item in api_snapshot.get("siblings", []):
        filename = item.get("rfilename")
        sha256 = (item.get("lfs") or {}).get("sha256")
        if filename and sha256:
            checksums[filename] = sha256
    return checksums


def write_metadata_snapshot(source: Path, destination: Path) -> None:
    raw = read_json(source)
    card = raw.get("cardData", {}) or {}
    snapshot = {
        "repo": raw.get("id") or raw.get("modelId"),
        "sha": raw.get("sha"),
        "lastModified": raw.get("lastModified"),
        "license": card.get("license"),
        "tags": raw.get("tags"),
        "downloads": raw.get("downloads"),
        "files": [
            {
                "rfilename": item.get("rfilename"),
                "size": (item.get("lfs") or {}).get("size", item.get("size")),
                "sha256": (item.get("lfs") or {}).get("sha256"),
            }
            for item in raw.get("siblings", [])
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    api_path = Path(args.api_snapshot).resolve()
    api = read_json(api_path)
    expected = lfs_checksums(api)
    if not expected:
        raise ValueError("snapshot da API sem checksums LFS; verificação impossível")
    files = []
    checksums_verified = True
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        observed = sha256_file(path)
        lfs = expected.get(relative)
        match = None if lfs is None else observed == lfs
        if match is False:
            checksums_verified = False
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256_observed": observed,
                "lfs_sha256": lfs,
                "checksum_match": match,
                "is_lfs": lfs is not None,
            }
        )
    safetensors_path = root / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"safetensors ausente em {root}")
    parameters = inspect_safetensors(safetensors_path)
    config = config_summary(root)
    parameters["tied_word_embeddings"] = bool((config or {}).get("tie_word_embeddings", False))
    parameters["config"] = config
    revision_sha = api.get("sha")
    return {
        "schema_version": 1,
        "kind": "weights",
        "id": args.id,
        "acquired_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "repo": args.repo,
            "url": f"https://huggingface.co/{args.repo}",
            "api_snapshot": f"research/lean/acquisitions/{Path(args.metadata_snapshot_out).name}",
            "api_response_sha256": sha256_file(api_path),
        },
        "version_pin": {
            "revision_sha": revision_sha,
            "last_modified": api.get("lastModified"),
            "resolved_via": "Hugging Face API ?blobs=true",
        },
        "license": {
            "id": (api.get("cardData", {}) or {}).get("license", "").upper() or None,
            "evidence": "cardData.license e tag license:* do repositório",
            "source": f"https://huggingface.co/{args.repo}",
        },
        "role": args.role,
        "usage_policy": {
            "training": False,
            "publish": False,
            "paid_service": False,
            "sealed_tests_touched": False,
        },
        "files": files,
        "checksums_verified": checksums_verified,
        "parameters": parameters,
        "local_path": str(root.relative_to(ROOT)),
        "status": "acquired_verified" if checksums_verified else "checksum_mismatch",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Manifesto de aquisição de pesos Hugging Face")
    parser.add_argument("--id", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--api-snapshot", required=True)
    parser.add_argument("--metadata-snapshot-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--role", required=True)
    args = parser.parse_args()
    write_metadata_snapshot(Path(args.api_snapshot), Path(args.metadata_snapshot_out))
    manifest = build_manifest(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifesto gravado: {args.out}")
    print(f"checksums_verified: {manifest['checksums_verified']}")
    print(f"parâmetros: {manifest['parameters']['stored_total']:,}")
    return 0 if manifest["checksums_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
