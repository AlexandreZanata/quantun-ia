#!/usr/bin/env python3
"""Gera o manifesto de aquisição de um dataset Lean a partir do arquivo baixado.

O manifesto registra fonte, DOI, revisão, licença, checksums, tamanho, papéis
dos splits e relatório de contaminação. Arquivos de teste entram apenas com
tamanho e sha256; o conteúdo nunca é inspecionado.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import mmap
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CROSS_BENCHMARK_PATTERNS = ("minif2f", "proofnet", "putnambench", "minictx")
TEST_FILE = "test.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_records(path: Path) -> int:
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            return mapped.read().count(b'"traced_tactics"')


def collect_files(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        entry: dict[str, Any] = {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.name in ("train.json", "val.json"):
            entry["record_count"] = count_records(path)
            entry["content_inspected"] = True
        elif path.name == TEST_FILE:
            entry["record_count"] = None
            entry["content_inspected"] = False
        entries.append(entry)
    return entries


def detect_splits(root: Path) -> list[str]:
    return sorted(
        directory.name
        for directory in root.iterdir()
        if directory.is_dir() and (directory / TEST_FILE).exists()
    )


SKIP_PARTS = {".git", ".venv", ".lake", "archive", "node_modules"}


def find_local_benchmark_files() -> dict[str, bool]:
    found = {pattern: False for pattern in CROSS_BENCHMARK_PATTERNS}
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_PARTS.intersection(path.parts):
            continue
        name = path.name.lower()
        for pattern in CROSS_BENCHMARK_PATTERNS:
            if pattern in name:
                found[pattern] = True
    return found


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    archive = Path(args.archive).resolve()
    zenodo = read_json(Path(args.zenodo_metadata).resolve())
    files = collect_files(root)
    by_path = {entry["path"]: entry for entry in files}
    metadata = read_json(root / "metadata.json")
    datasets = read_json(ROOT / "research" / "lean" / "datasets.json")["datasets"]
    dataset_status = {item["id"]: item["status"] for item in datasets}
    local_files = find_local_benchmark_files()
    split_names = detect_splits(root)
    splits = []
    for name in split_names:
        split: dict[str, Any] = {"name": name, "role": "training_and_development"}
        for part in ("train.json", "val.json", TEST_FILE):
            key = f"{name}/{part}"
            entry = dict(by_path[key])
            if part == TEST_FILE:
                entry["sealed_for"] = "seleção de arquitetura, hiperparâmetros e hipóteses"
            split[part.replace(".json", "")] = entry
        splits.append(split)
    archive_entry = {
        "filename": archive.name,
        "size_bytes": archive.stat().st_size,
        "md5_published": archive_entry_checksum(zenodo, archive.name, "md5"),
        "md5_observed": md5_file(archive),
        "sha256_observed": sha256_file(archive),
        "local_path": str(archive.relative_to(ROOT)),
    }
    archive_entry["checksum_verified"] = (
        archive_entry["md5_published"] == archive_entry["md5_observed"]
    )
    cross_benchmark = []
    for dataset in datasets:
        pattern = next(
            (item for item in CROSS_BENCHMARK_PATTERNS if item in dataset["id"].lower()),
            None,
        )
        if pattern is None:
            continue
        cross_benchmark.append(
            {
                "dataset_id": dataset["id"],
                "registry_status": dataset_status.get(dataset["id"], "unknown"),
                "local_files_found": local_files[pattern],
                "overlap_audit": "pending_until_acquisition",
            }
        )
    return {
        "schema_version": 1,
        "kind": "dataset",
        "id": args.id,
        "acquired_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "record": "https://zenodo.org/records/10114157",
            "doi": zenodo.get("doi"),
            "concept_doi": zenodo.get("conceptdoi"),
            "api_snapshot": f"research/lean/acquisitions/{Path(args.metadata_snapshot_out).name}",
            "api_response_sha256": sha256_file(Path(args.zenodo_metadata)),
        },
        "version_pin": {
            "zenodo_revision": zenodo.get("revision"),
            "zenodo_created": zenodo.get("created"),
            "leandojo_version": metadata.get("leandojo_version"),
            "mathlib_commit": metadata.get("from_repo", {}).get("commit"),
            "mathlib_url": metadata.get("from_repo", {}).get("url"),
        },
        "license": {
            "dataset": "CC BY 2.0",
            "source": "https://creativecommons.org/licenses/by/2.0/",
            "bundled_license_files": [
                entry["path"] for entry in files if entry["path"].startswith("licenses/")
            ],
            "note": "Licenças transitivas de mathlib e Lean incluídas pelo próprio dataset.",
        },
        "archive": archive_entry,
        "extracted_root": str(root.relative_to(ROOT)),
        "files": files,
        "splits": splits,
        "contamination_report": {
            "version": 1,
            "method": "Proveniência pinada, contagem de registros em treino/val e verificação de presença local dos demais benchmarks; deduplicação por hash de afirmação fica para LTP-03/LTP-04.",
            "sealed_sets_touched": [],
            "test_files": [
                {
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                    "content_inspected": False,
                }
                for entry in files
                if entry["path"].endswith(TEST_FILE)
            ],
            "cross_benchmark": cross_benchmark,
            "open_items": [
                "Deduplicação por afirmação contra miniF2F após aquisição.",
                "Deduplicação por afirmação contra ProofNet Lean 4 após auditoria de licença.",
                "Deduplicação por afirmação contra miniCTX v2 e PutnamBench após aquisição.",
                "Verificação de que nenhum arquivo de teste selado foi usado na seleção.",
            ],
        },
        "status": "acquired_verified",
    }


def archive_entry_checksum(zenodo: dict[str, Any], filename: str, algorithm: str) -> str | None:
    for item in zenodo.get("files", []):
        if item.get("key") != filename:
            continue
        checksum = item.get("checksum", "")
        if checksum.startswith(f"{algorithm}:"):
            return checksum.split(":", 1)[1]
    return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_metadata_snapshot(source: Path, destination: Path) -> None:
    raw = read_json(source)
    metadata = raw.get("metadata", {})
    snapshot = {
        "record": "https://zenodo.org/records/10114157",
        "id": raw.get("id"),
        "doi": raw.get("doi"),
        "conceptdoi": raw.get("conceptdoi"),
        "revision": raw.get("revision"),
        "created": raw.get("created"),
        "updated": raw.get("updated"),
        "title": raw.get("title"),
        "license": metadata.get("license"),
        "creators": [creator.get("name") for creator in metadata.get("creators", [])],
        "description": re.sub(r"<[^>]+>", " ", metadata.get("description", ""))[:2000],
        "files": [
            {"key": item.get("key"), "size": item.get("size"), "checksum": item.get("checksum")}
            for item in raw.get("files", [])
        ],
    }
    write_json(destination, snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description="Constrói manifesto de aquisição de dataset Lean")
    parser.add_argument("--id", required=True)
    parser.add_argument("--root", required=True, help="diretório extraído do dataset")
    parser.add_argument("--archive", required=True, help="arquivo baixado")
    parser.add_argument("--zenodo-metadata", required=True, help="snapshot JSON da API do Zenodo")
    parser.add_argument("--metadata-snapshot-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_metadata_snapshot(Path(args.zenodo_metadata), Path(args.metadata_snapshot_out))
    manifest = build_manifest(args)
    write_json(Path(args.out), manifest)
    print(f"manifesto gravado: {args.out}")
    print(f"checksum verificado: {manifest['archive']['checksum_verified']}")
    return 0 if manifest["archive"]["checksum_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
