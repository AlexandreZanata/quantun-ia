#!/usr/bin/env python3
"""Valida os registros estáticos do programa Lean Tiny Prover."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAN_RESEARCH = ROOT / "research" / "lean"
LEAN_ACQUISITIONS = LEAN_RESEARCH / "acquisitions"
REQUIRED_HYPOTHESIS_FIELDS = {
    "id", "family_code", "family", "title", "intervention", "rationale",
    "falsification_gate", "cost_tier", "parameter_target", "status", "preregistered",
}
REQUIRED_ACQUISITION_FIELDS = {"id", "source", "version_pin", "license", "archive", "splits", "contamination_report"}
ACQUISITION_ARCHIVE_FIELDS = {"filename", "size_bytes", "md5_published", "md5_observed", "sha256_observed", "local_path"}
MD5_HEX = re.compile(r"^[0-9a-f]{32}$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def load_json(name: str) -> dict:
    return json.loads((LEAN_RESEARCH / name).read_text(encoding="utf-8"))


def load_acquisitions() -> list[dict]:
    if not LEAN_ACQUISITIONS.exists():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(LEAN_ACQUISITIONS.glob("*.json"))
        if not path.name.endswith(".metadata.json")
    ]


def validate_acquisitions() -> list[str]:
    errors: list[str] = []
    for acquisition in load_acquisitions():
        identifier = acquisition.get("id", "<sem-id>")
        missing = REQUIRED_ACQUISITION_FIELDS - set(acquisition)
        if missing:
            errors.append(f"aquisição {identifier}: campos ausentes {sorted(missing)}")
            continue
        archive = acquisition["archive"]
        archive_missing = ACQUISITION_ARCHIVE_FIELDS - set(archive)
        if archive_missing:
            errors.append(f"aquisição {identifier}: archive sem campos {sorted(archive_missing)}")
            continue
        if archive["md5_published"] != archive["md5_observed"]:
            errors.append(f"aquisição {identifier}: md5 divergente do publicado")
        if archive.get("checksum_verified") is not True:
            errors.append(f"aquisição {identifier}: checksum_verified deveria ser true")
        if not MD5_HEX.match(archive["md5_observed"]):
            errors.append(f"aquisição {identifier}: md5 observado inválido")
        if not SHA256_HEX.match(archive["sha256_observed"]):
            errors.append(f"aquisição {identifier}: sha256 observado inválido")
        if not archive["local_path"].startswith(("data/raw/", ".local/")):
            errors.append(f"aquisição {identifier}: dados devem ficar em data/raw/ ou .local/")
        if not acquisition["splits"]:
            errors.append(f"aquisição {identifier}: nenhum split registrado")
        for split in acquisition["splits"]:
            test_entry = split.get("test", {})
            if test_entry.get("content_inspected") is not False:
                errors.append(f"aquisição {identifier}: test de {split.get('name')} não deveria ser inspecionado")
        report = acquisition["contamination_report"]
        if report.get("sealed_sets_touched") != []:
            errors.append(f"aquisição {identifier}: nenhum conjunto selado pode ter sido tocado")
    return errors


def validate() -> list[str]:
    errors: list[str] = []
    registry = load_json("hypotheses.json")
    hypotheses = registry.get("hypotheses", [])
    if len(hypotheses) != 100:
        errors.append(f"esperadas 100 hipóteses, encontradas {len(hypotheses)}")

    expected_ids = [f"LH-{index:03d}" for index in range(1, 101)]
    actual_ids = [item.get("id") for item in hypotheses]
    if actual_ids != expected_ids:
        errors.append("IDs devem ser únicos, ordenados e cobrir LH-001…LH-100")

    allowed_status = set(registry.get("status_vocabulary", []))
    allowed_tiers = set(registry.get("cost_tiers", {}))
    for item in hypotheses:
        missing = REQUIRED_HYPOTHESIS_FIELDS - set(item)
        if missing:
            errors.append(f"{item.get('id', '<sem-id>')}: campos ausentes {sorted(missing)}")
        if item.get("status") not in allowed_status:
            errors.append(f"{item.get('id')}: status inválido")
        if item.get("cost_tier") not in allowed_tiers:
            errors.append(f"{item.get('id')}: cost_tier inválido")
        if not item.get("falsification_gate"):
            errors.append(f"{item.get('id')}: gate de falsificação vazio")

    family_counts: dict[str, int] = {}
    for item in hypotheses:
        code = item.get("family_code", "")
        family_counts[code] = family_counts.get(code, 0) + 1
    if len(family_counts) != 10 or any(count != 10 for count in family_counts.values()):
        errors.append(f"esperadas 10 famílias com 10 hipóteses: {family_counts}")

    datasets = load_json("datasets.json").get("datasets", [])
    if not datasets:
        errors.append("registro de datasets vazio")
    for dataset in datasets:
        for field in ("id", "role", "source", "version_pin", "license", "status"):
            if not dataset.get(field):
                errors.append(f"dataset {dataset.get('id', '<sem-id>')}: {field} vazio")

    baselines = load_json("baselines.json").get("baselines", [])
    if not baselines:
        errors.append("registro de baselines vazio")
    if not any(item.get("consolidated") and item.get("kind") == "micro" for item in baselines):
        errors.append("falta baseline micro consolidado")
    if not any(item.get("parameters_nominal") == 0 for item in baselines):
        errors.append("falta baseline simbólico de zero parâmetros")

    errors.extend(validate_acquisitions())

    return errors


def main() -> int:
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERRO: {error}")
        return 1
    acquisitions = load_acquisitions()
    print(
        "OK: 100 hipóteses, datasets, baselines e "
        f"{len(acquisitions)} aquisição(ões) estruturalmente válidos"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

