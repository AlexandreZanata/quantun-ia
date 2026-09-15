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
REQUIRED_ACQUISITION_FIELDS = {"kind", "id", "source", "version_pin", "license"}
REQUIRED_DATASET_FIELDS = {"archive", "splits", "contamination_report"}
REQUIRED_WEIGHTS_FIELDS = {"files", "parameters", "checksums_verified", "local_path"}
ACQUISITION_ARCHIVE_FIELDS = {"filename", "size_bytes", "md5_published", "md5_observed", "sha256_observed", "local_path"}
ACQUISITION_KINDS = {"dataset", "weights"}
MD5_HEX = re.compile(r"^[0-9a-f]{32}$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
REVISION_HEX = re.compile(r"^[0-9a-f]{40}$")


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


def _validate_dataset_acquisition(identifier: str, acquisition: dict) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_DATASET_FIELDS - set(acquisition)
    if missing:
        errors.append(f"aquisição {identifier}: campos de dataset ausentes {sorted(missing)}")
        return errors
    archive = acquisition["archive"]
    archive_missing = ACQUISITION_ARCHIVE_FIELDS - set(archive)
    if archive_missing:
        errors.append(f"aquisição {identifier}: archive sem campos {sorted(archive_missing)}")
        return errors
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


def _validate_weights_acquisition(identifier: str, acquisition: dict) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_WEIGHTS_FIELDS - set(acquisition)
    if missing:
        errors.append(f"aquisição {identifier}: campos de pesos ausentes {sorted(missing)}")
        return errors
    revision = acquisition["version_pin"].get("revision_sha", "")
    if not REVISION_HEX.match(revision):
        errors.append(f"aquisição {identifier}: revisão deve ser um commit sha de 40 hex")
    if not acquisition["license"].get("id"):
        errors.append(f"aquisição {identifier}: licença sem identificador")
    if acquisition["checksums_verified"] is not True:
        errors.append(f"aquisição {identifier}: checksums_verified deveria ser true")
    files = acquisition["files"]
    if not files:
        errors.append(f"aquisição {identifier}: nenhum arquivo registrado")
    lfs_files = [item for item in files if item.get("is_lfs")]
    if not lfs_files:
        errors.append(f"aquisição {identifier}: nenhum arquivo LFS verificado")
    for item in files:
        if not SHA256_HEX.match(item.get("sha256_observed", "")):
            errors.append(f"aquisição {identifier}: sha256 inválido em {item.get('path')}")
        if item.get("is_lfs"):
            if item.get("lfs_sha256") != item.get("sha256_observed"):
                errors.append(f"aquisição {identifier}: sha256 divergente do LFS em {item.get('path')}")
            if item.get("checksum_match") is not True:
                errors.append(f"aquisição {identifier}: checksum_match deveria ser true em {item.get('path')}")
    parameters = acquisition["parameters"]
    if parameters.get("stored_total", 0) <= 0 or parameters.get("active_total", 0) <= 0:
        errors.append(f"aquisição {identifier}: recontagem de parâmetros vazia")
    if not parameters.get("method"):
        errors.append(f"aquisição {identifier}: método de recontagem ausente")
    if not acquisition["local_path"].startswith(("data/raw/", ".local/")):
        errors.append(f"aquisição {identifier}: dados devem ficar em data/raw/ ou .local/")
    if acquisition["usage_policy"].get("sealed_tests_touched") is not False:
        errors.append(f"aquisição {identifier}: nenhum conjunto selado pode ter sido tocado")
    return errors


def validate_acquisitions() -> list[str]:
    errors: list[str] = []
    for acquisition in load_acquisitions():
        identifier = acquisition.get("id", "<sem-id>")
        missing = REQUIRED_ACQUISITION_FIELDS - set(acquisition)
        if missing:
            errors.append(f"aquisição {identifier}: campos ausentes {sorted(missing)}")
            continue
        kind = acquisition["kind"]
        if kind not in ACQUISITION_KINDS:
            errors.append(f"aquisição {identifier}: kind inválido: {kind}")
            continue
        if kind == "dataset":
            errors.extend(_validate_dataset_acquisition(identifier, acquisition))
        else:
            errors.extend(_validate_weights_acquisition(identifier, acquisition))
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

