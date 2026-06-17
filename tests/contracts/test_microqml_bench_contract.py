"""Contract tests for MicroQML Bench v1 export schema."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from src.benchmark.microqml_bench import build_bench_export, load_bench_config
from tests.contracts.microqml_bench_schema import MICROQML_BENCH_V1_SCHEMA

FIXTURE = Path(__file__).parent / "fixtures" / "sample_microqml_bench_v1.json"


def test_bench_config_loads_primary_tasks():
    config = load_bench_config()
    assert config["bench_id"] == "microqml_bench"
    assert config["schema_version"] == "1"
    categories = {task["category"] for task in config["tasks"]}
    assert "tabular" in categories
    assert "sequence" in categories


def test_sample_fixture_validates_against_schema():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    jsonschema.validate(instance=data, schema=MICROQML_BENCH_V1_SCHEMA)


def test_build_export_validates_against_schema():
    export = build_bench_export(records=[])
    jsonschema.validate(instance=export, schema=MICROQML_BENCH_V1_SCHEMA)
    assert export["bench_id"] == "microqml_bench"
    assert isinstance(export["tasks"], list)
    assert export["leaderboard"] == []


def test_build_export_with_fixture_records():
    sample_jsonl = Path(__file__).parent / "fixtures" / "sample_experiments.jsonl"
    records = [
        json.loads(line)
        for line in sample_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    export = build_bench_export(records=records)
    jsonschema.validate(instance=export, schema=MICROQML_BENCH_V1_SCHEMA)
