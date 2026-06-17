"""Contract tests — validate JSONL experiment log records against schema."""

import json
from pathlib import Path

import jsonschema
import pytest

from tests.contracts.jsonl_schema import schema_for_record

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_JSONL = FIXTURES / "sample_experiments.jsonl"


@pytest.fixture
def sample_records() -> list[dict]:
    records = []
    with open(SAMPLE_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def test_sample_fixture_has_multiple_record_types(sample_records):
    types = {r.get("record_type") for r in sample_records}
    assert "multi_seed_summary" in types
    assert "paired_comparison_batch" in types
    assert None in types or "applicability_gate" in types


def test_every_sample_line_validates_against_schema(sample_records):
    for i, record in enumerate(sample_records):
        schema = schema_for_record(record)
        jsonschema.validate(instance=record, schema=schema)


def test_invalid_record_rejected():
    bad = {"model_name": "orphan"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema_for_record(bad))


def test_live_jsonl_if_present():
    log_path = Path("logs/experiments.jsonl")
    if not log_path.exists():
        pytest.skip("no logs/experiments.jsonl")
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        jsonschema.validate(instance=record, schema=schema_for_record(record))
