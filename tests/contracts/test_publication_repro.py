"""Contract tests for Phase C publication reproduction pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard.benchmark_data import latest_multi_seed_summaries
from scripts.export_latex_tables import _load_summaries, summary_to_latex

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION_FIXTURE = ROOT / "tests" / "contracts" / "fixtures" / "publication_experiments.jsonl"
REPRO_SCRIPT = ROOT / "scripts" / "repro_publication_ci.sh"


def test_publication_fixture_exists():
    assert PUBLICATION_FIXTURE.is_file()


def test_publication_fixture_includes_exp_024_flagship():
    records = [json.loads(line) for line in PUBLICATION_FIXTURE.read_text().splitlines() if line.strip()]
    exp_ids = {rec.get("exp_id") for rec in records}
    assert "exp_024" in exp_ids
    summaries = latest_multi_seed_summaries(records)
    assert "exp_024" in summaries
    assert "hybrid_sandwich" in summaries["exp_024"]
    assert "logistic_regression" in summaries["exp_024"]


def test_publication_fixture_generates_exp_024_latex_table():
    summaries = _load_summaries(PUBLICATION_FIXTURE)
    assert "exp_024" in summaries
    latex = summary_to_latex("exp_024", summaries["exp_024"])
    assert "hybrid\\_sandwich" in latex
    assert "logistic\\_regression" in latex
    assert "\\begin{table}" in latex


def test_repro_publication_script_exists_and_executable():
    assert REPRO_SCRIPT.is_file()
    assert REPRO_SCRIPT.stat().st_mode & 0o111, "repro_publication_ci.sh must be executable"


@pytest.mark.parametrize(
    "rel_path",
    [
        "model_cards/quantum_nano_bc.md",
        "docs/compute_environment.md",
    ],
)
def test_publication_bundle_docs_exist(rel_path: str):
    assert (ROOT / rel_path).is_file()
