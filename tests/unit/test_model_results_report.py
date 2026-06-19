"""Unit tests — consolidated model results report."""

from __future__ import annotations

import json
from pathlib import Path

from src.application.model_results_report import (
    DEFAULT_REPORT_PATH,
    build_model_results_report,
    load_model_results_report,
    write_model_results_report,
)


def test_report_schema_keys():
    report = build_model_results_report(n_rows=64, include_benchmark=False)
    assert report["report_type"] == "model_results_summary"
    assert "serve_models" in report
    assert "human_clinic_scenarios" in report
    assert "hardware" in report
    assert isinstance(report["serve_models"], list)


def test_write_and_load_roundtrip(tmp_path: Path):
    out = tmp_path / "model_results_summary.json"
    write_model_results_report(out, n_rows=64)
    assert out.is_file()
    loaded = load_model_results_report(out)
    assert loaded is not None
    assert loaded["report_type"] == "model_results_summary"
    json.loads(out.read_text(encoding="utf-8"))


def test_default_report_path_under_logs():
    assert DEFAULT_REPORT_PATH.name == "model_results_summary.json"
    assert DEFAULT_REPORT_PATH.parent.name == "logs"
