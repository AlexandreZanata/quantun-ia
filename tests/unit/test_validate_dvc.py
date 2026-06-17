"""Unit tests for DVC pipeline validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.validate_dvc import (
    EXPECTED_STAGES,
    collect_dvc_issues,
    validate_dvc,
)


def test_collect_dvc_issues_empty_in_repo():
    root = Path(__file__).resolve().parents[2]
    issues = collect_dvc_issues(root=root)
    blocking = [i for i in issues if "informational" not in i]
    assert blocking == []


def test_collect_dvc_issues_detects_missing_stage(tmp_path: Path):
    dvc = {
        "stages": {
            "export_results": {
                "cmd": "python scripts/export_results.py",
                "deps": ["scripts/export_results.py"],
                "outs": ["data/exports/results.csv"],
            }
        }
    }
    (tmp_path / "dvc.yaml").write_text(yaml.safe_dump(dvc), encoding="utf-8")
    (tmp_path / ".dvc").mkdir()
    (tmp_path / ".dvc" / "config.example").write_text("[core]\n", encoding="utf-8")
    issues = collect_dvc_issues(root=tmp_path)
    assert any("figures" in i for i in issues)


def test_validate_dvc_passes_in_repo():
    ok, issues = validate_dvc()
    assert ok, issues


def test_expected_stages_constant():
    assert "export_results" in EXPECTED_STAGES
    assert len(EXPECTED_STAGES) == 3
