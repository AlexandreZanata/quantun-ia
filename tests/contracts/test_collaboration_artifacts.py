"""Contract tests for collaboration and artifact-evaluation assets (Phase 23)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

CODEOWNERS = ROOT / ".github" / "CODEOWNERS"
ISSUE_TEMPLATES = ROOT / ".github" / "ISSUE_TEMPLATE"
REVIEWER_GUIDE = ROOT / "docs" / "reviewer_guide.md"
REVIEWER_REPRO = ROOT / "scripts" / "reviewer_repro.sh"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"


def test_codeowners_exists():
    assert CODEOWNERS.is_file(), "missing .github/CODEOWNERS"


def test_codeowners_has_default_owner():
    text = CODEOWNERS.read_text(encoding="utf-8")
    assert "AlexandreZanata" in text or "@AlexandreZanata" in text


def test_issue_templates_exist():
    expected = ("bug_report.yml", "experiment_replication.yml", "config.yml")
    for name in expected:
        path = ISSUE_TEMPLATES / name
        assert path.is_file(), f"missing issue template: {name}"


def test_experiment_replication_template_fields():
    data = yaml.safe_load((ISSUE_TEMPLATES / "experiment_replication.yml").read_text(encoding="utf-8"))
    assert data.get("name")
    body = data.get("body", [])
    field_ids = {item.get("id") for item in body if isinstance(item, dict)}
    assert "exp_id" in field_ids
    assert "environment" in field_ids


def test_reviewer_guide_exists():
    assert REVIEWER_GUIDE.is_file()
    text = REVIEWER_GUIDE.read_text(encoding="utf-8")
    assert "make repro" in text
    assert "make reviewer-repro" in text


def test_reviewer_repro_script_exists_and_executable():
    assert REVIEWER_REPRO.is_file()
    assert REVIEWER_REPRO.stat().st_mode & 0o111, "reviewer_repro.sh must be executable"


def test_contributing_mentions_replication_challenge():
    text = CONTRIBUTING.read_text(encoding="utf-8")
    assert "replication" in text.lower()
