"""Contract tests for release tag evidence records (Phase 26)."""

from __future__ import annotations

from pathlib import Path

import yaml

RELEASE_RECORD_V0916 = Path("docs/releases/v0.9.16.md")
RELEASE_RECORD_V0922 = Path("docs/releases/v0.9.22.md")
RELEASE_RECORD_V100RC1 = Path("docs/releases/v1.0.0-rc1.md")
RELEASE_RECORD_V100 = Path("docs/releases/v1.0.0.md")
RELEASE_RECORD_V110 = Path("docs/releases/v1.1.0.md")
RELEASE_RECORD_V120 = Path("docs/releases/v1.2.0.md")
CITATION_LOOP = Path("docs/citation_loop.md")


def test_release_record_v0916_exists():
    assert RELEASE_RECORD_V0916.is_file(), "missing docs/releases/v0.9.16.md"


def test_release_record_v0922_exists():
    assert RELEASE_RECORD_V0922.is_file(), "missing docs/releases/v0.9.22.md"


def test_release_record_v100rc1_exists():
    assert RELEASE_RECORD_V100RC1.is_file(), "missing docs/releases/v1.0.0-rc1.md"


def test_release_record_v100_exists():
    assert RELEASE_RECORD_V100.is_file(), "missing docs/releases/v1.0.0.md"


def test_release_record_v110_exists():
    assert RELEASE_RECORD_V110.is_file(), "missing docs/releases/v1.1.0.md"


def test_release_record_v120_exists():
    assert RELEASE_RECORD_V120.is_file(), "missing docs/releases/v1.2.0.md"


def test_release_record_v110_documents_application_tracks():
    text = RELEASE_RECORD_V110.read_text(encoding="utf-8")
    assert "v1.1.0" in text
    assert "exp_027" in text or "continuous" in text.lower()
    assert "RTX 4060" in text


def test_release_record_v120_documents_phase_l():
    text = RELEASE_RECORD_V120.read_text(encoding="utf-8")
    assert "v1.2.0" in text
    assert "exp_032" in text or "LargeNanoMLP" in text
    assert "exp_033" in text or "serve" in text.lower()
    assert "RTX 4060" in text


def test_release_record_v100_documents_stable_release():
    text = RELEASE_RECORD_V100.read_text(encoding="utf-8")
    assert "v1.0.0" in text
    assert "predictions" in text.lower() or "inference" in text.lower()
    assert "RTX 4060" in text


def test_release_record_v100rc1_documents_phase_d():
    text = RELEASE_RECORD_V100RC1.read_text(encoding="utf-8")
    assert "v1.0.0-rc1" in text
    assert "phase-d-preflight" in text or "Phase D" in text
    assert "RTX 4060" in text


def test_release_record_documents_tag_and_bundle():
    text = RELEASE_RECORD_V0916.read_text(encoding="utf-8")
    assert "v0.9.16" in text
    assert "git tag" in text
    assert "citation-ready-full" in text or "make release" in text
    assert "65 artifacts" in text or "release bundle" in text.lower()


def test_citation_loop_documents_finalize_command():
    text = CITATION_LOOP.read_text(encoding="utf-8")
    assert "finalize-citation" in text


def test_arxiv_metadata_matches_released_version():
    data = yaml.safe_load(Path("paper/arxiv_metadata.yaml").read_text(encoding="utf-8"))
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    for line in pyproject.splitlines():
        if line.startswith("version = "):
            version = line.split("=", 1)[1].strip().strip('"')
            break
    else:
        raise AssertionError("pyproject version missing")
    assert data["software_version"] == version
