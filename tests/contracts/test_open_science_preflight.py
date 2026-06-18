"""Contract tests for open-science preflight (Phase E — Zenodo/arXiv readiness)."""

from __future__ import annotations

from pathlib import Path

from scripts.open_science_preflight import PUBLICATION_JSONL, seed_publication_logs
from scripts.prepare_release import RELEASE_VERSION, verify_manifest

ROOT = Path(__file__).resolve().parents[2]


def test_publication_fixture_exists():
    assert PUBLICATION_JSONL.is_file()


def test_authors_md_exists():
    text = (ROOT / "AUTHORS.md").read_text(encoding="utf-8")
    assert "Alexandre Zanata" in text
    assert "ORCID" in text


def test_seed_publication_logs(tmp_path, monkeypatch):
    logs = tmp_path / "logs" / "experiments.jsonl"
    monkeypatch.setattr("scripts.open_science_preflight.LOGS_JSONL", logs)
    path = seed_publication_logs(force=True)
    assert path.is_file()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 4


def test_release_version_matches_pyproject():
    version_line = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{RELEASE_VERSION}"' in version_line


def test_release_bundle_includes_flagship_artifacts():
    """Requires prior `make open-science-preflight` or `make release` in dev/CI."""
    dist = ROOT / "dist" / "release"
    manifest = dist / "MANIFEST.txt"
    if not manifest.is_file():
        return
    assert verify_manifest(manifest, dist)
    assert (dist / "leaderboard" / "v1.json").is_file()
    assert (dist / "model_cards" / "quantum_nano_bc.md").is_file()
