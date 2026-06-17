"""Unit tests for Zenodo release bundle preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.prepare_release import (
    RELEASE_VERSION,
    STATIC_ARTIFACTS,
    build_manifest_text,
    sha256_file,
    verify_manifest,
)


def test_release_version_matches_pyproject():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{RELEASE_VERSION}"' in pyproject


def test_static_artifacts_exist():
    for name in STATIC_ARTIFACTS:
        assert Path(name).is_file(), f"missing static release artifact: {name}"


def test_release_notes_mentions_current_version():
    notes = Path("RELEASE_NOTES.md").read_text(encoding="utf-8")
    assert f"v{RELEASE_VERSION}" in notes


def test_sha256_file(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_text("quantun-ia", encoding="utf-8")
    expected = hashlib.sha256(b"quantun-ia").hexdigest()
    assert sha256_file(sample) == expected


def test_build_manifest_text_includes_checksums(tmp_path: Path):
    artifact = tmp_path / "results.csv"
    artifact.write_text("exp_id,model_name\n", encoding="utf-8")
    digest = sha256_file(artifact)
    manifest = build_manifest_text([artifact], base_dir=tmp_path)
    assert f"results.csv sha256:{digest}" in manifest


def test_verify_manifest_accepts_valid_bundle(tmp_path: Path):
    artifact = tmp_path / "figures" / "holdout.pdf"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"%PDF-1.4")
    manifest_path = tmp_path / "MANIFEST.txt"
    manifest_path.write_text(build_manifest_text([artifact], base_dir=tmp_path), encoding="utf-8")
    assert verify_manifest(manifest_path, tmp_path) is True


def test_verify_manifest_rejects_tampered_file(tmp_path: Path):
    artifact = tmp_path / "results.csv"
    artifact.write_text("original", encoding="utf-8")
    manifest_path = tmp_path / "MANIFEST.txt"
    manifest_path.write_text(build_manifest_text([artifact], base_dir=tmp_path), encoding="utf-8")
    artifact.write_text("tampered", encoding="utf-8")
    assert verify_manifest(manifest_path, tmp_path) is False


def test_verify_manifest_rejects_missing_file(tmp_path: Path):
    manifest_path = tmp_path / "MANIFEST.txt"
    manifest_path.write_text("missing.csv sha256:deadbeef\n", encoding="utf-8")
    assert verify_manifest(manifest_path, tmp_path) is False
