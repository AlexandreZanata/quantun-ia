"""Unit tests for DVC local remote bootstrap (Phase 29)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.dvc_remote_setup import (
    dvc_module_available,
    read_remote_url,
    remote_configured,
    resolve_storage_path,
    setup_local_remote,
)


def test_resolve_storage_path_default_parent():
    root = Path("/tmp/quantun-ia")
    path = resolve_storage_path(None, root=root)
    assert path == root.parent / "quantun-ia-dvc-storage"


def test_remote_configured_false_when_missing(tmp_path: Path):
    (tmp_path / ".dvc").mkdir()
    assert remote_configured(tmp_path, "localstore") is False


def test_remote_configured_true_when_present(tmp_path: Path):
    dvc_dir = tmp_path / ".dvc"
    dvc_dir.mkdir()
    (dvc_dir / "config").write_text(
        "[core]\n"
        "    remote = localstore\n"
        "['remote \"localstore\"']\n"
        "    url = ../quantun-ia-dvc-storage\n",
        encoding="utf-8",
    )
    assert remote_configured(tmp_path, "localstore") is True
    assert read_remote_url(tmp_path, "localstore") == "../quantun-ia-dvc-storage"


def test_dvc_module_available():
    assert isinstance(dvc_module_available(), bool)


@pytest.mark.skipif(not dvc_module_available(), reason="dvc not installed in venv")
def test_setup_local_remote_creates_config(tmp_path: Path, monkeypatch):
    dvc_dir = tmp_path / ".dvc"
    dvc_dir.mkdir()
    (dvc_dir / "config").write_text("[core]\n    no_scm = True\n", encoding="utf-8")
    storage = tmp_path / "storage"
    monkeypatch.chdir(tmp_path)

    result = setup_local_remote(storage_path=storage, root=tmp_path)

    assert result["status"] in {"created", "already_configured"}
    assert storage.is_dir()
    assert remote_configured(tmp_path, "localstore")
