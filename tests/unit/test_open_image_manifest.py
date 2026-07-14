"""Unit tests for Cycle v4 open image manifest helpers."""

from pathlib import Path

from src.data.open_image_manifest import build_image_pack_manifest_entry, upsert_manifest_dataset


def test_upsert_manifest_dataset(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema_version":"1","datasets":[]}\n', encoding="utf-8")
    proc = tmp_path / "data" / "open" / "images" / "stl10" / "processed" / "v1"
    proc.mkdir(parents=True)
    (proc / "stats.json").write_text('{"ok":true}\n', encoding="utf-8")
    (proc / "split_indices.npz").write_bytes(b"0123456789abcdef")
    # Use ROOT pointing at tmp_path via processed relative path under it
    from src.data.open_image_manifest import ROOT_DEFAULT

    entry = {
        "id": "demo_pack_v1",
        "path": "images/stl10/processed/v1",
        "ready": True,
        "files": {"stats": "stats.json"},
        "checksums": {"stats": "a" * 64},
    }
    upsert_manifest_dataset(manifest, entry)
    text = manifest.read_text(encoding="utf-8")
    assert "demo_pack_v1" in text
    assert ROOT_DEFAULT.is_dir()
    assert callable(build_image_pack_manifest_entry)
