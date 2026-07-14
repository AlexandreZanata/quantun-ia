"""Unit tests for Cycle v3 paper table export."""

from pathlib import Path

from src.training.cycle_v3_tables import export_cycle_v3_tables, load_cycle_v3_registry

ROOT = Path(__file__).resolve().parents[2]


def test_export_cycle_v3_tables(tmp_path: Path):
    registry = load_cycle_v3_registry()
    assert "image_nano_i2i" in registry
    written = export_cycle_v3_tables(out_dir=tmp_path)
    assert (tmp_path / "image_nano_i2i.tex").is_file()
    assert (tmp_path / "image_nano_t2i.tex").is_file()
    assert (tmp_path / "quantum_v3.tex").is_file()
    assert "tab:image_nano_i2i" in written["image_nano_i2i.tex"].read_text(encoding="utf-8")
    assert ROOT.exists()
