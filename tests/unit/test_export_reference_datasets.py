"""Unit tests for Zenodo reference dataset export."""

from pathlib import Path

import pandas as pd

from scripts.export_reference_datasets import export_all, export_breast_cancer, export_circles


def test_export_breast_cancer_writes_csv_and_metadata(tmp_path: Path):
    csv_path = export_breast_cancer(tmp_path)
    assert csv_path.is_file()
    meta_path = tmp_path / "breast_cancer.meta.json"
    assert meta_path.is_file()
    frame = pd.read_csv(csv_path)
    assert frame.shape[0] == 569
    assert "label" in frame.columns
    assert len([c for c in frame.columns if c.startswith("feature_")]) == 30


def test_export_circles_writes_expected_shape(tmp_path: Path):
    csv_path = export_circles(tmp_path, n_samples=100, noise=0.2)
    frame = pd.read_csv(csv_path)
    assert frame.shape == (100, 3)
    meta = (tmp_path / "circles.meta.json").read_text(encoding="utf-8")
    assert "make_circles" in meta


def test_export_all_writes_both_datasets(tmp_path: Path):
    paths = export_all(tmp_path)
    assert len(paths) == 2
    assert (tmp_path / "breast_cancer.csv").is_file()
    assert (tmp_path / "circles.csv").is_file()
