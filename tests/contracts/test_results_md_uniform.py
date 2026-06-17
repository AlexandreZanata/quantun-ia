"""Contract tests — uniform results.md structure for publication experiments."""

from pathlib import Path

import pytest

from src.training.results_writer import REQUIRED_SECTIONS

EXP_RESULTS = [
    "experiments/exp_011_uci_tabular_qml/results.md",
    "experiments/exp_012_mnist_pca_qml/results.md",
    "experiments/exp_013_augmentation_robustness/results.md",
    "experiments/exp_014_sequence_baselines/results.md",
    "experiments/exp_015_adaptive_qnn/results.md",
    "experiments/exp_016_hybrid_nas/results.md",
    "experiments/exp_017_poison_topology/results.md",
    "experiments/exp_018_feature_fusion/results.md",
    "experiments/exp_021_qml_backend_parity/results.md",
    "experiments/exp_022_nano_quantum_parity/results.md",
]


@pytest.mark.parametrize("results_path", EXP_RESULTS)
def test_results_md_has_required_sections(results_path: str):
    path = Path(results_path)
    if not path.is_file():
        pytest.skip(f"missing {results_path}")
    text = path.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in text, f"{results_path} missing {section}"


@pytest.mark.parametrize("results_path", EXP_RESULTS)
def test_results_md_cohens_d_includes_magnitude(results_path: str):
    path = Path(results_path)
    if not path.is_file():
        pytest.skip(f"missing {results_path}")
    text = path.read_text(encoding="utf-8")
    if "## Paired Wilcoxon" not in text:
        pytest.skip("no paired comparisons")
    assert "(negligible)" in text or "(small)" in text or "(medium)" in text or "(large)" in text
