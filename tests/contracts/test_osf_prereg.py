"""Contract tests — OSF pre-registration links in hypothesis.md (exp_021+)."""

from __future__ import annotations

import re
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"

# Experiments that require a filed OSF link before publication-profile citation.
PREREG_REQUIRED = {
    "exp_021_qml_backend_parity",
    "exp_022_nano_quantum_parity",
}

OSF_URL_PATTERN = re.compile(r"https://osf\.io/[a-z0-9]+", re.IGNORECASE)
PLANNED_ONLY = re.compile(r"OSF entry planned|planned before publication", re.IGNORECASE)


def test_publication_experiments_have_osf_link_in_hypothesis():
    paths = sorted(EXPERIMENTS_DIR.glob("exp_*/hypothesis.md"))
    assert paths, "expected at least one experiment hypothesis.md"
    checked = 0
    for path in paths:
        folder = path.parent.name
        if folder not in PREREG_REQUIRED:
            continue
        checked += 1
        text = path.read_text(encoding="utf-8")
        assert OSF_URL_PATTERN.search(text), (
            f"{path} must include an https://osf.io/... pre-registration URL "
            f"(required for {folder})"
        )
        assert not PLANNED_ONLY.search(text), (
            f"{path} still says OSF is 'planned' — file the registration and paste the URL"
        )
    assert checked == len(PREREG_REQUIRED), "PREREG_REQUIRED folders missing hypothesis.md"
