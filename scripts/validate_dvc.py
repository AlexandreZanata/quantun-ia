#!/usr/bin/env python3
"""Validate DVC pipeline definition and optional remote configuration."""

from __future__ import annotations

import argparse
import configparser
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DVC_YAML = ROOT / "dvc.yaml"
DVC_DIR = ROOT / ".dvc"
DVC_CONFIG_EXAMPLE = DVC_DIR / "config.example"
DVC_CONFIG = DVC_DIR / "config"

EXPECTED_STAGES = frozenset({"export_results", "figures", "latex_tables"})

STAGE_SCRIPTS = {
    "export_results": "scripts/export_results.py",
    "figures": "scripts/generate_figures.py",
    "latex_tables": "scripts/export_latex_tables.py",
}


def dvc_cli_available() -> bool:
    if shutil.which("dvc"):
        return True
    try:
        subprocess.run(
            [sys.executable, "-m", "dvc", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _load_stages(root: Path) -> dict:
    path = root / "dvc.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    stages = data.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("dvc.yaml missing stages map")
    return stages


def collect_dvc_issues(root: Path = ROOT) -> list[str]:
    """Return DVC validation issues (informational suffix skips blocking)."""
    issues: list[str] = []

    if not (root / ".dvc").is_dir():
        issues.append("missing .dvc/ directory")
        return issues

    if not DVC_CONFIG_EXAMPLE.is_file():
        issues.append("missing .dvc/config.example")

    try:
        stages = _load_stages(root)
    except (FileNotFoundError, ValueError) as exc:
        issues.append(str(exc))
        return issues

    missing_stages = EXPECTED_STAGES - stages.keys()
    if missing_stages:
        issues.append(f"dvc.yaml missing stages: {sorted(missing_stages)}")

    for stage_name, script_rel in STAGE_SCRIPTS.items():
        if stage_name in stages and not (root / script_rel).is_file():
            issues.append(f"missing script for {stage_name}: {script_rel}")

    if not dvc_cli_available():
        issues.append(
            "dvc CLI not available — run make dvc-setup (installs via requirements-dev.txt) — informational"
        )

    if DVC_CONFIG.is_file():
        from scripts.dvc_remote_setup import read_remote_url, remote_configured

        parser = configparser.ConfigParser()
        parser.read(DVC_CONFIG)
        remote = parser.get("core", "remote", fallback="").strip()
        if remote:
            if remote_configured(root, remote):
                url = read_remote_url(root, remote) or ""
                issues.append(f"dvc remote configured: {remote} → {url} — informational")
            else:
                issues.append(f"dvc core.remote={remote} but section not found")
        else:
            issues.append(
                "dvc remote not configured — run make dvc-setup (informational)"
            )
    else:
        issues.append("dvc config missing — copy from .dvc/config.example (informational)")

    return issues


def validate_dvc(root: Path = ROOT) -> tuple[bool, list[str]]:
    """Validate DVC pipeline. Returns (ok, all_issues)."""
    issues = collect_dvc_issues(root=root)
    blocking = [i for i in issues if "informational" not in i]
    return len(blocking) == 0, issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DVC pipeline and remote")
    parser.parse_args()

    ok, issues = validate_dvc()
    for item in issues:
        prefix = "INFO" if "informational" in item else "ERROR"
        print(f"{prefix}: {item}")

    if ok:
        print("DVC pipeline validation passed.")
        print("Next: make dvc-setup && make dvc-push")
        return 0

    print("DVC validation failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
