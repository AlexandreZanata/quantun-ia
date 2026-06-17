"""Contract tests for DVC artifact pipeline (Phase 27)."""

from __future__ import annotations

from pathlib import Path

import yaml

DVC_YAML = Path("dvc.yaml")
DVC_EXAMPLE = Path(".dvc/config.example")
DVC_REMOTE_DOC = Path("docs/dvc_remote.md")

EXPECTED_STAGES = frozenset({"export_results", "figures", "latex_tables"})


def test_dvc_yaml_exists_and_has_stages():
    assert DVC_YAML.is_file()
    data = yaml.safe_load(DVC_YAML.read_text(encoding="utf-8"))
    stages = data.get("stages", {})
    assert EXPECTED_STAGES.issubset(stages.keys())


def test_dvc_stage_scripts_exist():
    data = yaml.safe_load(DVC_YAML.read_text(encoding="utf-8"))
    for stage_name, stage in data["stages"].items():
        cmd = stage.get("cmd", "")
        script = cmd.split()[-1] if cmd else ""
        if script.endswith(".py"):
            assert Path(script).is_file(), f"{stage_name} missing script {script}"


def test_dvc_config_example_exists():
    assert DVC_EXAMPLE.is_file()
    text = DVC_EXAMPLE.read_text(encoding="utf-8")
    assert "remote" in text


def test_dvc_remote_doc_mentions_dvc_check():
    text = DVC_REMOTE_DOC.read_text(encoding="utf-8")
    assert "dvc-check" in text or "make dvc-check" in text


def test_dvc_stage_outputs_match_release_pipeline():
    data = yaml.safe_load(DVC_YAML.read_text(encoding="utf-8"))
    export_outs = data["stages"]["export_results"]["outs"]
    assert "data/exports/results.csv" in export_outs
    assert "figures/" in data["stages"]["figures"]["outs"]
    assert "paper/tables/" in data["stages"]["latex_tables"]["outs"]
