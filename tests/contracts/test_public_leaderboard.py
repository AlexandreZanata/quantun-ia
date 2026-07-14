"""Contract tests for Phase D public leaderboard (GitHub Pages)."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from scripts.publish_leaderboard import (
    DEFAULT_OUT_DIR,
    PUBLIC_JSON_URL,
    PUBLIC_VIEWER_URL,
    publish_leaderboard,
    verify_published_leaderboard,
)
from tests.contracts.microqml_bench_schema import MICROQML_BENCH_V1_SCHEMA

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION_FIXTURE = ROOT / "tests" / "contracts" / "fixtures" / "publication_experiments.jsonl"


def test_public_leaderboard_files_exist():
    assert (DEFAULT_OUT_DIR / "v1.json").is_file()
    assert (DEFAULT_OUT_DIR / "meta.json").is_file()
    assert (DEFAULT_OUT_DIR / "index.html").is_file()


def test_committed_v1_json_validates_against_schema():
    export = json.loads((DEFAULT_OUT_DIR / "v1.json").read_text(encoding="utf-8"))
    jsonschema.validate(instance=export, schema=MICROQML_BENCH_V1_SCHEMA)
    assert export["bench_id"] == "microqml_bench"
    assert len(export["leaderboard"]) >= 1


def test_committed_leaderboard_includes_flagship_task():
    export = json.loads((DEFAULT_OUT_DIR / "v1.json").read_text(encoding="utf-8"))
    task_ids = {row["task_id"] for row in export["leaderboard"]}
    assert "quantum_nano_bc" in task_ids
    flagship_rows = [r for r in export["leaderboard"] if r["task_id"] == "quantum_nano_bc"]
    models = {r["model"] for r in flagship_rows}
    assert "hybrid_sandwich" in models
    assert "logistic_regression" in models


def test_meta_sidecar_has_canonical_urls():
    meta = json.loads((DEFAULT_OUT_DIR / "meta.json").read_text(encoding="utf-8"))
    assert meta["canonical_json_url"] == PUBLIC_JSON_URL
    assert meta["viewer_url"] == PUBLIC_VIEWER_URL
    export = json.loads((DEFAULT_OUT_DIR / "v1.json").read_text(encoding="utf-8"))
    assert meta["leaderboard_rows"] == len(export["leaderboard"])


def test_verify_published_leaderboard_passes():
    assert verify_published_leaderboard() is True


def test_committed_cycle_scorecards_exist():
    cycle2 = json.loads((DEFAULT_OUT_DIR / "cycle2.json").read_text(encoding="utf-8"))
    cycle3 = json.loads((DEFAULT_OUT_DIR / "cycle3.json").read_text(encoding="utf-8"))
    assert cycle2["bench_id"] == "cycle2_grand_leaderboard"
    assert cycle3["bench_id"] == "cycle3_grand_leaderboard"
    assert "exp_092" in cycle2.get("observed_accepts", [])
    assert "exp_102" in cycle3.get("observed_accepts", [])


def test_meta_includes_cycle_summaries():
    meta = json.loads((DEFAULT_OUT_DIR / "meta.json").read_text(encoding="utf-8"))
    assert "cycles" in meta
    assert meta["cycles"]["v3_image"]["json"] == "cycle3.json"
    assert "exp_109" in meta["cycles"]["v3_image"]["accepted"]


def test_publish_leaderboard_regenerates_valid_bundle(tmp_path):
    out = tmp_path / "leaderboard"
    json_path, meta_path = publish_leaderboard(jsonl_path=PUBLICATION_FIXTURE, output_dir=out)
    assert json_path.is_file()
    assert meta_path.is_file()
    assert (out / "cycle2.json").is_file()
    assert (out / "cycle3.json").is_file()
    export = json.loads(json_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=export, schema=MICROQML_BENCH_V1_SCHEMA)
    assert len(export["leaderboard"]) >= 2
