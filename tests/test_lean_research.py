from scripts.validate_lean_research import (
    load_acquisitions,
    load_json,
    validate,
    validate_acquisitions,
)


def test_lean_research_registry_is_valid():
    assert validate() == []


def test_hypotheses_cover_ten_families():
    hypotheses = load_json("hypotheses.json")["hypotheses"]
    assert len(hypotheses) == 100
    assert len({item["family_code"] for item in hypotheses}) == 10
    assert all(item["falsification_gate"] for item in hypotheses)


def test_baselines_keep_large_models_as_references():
    baselines = load_json("baselines.json")["baselines"]
    large = [item for item in baselines if item["parameters_nominal"] > 1_500_000_000]
    assert large
    assert all(item["kind"] == "reference" for item in large)


def test_lean_acquisitions_are_checksum_verified():
    acquisitions = load_acquisitions()
    assert acquisitions
    assert validate_acquisitions() == []
    for acquisition in acquisitions:
        archive = acquisition["archive"]
        assert archive["checksum_verified"] is True
        assert archive["md5_published"] == archive["md5_observed"]
        assert archive["local_path"].startswith("data/raw/")
        assert acquisition["contamination_report"]["sealed_sets_touched"] == []
        for split in acquisition["splits"]:
            assert split["test"]["content_inspected"] is False

