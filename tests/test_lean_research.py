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


def test_lean_acquisitions_are_structurally_valid():
    acquisitions = load_acquisitions()
    assert acquisitions
    assert validate_acquisitions() == []


def test_dataset_acquisitions_are_checksum_verified():
    datasets = [item for item in load_acquisitions() if item["kind"] == "dataset"]
    assert datasets
    for acquisition in datasets:
        archive = acquisition["archive"]
        assert archive["checksum_verified"] is True
        assert archive["md5_published"] == archive["md5_observed"]
        assert archive["local_path"].startswith("data/raw/")
        assert acquisition["contamination_report"]["sealed_sets_touched"] == []
        for split in acquisition["splits"]:
            assert split["test"]["content_inspected"] is False


def test_weight_acquisitions_verify_lfs_and_parameter_recount():
    weights = {item["id"]: item for item in load_acquisitions() if item["kind"] == "weights"}
    assert {"reprover_tacgen_byt5_small", "reprover_retriever_byt5_small"} <= set(weights)
    for acquisition in weights.values():
        assert acquisition["checksums_verified"] is True
        assert len(acquisition["version_pin"]["revision_sha"]) == 40
        lfs_files = [item for item in acquisition["files"] if item["is_lfs"]]
        assert lfs_files
        assert all(item["checksum_match"] is True for item in lfs_files)
        parameters = acquisition["parameters"]
        assert parameters["stored_total"] > 0
        assert parameters["active_total"] == parameters["stored_total"]
        assert parameters["layout_consistent"] is True
    tacgen = weights["reprover_tacgen_byt5_small"]["parameters"]["stored_total"]
    retriever = weights["reprover_retriever_byt5_small"]["parameters"]["stored_total"]
    assert 250_000_000 <= tacgen <= 300_000_000
    assert 150_000_000 <= retriever <= 250_000_000

