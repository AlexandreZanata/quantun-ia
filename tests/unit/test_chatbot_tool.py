"""Unit tests for chatbot tool adapter."""

from __future__ import annotations

from pathlib import Path

from src.application.chatbot_tool import (
    RESEARCH_DISCLAIMER,
    TOOL_SCORE_BREAST_CANCER,
    ChatbotToolCallDTO,
    build_openai_tool_schema,
    execute_tool_call,
    format_assistant_message,
    load_dialogue_fixtures,
    max_probability_delta,
    parse_tool_arguments,
    predict_request_payload,
    validate_feature_rows,
)
from src.application.dto import PredictNanomodelDTO, TrainNanomodelDTO
from src.application.predict_nanomodel import execute as predict_execute
from src.application.train_nanomodel import execute as train_execute
from src.shared.result import Fail, Ok

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "chatbot_dialogues"


def test_openai_tool_schema_has_30_features():
    schema = build_openai_tool_schema()
    items = schema["function"]["parameters"]["properties"]["features"]["items"]
    assert items["minItems"] == 30
    assert items["maxItems"] == 30
    assert schema["function"]["name"] == TOOL_SCORE_BREAST_CANCER


def test_validate_feature_rows_rejects_wrong_count():
    err = validate_feature_rows([[1.0] * 29])
    assert err is not None
    assert err.code == "INVALID_FEATURES"


def test_validate_feature_rows_accepts_30():
    assert validate_feature_rows([[0.0] * 30, [1.0] * 30]) is None


def test_parse_tool_arguments_maps_features():
    outcome = parse_tool_arguments(TOOL_SCORE_BREAST_CANCER, {"features": [[0.0] * 30]})
    assert isinstance(outcome, Ok)
    assert len(outcome.value) == 1
    assert len(outcome.value[0]) == 30


def test_parse_tool_arguments_rejects_unknown_tool():
    outcome = parse_tool_arguments("unknown_tool", {"features": [[0.0] * 30]})
    assert isinstance(outcome, Fail)
    assert outcome.error.code == "UNKNOWN_TOOL"


def test_format_assistant_message_includes_disclaimer():
    text = format_assistant_message(probabilities=[0.42], labels=[0])
    assert RESEARCH_DISCLAIMER in text
    assert "0.420" in text or "0.42" in text


def test_max_probability_delta():
    assert max_probability_delta([0.1, 0.9], [0.1000001, 0.8999999]) < 1e-5


def test_load_dialogue_fixtures_has_ten():
    dialogues = load_dialogue_fixtures(FIXTURES)
    assert len(dialogues) == 10
    for dialogue in dialogues:
        assert len(dialogue.arguments["features"][0]) == 30


def test_execute_tool_call_matches_direct_predict(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")

    train_execute(
        TrainNanomodelDTO(
            model_name="hybrid_sandwich",
            dataset="breast_cancer",
            profile="ci",
            epochs=6,
            seed=11,
            exp_id="chatbot_tool_test",
            save_checkpoints=True,
        )
    )

    row = [[0.0] * 30]
    dto = ChatbotToolCallDTO(
        tool_name=TOOL_SCORE_BREAST_CANCER,
        arguments={"features": row},
        exp_id="chatbot_tool_test",
        seed=11,
    )
    tool_out = execute_tool_call(dto)
    assert isinstance(tool_out, Ok)

    direct = predict_execute(
        PredictNanomodelDTO(
            exp_id="chatbot_tool_test",
            model_name="hybrid_sandwich",
            dataset="breast_cancer",
            seed=11,
            features=row,
        )
    )
    assert isinstance(direct, Ok)
    delta = max_probability_delta(tool_out.value.probabilities, direct.value.probabilities)
    assert delta < 1e-5


def test_predict_request_payload_shape():
    dto = ChatbotToolCallDTO(
        tool_name=TOOL_SCORE_BREAST_CANCER,
        arguments={"features": [[0.0] * 30]},
        exp_id="exp_test",
        seed=7,
    )
    payload = predict_request_payload(dto)
    assert payload["exp_id"] == "exp_test"
    assert payload["seed"] == 7
    assert len(payload["features"][0]) == 30
