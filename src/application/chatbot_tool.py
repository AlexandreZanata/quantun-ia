"""Chatbot tool adapter — validate tool calls and run nanomodel inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.datasets import load_breast_cancer

from src.application.dto import PredictNanomodelDTO
from src.application.predict_nanomodel import execute as predict_execute
from src.shared.result import Result, fail, ok
from src.training.structured_log import init_correlation_id, log_event

TOOL_SCORE_BREAST_CANCER = "score_breast_cancer"
BREAST_CANCER_FEATURE_COUNT = 30
RESEARCH_DISCLAIMER = "This output is from a research prototype — not a clinical diagnosis."

DEFAULT_EXP_ID = "quantum_nano_bc_app"
DEFAULT_MODEL = "hybrid_sandwich"
DEFAULT_DATASET = "breast_cancer"
DEFAULT_SEED = 42

FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "chatbot_dialogues"


@dataclass(frozen=True)
class ChatbotToolError:
    code: str
    message: str


@dataclass(frozen=True)
class ChatbotToolCallDTO:
    tool_name: str
    arguments: dict[str, Any]
    exp_id: str = DEFAULT_EXP_ID
    model_name: str = DEFAULT_MODEL
    dataset: str = DEFAULT_DATASET
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class ChatbotToolResult:
    exp_id: str
    model_name: str
    dataset: str
    seed: int
    probabilities: list[float]
    labels: list[int]
    checkpoint_path: str
    feature_count: int
    message: str
    record_source: str = "chatbot_tool"


def breast_cancer_feature_names() -> list[str]:
    return list(load_breast_cancer().feature_names)


def build_openai_tool_schema() -> dict[str, Any]:
    """OpenAI-compatible function tool definition for local LLM tool routing."""
    return {
        "type": "function",
        "function": {
            "name": TOOL_SCORE_BREAST_CANCER,
            "description": (
                "Score Wisconsin Breast Cancer tabular features and return malignancy "
                "probability. Requires exactly 30 numeric features per row (raw UCI scale)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "features": {
                        "type": "array",
                        "description": (
                            "Rows of raw Wisconsin Breast Cancer features "
                            f"({BREAST_CANCER_FEATURE_COUNT} values per row)."
                        ),
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": BREAST_CANCER_FEATURE_COUNT,
                            "maxItems": BREAST_CANCER_FEATURE_COUNT,
                        },
                        "minItems": 1,
                    }
                },
                "required": ["features"],
            },
        },
    }


def validate_feature_rows(
    rows: list[list[float]],
    *,
    expected_count: int = BREAST_CANCER_FEATURE_COUNT,
) -> ChatbotToolError | None:
    if not rows:
        return ChatbotToolError("INVALID_FEATURES", "features must not be empty")
    for idx, row in enumerate(rows):
        if len(row) != expected_count:
            return ChatbotToolError(
                "INVALID_FEATURES",
                f"row {idx} must have {expected_count} features (got {len(row)})",
            )
    return None


def parse_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
) -> Result[list[list[float]], ChatbotToolError]:
    if tool_name != TOOL_SCORE_BREAST_CANCER:
        return fail(ChatbotToolError("UNKNOWN_TOOL", f"unsupported tool: {tool_name}"))

    raw = arguments.get("features")
    if not isinstance(raw, list):
        return fail(ChatbotToolError("INVALID_FEATURES", "features must be a list of rows"))

    rows: list[list[float]] = []
    for item in raw:
        if not isinstance(item, list):
            return fail(ChatbotToolError("INVALID_FEATURES", "each feature row must be a list"))
        rows.append([float(v) for v in item])

    err = validate_feature_rows(rows)
    if err is not None:
        return fail(err)
    return ok(rows)


def format_assistant_message(
    *,
    probabilities: list[float],
    labels: list[int],
) -> str:
    lines = ["Breast cancer risk scores (research prototype):"]
    for idx, (prob, label) in enumerate(zip(probabilities, labels, strict=True)):
        lines.append(f"- Row {idx + 1}: probability={prob:.4f}, label={label}")
    lines.append("")
    lines.append(RESEARCH_DISCLAIMER)
    return "\n".join(lines)


def predict_request_payload(dto: ChatbotToolCallDTO) -> dict[str, Any]:
    from src.shared.result import Fail

    parsed = parse_tool_arguments(dto.tool_name, dto.arguments)
    if isinstance(parsed, Fail):
        raise ValueError(parsed.error.message)
    return {
        "exp_id": dto.exp_id,
        "model_name": dto.model_name,
        "dataset": dto.dataset,
        "seed": dto.seed,
        "features": parsed.value,
    }


def execute_tool_call(
    dto: ChatbotToolCallDTO,
) -> Result[ChatbotToolResult, ChatbotToolError]:
    """Run tool call through the same predict use case as POST /api/v1/predictions."""
    init_correlation_id()
    parsed = parse_tool_arguments(dto.tool_name, dto.arguments)
    from src.shared.result import Fail

    if isinstance(parsed, Fail):
        return fail(parsed.error)

    outcome = predict_execute(
        PredictNanomodelDTO(
            exp_id=dto.exp_id,
            model_name=dto.model_name,
            dataset=dto.dataset,
            seed=dto.seed,
            features=parsed.value,
        )
    )
    if isinstance(outcome, Fail):
        return fail(ChatbotToolError(outcome.error.code, outcome.error.message))

    from src.shared.result import Ok

    assert isinstance(outcome, Ok)
    result = outcome.value
    message = format_assistant_message(
        probabilities=result.probabilities,
        labels=result.labels,
    )
    log_event(
        "info",
        "chatbot tool call",
        tool_name=dto.tool_name,
        exp_id=dto.exp_id,
        dataset=dto.dataset,
        n_rows=len(parsed.value),
        record_source="chatbot_tool",
    )
    return ok(
        ChatbotToolResult(
            exp_id=result.exp_id,
            model_name=result.model_name,
            dataset=result.dataset,
            seed=result.seed,
            probabilities=result.probabilities,
            labels=result.labels,
            checkpoint_path=result.checkpoint_path,
            feature_count=BREAST_CANCER_FEATURE_COUNT,
            message=message,
        )
    )


def max_probability_delta(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("probability lists must have equal length")
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def load_dialogue_fixtures(directory: Path | None = None) -> list[ChatbotToolCallDTO]:
    root = directory or FIXTURES_ROOT
    dialogues: list[ChatbotToolCallDTO] = []
    for path in sorted(root.glob("dialogue_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dialogues.append(
            ChatbotToolCallDTO(
                tool_name=str(payload["tool_name"]),
                arguments=dict(payload["arguments"]),
                exp_id=str(payload.get("exp_id", DEFAULT_EXP_ID)),
                model_name=str(payload.get("model_name", DEFAULT_MODEL)),
                dataset=str(payload.get("dataset", DEFAULT_DATASET)),
                seed=int(payload.get("seed", DEFAULT_SEED)),
            )
        )
    return dialogues
