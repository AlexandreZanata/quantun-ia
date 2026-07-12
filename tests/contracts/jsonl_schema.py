"""JSON Schema definitions for logs/experiments.jsonl records."""

from __future__ import annotations

EXPERIMENT_RECORD_SCHEMA: dict = {
    "type": "object",
    "required": ["exp_id", "model_name", "started_at"],
    "properties": {
        "exp_id": {"type": "string", "minLength": 1},
        "model_name": {"type": "string", "minLength": 1},
        "started_at": {"type": "string"},
        "record_type": {
            "type": "string",
            "enum": [
                "multi_seed_summary",
                "paired_comparison",
                "paired_comparison_batch",
                "applicability_gate",
                "clinical_validation",
                "agro_validation",
                "calibration_evaluation",
                "sample_scale_curve",
            ],
        },
        "seed": {"type": ["integer", "null"]},
        "profile": {"type": ["string", "null"]},
        "elapsed_s": {"type": "number"},
        "final_acc": {"type": ["number", "null"]},
        "test_accuracy": {"type": ["number", "null"]},
        "test_loss": {"type": ["number", "null"]},
        "eval_set": {"type": "string"},
        "n_epochs": {"type": "integer"},
        "history": {"type": "array"},
        "adaptive_lr": {"type": "boolean"},
    },
    "additionalProperties": True,
}

MULTI_SEED_SUMMARY_SCHEMA: dict = {
    "type": "object",
    "required": ["exp_id", "record_type", "summary"],
    "properties": {
        "exp_id": {"type": "string"},
        "record_type": {"const": "multi_seed_summary"},
        "summary": {
            "type": "object",
            "additionalProperties": {
                "anyOf": [
                    {
                        "type": "object",
                        "required": ["mean", "ci_low", "ci_high"],
                        "properties": {
                            "mean": {"type": "number"},
                            "std": {"type": "number"},
                            "ci_low": {"type": "number"},
                            "ci_high": {"type": "number"},
                            "n_seeds": {"type": "integer"},
                        },
                    },
                    {
                        "type": "object",
                        "not": {"required": ["mean"]},
                        "additionalProperties": True,
                    },
                ]
            },
        },
    },
    "additionalProperties": True,
}

PAIRED_COMPARISON_BATCH_SCHEMA: dict = {
    "type": "object",
    "required": ["exp_id", "record_type", "comparisons"],
    "properties": {
        "exp_id": {"type": "string"},
        "record_type": {"const": "paired_comparison_batch"},
        "comparisons": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label_a", "label_b", "mean_diff"],
                "properties": {
                    "label_a": {"type": "string"},
                    "label_b": {"type": "string"},
                    "mean_diff": {"type": "number"},
                    "p_value": {"type": ["number", "null"]},
                    "effect_size_cohens_d": {"type": ["number", "null"]},
                },
            },
        },
    },
    "additionalProperties": True,
}


def schema_for_record(record: dict) -> dict:
    record_type = record.get("record_type")
    if record_type == "multi_seed_summary":
        return MULTI_SEED_SUMMARY_SCHEMA
    if record_type == "paired_comparison_batch":
        return PAIRED_COMPARISON_BATCH_SCHEMA
    if record_type in {
        "paired_comparison",
        "applicability_gate",
        "clinical_validation",
        "agro_validation",
        "calibration_evaluation",
        "sample_scale_curve",
    }:
        return {"type": "object", "required": ["exp_id", "record_type"], "additionalProperties": True}
    return EXPERIMENT_RECORD_SCHEMA
