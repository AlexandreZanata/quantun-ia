"""JSON Schema for MicroQML Bench v1 export bundles."""

from __future__ import annotations

MICROQML_BENCH_V1_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "MicroQML Bench v1",
    "type": "object",
    "required": [
        "bench_id",
        "version",
        "schema_version",
        "display_name",
        "protocol",
        "tasks",
        "leaderboard",
        "generated_at",
    ],
    "properties": {
        "bench_id": {"const": "microqml_bench"},
        "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
        "schema_version": {"const": "1"},
        "display_name": {"type": "string", "minLength": 1},
        "description": {"type": "string"},
        "protocol": {
            "type": "object",
            "required": ["holdout_fraction", "seeds", "profile", "eval_set"],
            "properties": {
                "holdout_fraction": {"type": "number", "minimum": 0, "maximum": 1},
                "seeds": {"type": "integer", "minimum": 1},
                "profile": {"type": "string"},
                "significance": {"type": "string"},
                "eval_set": {"type": "string"},
                "preprocessing": {"type": "string"},
                "max_qubits_default": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
        "tasks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["task_id", "category", "exp_id", "dataset"],
                "properties": {
                    "task_id": {"type": "string", "minLength": 1},
                    "category": {
                        "type": "string",
                        "enum": ["synthetic", "tabular", "image_tabular", "sequence"],
                    },
                    "exp_id": {"type": "string", "pattern": r"^exp_\d{3}$"},
                    "dataset": {"type": "string"},
                    "description": {"type": "string"},
                    "primary": {"type": "boolean"},
                    "flagship": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        },
        "leaderboard": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["task_id", "exp_id", "model", "accuracy_pct", "eval_set"],
                "properties": {
                    "task_id": {"type": "string"},
                    "exp_id": {"type": "string"},
                    "model": {"type": "string"},
                    "accuracy_pct": {"type": "number"},
                    "ci_low_pct": {"type": ["number", "null"]},
                    "ci_high_pct": {"type": ["number", "null"]},
                    "std_pct": {"type": ["number", "null"]},
                    "n_seeds": {"type": ["integer", "null"]},
                    "n_params": {"type": ["integer", "null"]},
                    "elapsed_s": {"type": ["number", "null"]},
                    "epochs": {"type": ["integer", "null"]},
                    "eval_set": {"type": "string"},
                    "source": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "generated_at": {"type": "string", "minLength": 1},
        "software_version": {"type": "string"},
        "citation": {"type": "string"},
    },
    "additionalProperties": False,
}
