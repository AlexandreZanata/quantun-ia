"""Unit tests for sample-scale evaluation helpers."""

from __future__ import annotations

from src.application.sample_scale_evaluation import (
    SAMPLE_SCALE_SIZES,
    HoldoutPredictionRow,
    HoldoutPredictionsExport,
    SampleScaleCurveResult,
    SampleScalePoint,
    curve_to_dict,
    predictions_to_dict,
)


def test_sample_scale_sizes_cover_100_to_2000():
    assert SAMPLE_SCALE_SIZES[0] == 100
    assert SAMPLE_SCALE_SIZES[-1] == 2000
    assert len(SAMPLE_SCALE_SIZES) == 20
    assert list(SAMPLE_SCALE_SIZES) == list(range(100, 2001, 100))


def test_curve_to_dict_serializes_points():
    result = SampleScaleCurveResult(
        exp_id="exp_034",
        model_name="large_nano_mlp",
        dataset="synthea_cv_risk_v1",
        seed=42,
        split="val",
        points=(
            SampleScalePoint(
                n_rows=100,
                n_negatives=10,
                n_positives=90,
                accuracy=0.99,
                precision=0.99,
                recall=0.99,
                f1=0.99,
                roc_auc=0.85,
                pr_auc=0.88,
                brier_score=0.01,
                ece=0.05,
                true_positive=89,
                true_negative=9,
                false_positive=1,
                false_negative=1,
            ),
        ),
    )
    payload = curve_to_dict(result)
    assert payload["exp_id"] == "exp_034"
    assert payload["points"][0]["n_rows"] == 100
    assert payload["points"][0]["roc_auc"] == 0.85
    assert payload["points"][0]["confusion"]["tp"] == 89


def test_predictions_to_dict_serializes_rows():
    export = HoldoutPredictionsExport(
        exp_id="exp_034",
        model_name="large_nano_mlp",
        dataset="synthea_cv_risk_v1",
        seed=42,
        split="val",
        n_rows=2,
        n_negatives=0,
        n_positives=2,
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        roc_auc=0.9,
        rows=(
            HoldoutPredictionRow(
                row_index=0,
                probability=0.95,
                predicted=1,
                actual=1,
                correct=1,
            ),
            HoldoutPredictionRow(
                row_index=1,
                probability=0.88,
                predicted=1,
                actual=1,
                correct=1,
            ),
        ),
    )
    payload = predictions_to_dict(export)
    assert payload["n_rows"] == 2
    assert payload["rows"][0]["probability"] == 0.95
    assert payload["precision"] == 1.0
