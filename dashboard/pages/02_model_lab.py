"""Streamlit Model Lab — real inference on trained serve checkpoints."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.application.dto import PredictNanomodelDTO
from src.application.evaluate_serve_model import (
    SERVE_MODELS,
    EvaluateServeModelDTO,
    execute as evaluate_execute,
    load_open_split_labeled,
)
from src.application.open_serve import open_dataset_feature_count
from src.application.predict_nanomodel import execute as predict_execute
from src.shared.result import Fail, Ok

st.set_page_config(page_title="Model Lab", page_icon="🧪", layout="wide")

RETRO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Share Tech Mono', monospace !important; }
h1, h2, h3 { font-family: 'VT323', monospace !important; color: #33ff66 !important; }
.metric-card {
  border: 1px solid #33ff6644; background: #080f08; padding: 1rem; border-radius: 4px;
}
</style>
"""

st.markdown(RETRO, unsafe_allow_html=True)
st.title("MODEL LAB")
st.caption("Real inference on trained checkpoints — RTX 4060 (`QML_DEVICE=cuda`)")

os.environ.setdefault("MLFLOW_DISABLE", "1")
os.environ.setdefault("QML_DEVICE", "cuda")

model_options = {m.label: m for m in SERVE_MODELS}

with st.sidebar:
    st.header("Serve checkpoint")
    selected_label = st.selectbox("Model", list(model_options.keys()))
    profile = model_options[selected_label]
    split = st.selectbox("Holdout split", ["val", "test"], index=0)
    n_rows = st.slider("Evaluation rows", min_value=500, max_value=20000, value=5000, step=500)
    chunk_size = st.number_input("Batch chunk size", min_value=64, max_value=4096, value=2048, step=64)

tab_eval, tab_predict, tab_api = st.tabs(["Evaluation panel", "Mini predictor", "API example"])

with tab_eval:
    st.subheader("Holdout evaluation")
    st.markdown(
        f"**{profile.label}** · `{profile.exp_id}` / `{profile.model_name}` / "
        f"`{profile.dataset}` · seed **{profile.seed}**"
    )

    if st.button("RUN REAL EVALUATION", type="primary"):
        with st.spinner(f"Scoring {n_rows:,} rows on {split} split…"):
            outcome = evaluate_execute(
                EvaluateServeModelDTO(
                    exp_id=profile.exp_id,
                    model_name=profile.model_name,
                    dataset=profile.dataset,
                    seed=profile.seed,
                    split=split,
                    n_rows=n_rows,
                    chunk_size=int(chunk_size),
                )
            )

        if isinstance(outcome, Fail):
            st.error(f"{outcome.error.code}: {outcome.error.message}")
        else:
            assert isinstance(outcome, Ok)
            r = outcome.value
            st.session_state["eval_result"] = r

    if "eval_result" in st.session_state:
        r = st.session_state["eval_result"]
        if r.exp_id != profile.exp_id or r.model_name != profile.model_name:
            st.info("Configuration changed — run evaluation again.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ROC-AUC", f"{r.roc_auc:.4f}")
            c2.metric("Accuracy", f"{r.accuracy * 100:.2f}%")
            c3.metric("Brier score", f"{r.brier_score:.4f}")
            c4.metric("Rows scored", f"{r.n_rows:,}")

            st.caption(f"Checkpoint: `{r.checkpoint_path}`")

            left, right = st.columns(2)
            with left:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=r.fpr,
                        y=r.tpr,
                        mode="lines",
                        name="ROC",
                        line=dict(color="#33ff66", width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random",
                        line=dict(color="#666", dash="dash"),
                    )
                )
                fig.update_layout(
                    title="ROC curve",
                    xaxis_title="False positive rate",
                    yaxis_title="True positive rate",
                    paper_bgcolor="#050805",
                    plot_bgcolor="#080f08",
                    font=dict(color="#33ff66"),
                )
                st.plotly_chart(fig, use_container_width=True)

            with right:
                cm = pd.DataFrame(
                    [
                        [r.confusion.true_negative, r.confusion.false_positive],
                        [r.confusion.false_negative, r.confusion.true_positive],
                    ],
                    index=["Actual 0", "Actual 1"],
                    columns=["Pred 0", "Pred 1"],
                )
                st.markdown("**Confusion matrix**")
                st.dataframe(cm, use_container_width=True)
                st.markdown(
                    f"- Mean probability: **{r.mean_probability:.4f}**  \n"
                    f"- Predicted positive rate: **{r.positive_rate * 100:.1f}%**"
                )

            st.markdown("**Sample predictions (first 10 rows)**")
            st.dataframe(pd.DataFrame(r.sample_rows), use_container_width=True, hide_index=True)

with tab_predict:
    st.subheader("Single-row mini predictor")
    n_features = open_dataset_feature_count(profile.dataset)
    st.markdown(f"Enter **{n_features}** raw features (comma-separated). Scaling is applied by the checkpoint.")

    if st.button("Load random row from holdout"):
        rows, labels = load_open_split_labeled(
            profile.dataset,
            PROJECT_ROOT,
            split=split,
            n_rows=1,
            random_state=profile.seed + 7,
        )
        st.session_state["demo_features"] = ", ".join(f"{v:.6f}" for v in rows[0])
        st.session_state["demo_actual"] = labels[0]

    default_features = st.session_state.get("demo_features", "")
    feature_text = st.text_area("Features", value=default_features, height=80)

    if st.button("PREDICT", type="primary"):
        try:
            values = [float(v.strip()) for v in feature_text.split(",") if v.strip()]
        except ValueError:
            st.error("Invalid numbers in feature input.")
        elif len(values) != n_features:
            st.error(f"Expected {n_features} features, got {len(values)}.")
        else:
            outcome = predict_execute(
                PredictNanomodelDTO(
                    exp_id=profile.exp_id,
                    model_name=profile.model_name,
                    dataset=profile.dataset,
                    seed=profile.seed,
                    features=[values],
                )
            )
            if isinstance(outcome, Fail):
                st.error(f"{outcome.error.code}: {outcome.error.message}")
            else:
                assert isinstance(outcome, Ok)
                pr = outcome.value
                prob = pr.probabilities[0]
                label = pr.labels[0]
                st.success(f"Probability: **{prob:.4f}** → class **{label}**")
                if "demo_actual" in st.session_state:
                    actual = st.session_state["demo_actual"]
                    match = "✓" if label == actual else "✗"
                    st.info(f"Holdout actual label: **{actual}** {match}")

with tab_api:
    st.subheader("REST API — POST /api/v1/predictions")
    example_rows, _ = load_open_split_labeled(
        profile.dataset,
        PROJECT_ROOT,
        split="val",
        n_rows=2,
        random_state=profile.seed,
    )
    payload = {
        "exp_id": profile.exp_id,
        "model_name": profile.model_name,
        "dataset": profile.dataset,
        "seed": profile.seed,
        "features": example_rows,
    }
    st.code(
        "\n".join(
            [
                "curl -s -X POST http://127.0.0.1:8000/api/v1/predictions \\",
                "  -H 'Content-Type: application/json' \\",
                f"  -d '{json.dumps(payload)}'",
            ]
        ),
        language="bash",
    )
    st.markdown("Start API: `make api` · Dashboard: `make model-lab`")
