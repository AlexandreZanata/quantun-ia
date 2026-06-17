"""Streamlit Nano Trainer page — mini real-data training."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.application.dto import TrainNanomodelDTO
from src.application.nanotrainer_config import load_nanotrainer_config
from src.application.train_nanomodel import execute
from src.shared.result import Fail, Ok

st.set_page_config(page_title="Nano Trainer", page_icon="⚛", layout="wide")

st.title("NANO TRAINER")
st.caption("Mini real-data training — logs append to `logs/experiments.jsonl` (exp_id=nano_train)")

cfg = load_nanotrainer_config()
pairs = cfg.get("pairs", [])
models = sorted({p["model"] for p in pairs})
datasets_by_model: dict[str, list[str]] = {}
for p in pairs:
    datasets_by_model.setdefault(p["model"], []).append(p["dataset"])
for k in datasets_by_model:
    datasets_by_model[k] = sorted(datasets_by_model[k])

profiles = sorted(cfg.get("profiles", {}).keys())

with st.sidebar:
    st.header("Configuration")
    model = st.selectbox("Model", models)
    dataset = st.selectbox("Dataset", datasets_by_model.get(model, []))
    profile = st.selectbox("Profile", profiles, index=profiles.index("mini") if "mini" in profiles else 0)
    epochs_override = st.number_input("Epochs (0 = profile default)", min_value=0, value=0, step=1)
    seed_override = st.number_input("Seed (0 = profile default)", min_value=0, value=0, step=1)
    run_btn = st.button("RUN MINI TRAINING", type="primary")

if run_btn:
    dto = TrainNanomodelDTO(
        model_name=model,
        dataset=dataset,
        profile=profile,
        epochs=int(epochs_override) if epochs_override > 0 else None,
        seed=int(seed_override) if seed_override > 0 else None,
    )
    with st.spinner(f"Training {model} on {dataset}…"):
        result = execute(dto)

    if isinstance(result, Fail):
        st.error(f"{result.error.code}: {result.error.message}")
    else:
        assert isinstance(result, Ok)
        r = result.value
        st.success(f"Holdout accuracy: **{r.accuracy * 100:.1f}%**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Loss", f"{r.loss:.4f}")
        col2.metric("Elapsed", f"{r.elapsed_s}s")
        col3.metric("Params", r.n_params)
        col4.metric("Epochs", r.n_epochs)

        st.info(
            "Refresh the main Benchmark dashboard to see numbered experiments. "
            "`nano_train` runs are excluded from the publication leaderboard."
        )

st.markdown("---")
st.subheader("Allowed model × dataset pairs")
rows = [{"model": p["model"], "dataset": p["dataset"]} for p in pairs]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
