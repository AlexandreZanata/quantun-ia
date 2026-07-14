"""Image Nano Lab — Cycle v3 floors + optional NanoUNet samples (Phase K / K-T3)."""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.application.image_nano_predict import load_image_nano_model_card, predict_image_i2i
from src.shared.result import Fail, Ok

st.set_page_config(page_title="Image Nano Lab", page_icon="🖼️", layout="wide")

os.environ.setdefault("MLFLOW_DISABLE", "1")
os.environ.setdefault("QML_DEVICE", "cuda")

LEADERBOARD = PROJECT_ROOT / "dist" / "leaderboards" / "cycle3_grand_leaderboard.json"


st.title("Image Nano Lab")
st.markdown(
    "Cycle v3 image nano scorecard (exp_101–111) beside optional NanoUNet CIFAR samples "
    "from the Phase K serve bundle."
)

card = load_image_nano_model_card(PROJECT_ROOT)
st.subheader("Serve bundle")
st.json(card)

if LEADERBOARD.is_file():
    payload = json.loads(LEADERBOARD.read_text(encoding="utf-8"))
    st.subheader("Cycle v3 grand leaderboard")
    st.caption(
        f"Confirmed={payload.get('hypothesis_confirmed')} · "
        f"accepts={payload.get('observed_accepts')}"
    )
    st.dataframe(payload.get("rows", []), use_container_width=True)
else:
    st.info("Leaderboard JSON missing — run `make exp-112-publication`.")

st.subheader("Sample NanoUNet (I2I)")
n = st.slider("Samples", 1, 8, 4)
if st.button("Generate", type="primary"):
    outcome = predict_image_i2i(n=n, root=PROJECT_ROOT, seed=42)
    if isinstance(outcome, Fail):
        st.error(f"{outcome.error.code}: {outcome.error.message}")
    else:
        assert isinstance(outcome, Ok)
        cols = st.columns(min(4, outcome.value.n))
        for i, b64 in enumerate(outcome.value.png_base64):
            cols[i % len(cols)].image(
                base64.b64decode(b64),
                caption=f"sample {i}",
                use_container_width=True,
            )
