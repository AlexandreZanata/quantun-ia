"""Agro Maize Lab — distill nano + MC uncertainty + Cycle v2 floors (Phase E / E-T3)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.application.agro_maize_compare import compare_agro_maize
from src.application.agro_maize_predict import load_agro_maize_model_card
from src.application.data_provenance_check import check_dataset_provenance
from src.application.human_agro_scorer import (
    AGRO_RESEARCH_DISCLAIMER,
    AgroMunicipalityProfile,
    WeatherBlockStats,
)
from src.shared.result import Fail, Ok

st.set_page_config(page_title="Agro Maize Lab", page_icon="🌽", layout="wide")

os.environ.setdefault("MLFLOW_DISABLE", "1")
os.environ.setdefault("QML_DEVICE", "cuda")

DATASET_ID = "acyd_maize_brazil_v1"
TIER_COLORS = {"low": "#33ff66", "moderate": "#ffb000", "high": "#ff3366"}


def _disclaimer_banner() -> None:
    st.markdown(
        f"""
        <div style="background:#0a2a10;border:2px solid #33ff66;border-radius:8px;
        padding:12px 16px;margin-bottom:12px;">
        <strong>Maize agro-climate research demo</strong> — {AGRO_RESEARCH_DISCLAIMER.replace("soybean", "maize")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _risk_gauge(percent: float, tier: str, title: str = "P(low yield)") -> go.Figure:
    color = TIER_COLORS.get(tier, "#888")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percent,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 35], "color": "#0a2010"},
                    {"range": [35, 65], "color": "#2a2008"},
                    {"range": [65, 100], "color": "#2a0810"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#050805",
        font={"color": "#33ff66"},
        height=240,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    return fig


st.title("Agro Maize Lab")
st.markdown(
    "ACYD Brazil **maize** low-yield probability — live distill ResidualNano "
    "with MC-dropout uncertainty, beside published HistGB / quantum floors."
)
_disclaimer_banner()
prov = check_dataset_provenance(DATASET_ID, root=PROJECT_ROOT)
st.caption(
    f"Dataset: `{DATASET_ID}` · serve `{load_agro_maize_model_card().model_id}` · "
    f"verified: {'yes' if prov.verified else 'no'}"
)

col_l, col_r = st.columns(2)
with col_l:
    municipality = st.text_input("Municipality", value="Sorriso")
    state = st.text_input("State (UF)", value="MT", max_chars=2)
    crop_year = st.number_input("Crop year", min_value=2000, max_value=2035, value=2020)
    latitude = st.number_input("Latitude", value=-12.54)
    longitude = st.number_input("Longitude", value=-55.71)
with col_r:
    area = st.number_input("Area harvested (ha)", min_value=1.0, value=30000.0)
    precip = st.number_input("Precip mean", value=5.0)
    tmax = st.number_input("Tmax peak (K)", value=305.0)
    ndvi = st.number_input("NDVI mean", value=3.5)

if st.button("Score maize profile", type="primary"):
    profile = AgroMunicipalityProfile(
        municipality=municipality,
        state=state.upper(),
        crop_year=int(crop_year),
        latitude=float(latitude),
        longitude=float(longitude),
        area_harvested_ha=float(area),
        precipitation=WeatherBlockStats(precip, 1.0, max(0.1, precip * 0.3), precip * 2.0),
        t2m_max=WeatherBlockStats(tmax - 3.0, 2.0, tmax - 6.0, tmax),
        ndvi=WeatherBlockStats(ndvi, 0.4, max(0.5, ndvi - 0.8), ndvi + 0.8),
    )
    outcome = compare_agro_maize(profile, with_uncertainty=True)
    if isinstance(outcome, Fail):
        st.error(outcome.error.message)
    else:
        assert isinstance(outcome, Ok)
        st.session_state["maize_compare"] = outcome.value

if "maize_compare" in st.session_state:
    bundle = st.session_state["maize_compare"]
    live = bundle.live_nano
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Live distill nano")
        st.plotly_chart(
            _risk_gauge(live.risk_percent, live.risk_tier),
            use_container_width=True,
        )
        st.caption(f"Model: `{bundle.model_id}`")
        if live.uncertainty_std is not None:
            st.metric(
                "MC-dropout σ",
                f"{live.uncertainty_std:.4f}",
                help=f"method={live.uncertainty_method}; mean={live.mc_mean_probability}",
            )
        st.markdown(f"**{live.risk_label}** — {live.human_summary}")
    with c2:
        st.subheader("Published floors")
        for row in bundle.floors:
            if "Hist" in row.name or "distill" in row.name.lower():
                st.metric(row.name, f"{row.roc_auc:.4f}", help=row.notes)
    with c3:
        st.subheader("Quantum (publication)")
        q_rows = [r for r in bundle.floors if r.source.startswith("exp_086")]
        if not q_rows:
            st.info("No quantum floors in registry.")
        for row in q_rows:
            st.metric(row.name.split("·")[-1].strip(), f"{row.roc_auc:.4f}", help=row.notes)
        st.caption(bundle.quantum_note)

    st.markdown("**Top climate drivers (heuristic):**")
    for driver in live.top_drivers:
        st.caption(f"• **{driver.name}** — {driver.direction}: {driver.detail}")
