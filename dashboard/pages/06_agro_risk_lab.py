"""Agro Risk Lab — human-friendly demo of C4 soybean low-yield model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.application.agro_soy_predict import load_agro_soy_model_card, predict_agro_soy
from src.application.agro_validation_cases import AGRO_VALIDATION_CASES, case_by_id
from src.application.data_provenance_check import check_dataset_provenance
from src.application.human_agro_scorer import (
    AGRO_RESEARCH_DISCLAIMER,
    AgroMunicipalityProfile,
    WeatherBlockStats,
)
from src.shared.result import Fail, Ok

st.set_page_config(page_title="Agro Risk Lab", page_icon="🌱", layout="wide")

os.environ.setdefault("MLFLOW_DISABLE", "1")
os.environ.setdefault("QML_DEVICE", "cuda")

DATASET_ID = "acyd_soy_brazil_v1"
TIER_COLORS = {"low": "#33ff66", "moderate": "#ffb000", "high": "#ff3366"}


def _disclaimer_banner() -> None:
    st.markdown(
        f"""
        <div style="background:#0a2a10;border:2px solid #33ff66;border-radius:8px;
        padding:12px 16px;margin-bottom:12px;">
        <strong>Agro-climate research demo</strong> — {AGRO_RESEARCH_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _provenance_banner() -> None:
    prov = check_dataset_provenance(DATASET_ID, root=PROJECT_ROOT)
    st.caption(
        f"Dataset: `{DATASET_ID}` · verified: {'yes' if prov.verified else 'no'} · "
        f"license {prov.license_name}"
    )


def _risk_gauge(percent: float, tier: str) -> go.Figure:
    color = TIER_COLORS.get(tier, "#888")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percent,
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": "P(low yield)"},
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
        height=260,
        margin=dict(l=20, r=20, t=60, b=10),
    )
    return fig


def _show_score(result, *, title: str = "Model prediction") -> None:
    st.subheader(title)
    st.plotly_chart(_risk_gauge(result.risk_percent, result.risk_tier), use_container_width=True)
    st.markdown(f"**{result.risk_label}** — {result.human_summary}")
    st.markdown("**Top climate drivers (heuristic):**")
    for driver in result.top_drivers:
        st.caption(f"• **{driver.name}** — {driver.direction}: {driver.detail}")


st.title("Agro Risk Lab")
st.markdown("Brazilian soybean **low-yield probability** from climate + soil tabular features (ACYD C4).")
_disclaimer_banner()
_provenance_banner()

tab_cases, tab_custom, tab_science = st.tabs(
    ["Brazilian scenarios", "Custom municipality", "Science validation (exp_078)"]
)

with tab_cases:
    st.markdown("Eight hand-crafted scenarios from favorable pampa to severe MATOPIBA stress.")
    choice = st.selectbox(
        "Scenario",
        [f"{c.case_id} — {c.title}" for c in AGRO_VALIDATION_CASES],
    )
    case_id = choice.split(" — ", 1)[0]
    case = case_by_id(case_id)
    st.info(case.science_note)
    if st.button("Score scenario", type="primary"):
        outcome = predict_agro_soy(case.profile, log_prediction=False)
        if isinstance(outcome, Fail):
            st.error(outcome.error.message)
        else:
            assert isinstance(outcome, Ok)
            st.session_state["case_score"] = outcome.value
    if "case_score" in st.session_state:
        _show_score(st.session_state["case_score"], title=f"Scenario {case_id}")

with tab_custom:
    st.caption("Adjust municipality location and season climate aggregates.")
    c1, c2, c3 = st.columns(3)
    municipality = c1.text_input("Municipality", "Campo Novo do Parecis")
    state = c2.text_input("State (UF)", "MT")
    crop_year = c3.number_input("Crop year", min_value=2000, max_value=2030, value=2020)
    c4, c5, c6 = st.columns(3)
    latitude = c4.number_input("Latitude", value=-13.06, format="%.4f")
    longitude = c5.number_input("Longitude", value=-57.55, format="%.4f")
    area_ha = c6.number_input("Area harvested (ha)", min_value=100.0, value=10_000.0)
    c7, c8, c9 = st.columns(3)
    precip_mean = c7.number_input("Precip mean (mm/week)", value=3.5, format="%.2f")
    tmax_peak = c8.number_input("Tmax peak (K)", value=304.0, format="%.1f")
    ndvi_mean = c9.number_input("NDVI mean", value=3.0, format="%.2f")

    profile = AgroMunicipalityProfile(
        municipality=municipality,
        state=state,
        crop_year=int(crop_year),
        latitude=float(latitude),
        longitude=float(longitude),
        area_harvested_ha=float(area_ha),
        precipitation=WeatherBlockStats(precip_mean, 1.0, precip_mean * 0.3, precip_mean * 2.0),
        t2m_max=WeatherBlockStats(tmax_peak - 3.0, 2.0, tmax_peak - 6.0, tmax_peak),
        ndvi=WeatherBlockStats(ndvi_mean, 0.4, ndvi_mean - 0.8, ndvi_mean + 0.8),
    )

    if st.button("Score custom profile", type="primary"):
        outcome = predict_agro_soy(profile, log_prediction=True)
        if isinstance(outcome, Fail):
            st.error(outcome.error.message)
        else:
            assert isinstance(outcome, Ok)
            _show_score(outcome.value)

with tab_science:
    st.markdown(
        "Runs **exp_078** — checks whether C4 ranks all 8 scenarios with Spearman ρ ≥ 0.85."
    )
    if st.button("Run all 8 agro science cases", type="primary"):
        from experiments.exp_078_agro_clinical_cases.run import run_exp_078

        st.session_state["agro_science"] = run_exp_078(verbose=False)

    if "agro_science" in st.session_state:
        res = st.session_state["agro_science"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Spearman ρ", f"{res.spearman_rho:.3f}")
        m2.metric("Low vs high separation", f"{res.separation_pp:+.2f} pp")
        m3.metric("Verdict", "PASS ✅" if res.passed else "FAIL ❌")

        import pandas as pd

        rows = [
            {
                "Case": c.case_id,
                "Expected order": c.expected_rank,
                "Tier": c.expected_tier,
                "Model risk %": round(c.risk_percent, 2),
                "Profile": c.title,
            }
            for c in sorted(res.case_scores, key=lambda x: x.expected_rank)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("Model card"):
    card = load_agro_soy_model_card()
    st.markdown(card.markdown)

st.caption("Model: exp_060 LargeNanoMLP · ACYD Brazil soybean · C4 anchor")
