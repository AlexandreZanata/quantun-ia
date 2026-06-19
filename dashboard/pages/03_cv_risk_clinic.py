"""CV Risk Clinic — human-friendly demo of the trained nano model."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.application.evaluate_serve_model import load_open_split_labeled
from src.application.human_cv_scorer import (
    RESEARCH_DISCLAIMER,
    PatientProfile,
    compare_profiles,
    features_to_profile,
    profile_summary,
    score_patient,
)

st.set_page_config(page_title="CV Risk Clinic", page_icon="❤️", layout="wide")

os.environ.setdefault("MLFLOW_DISABLE", "1")
os.environ.setdefault("QML_DEVICE", "cuda")

BAND_COLORS = {"low": "#33ff66", "moderate": "#ffb000", "high": "#ff3366"}


def _risk_gauge(percent: float, band: str) -> go.Figure:
    color = BAND_COLORS.get(band, "#888")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=percent,
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": "12-month CV event risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#0a2010"},
                    {"range": [40, 70], "color": "#2a2008"},
                    {"range": [70, 100], "color": "#2a0810"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": percent},
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#050805",
        font={"color": "#33ff66"},
        height=280,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    return fig


def _profile_form(key: str, defaults: PatientProfile | None = None) -> PatientProfile:
    d = defaults or PatientProfile()
    st.markdown(f"**Patient profile** `{key}`")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age (years)", 25, 95, int(d.age_years), key=f"{key}_age")
        sex_male = st.radio("Sex", ["Female", "Male"], index=1 if d.sex_male else 0, key=f"{key}_sex")
        bmi = st.slider("BMI", 16.0, 45.0, float(d.bmi), 0.5, key=f"{key}_bmi")
        smoker = st.checkbox("Smoker", d.smoker, key=f"{key}_smoke")
        diabetes = st.checkbox("Diabetes", d.diabetes, key=f"{key}_diab")
    with c2:
        sbp = st.slider("Systolic BP (mmHg)", 90, 200, int(d.systolic_bp), key=f"{key}_sbp")
        dbp = st.slider("Diastolic BP (mmHg)", 50, 120, int(d.diastolic_bp), key=f"{key}_dbp")
        chol = st.slider("Total cholesterol (mg/dL)", 120, 320, int(d.total_cholesterol), key=f"{key}_chol")
        hdl = st.slider("HDL (mg/dL)", 25, 90, int(d.hdl), key=f"{key}_hdl")
    with c3:
        prior_mi = st.checkbox("Prior heart attack", d.prior_mi, key=f"{key}_mi")
        prior_stroke = st.checkbox("Prior stroke", d.prior_stroke, key=f"{key}_stroke")
        family_hx = st.checkbox("Family history of CVD", d.family_history_cvd, key=f"{key}_fam")
        afib = st.checkbox("Atrial fibrillation", d.atrial_fibrillation, key=f"{key}_afib")

    return PatientProfile(
        age_years=float(age),
        sex_male=sex_male == "Male",
        bmi=bmi,
        systolic_bp=float(sbp),
        diastolic_bp=float(dbp),
        total_cholesterol=float(chol),
        hdl=float(hdl),
        smoker=smoker,
        diabetes=diabetes,
        prior_mi=prior_mi,
        prior_stroke=prior_stroke,
        family_history_cvd=family_hx,
        atrial_fibrillation=afib,
    )


def _show_score(result, *, title: str = "Result") -> None:
    st.plotly_chart(_risk_gauge(result.risk_percent, result.risk_band), use_container_width=True)
    st.markdown(f"### {title}: {result.risk_label}")
    st.markdown(result.human_summary)
    st.caption(f"Model probability = {result.probability:.4f} · checkpoint `{result.checkpoint_path}`")


st.title("❤️ Cardiovascular Risk Clinic")
st.markdown(
    "Score **real patient profiles** with the trained **LargeNanoMLP** (~1.2M params, `exp_034`). "
    "You enter values humans understand — the model returns a **12-month heart/stroke risk %**."
)
st.warning(RESEARCH_DISCLAIMER)
st.info(
    "**How to read the score:** This synthetic cohort has a very high baseline event rate (~99%), "
    "so absolute percentages look high for most profiles. What matters for humans: "
    "**(1)** comparing two patients (mini game), **(2)** before/after what-if changes in percentage points, "
    "**(3)** whether the model agrees with the recorded outcome (random patient tab)."
)

tab_score, tab_whatif, tab_game, tab_random, tab_science = st.tabs(
    ["Score a patient", "What-if scenario", "Mini game: Who is higher risk?", "Random real patient", "Science validation (exp_041)"]
)

# ── Tab 1: Score ─────────────────────────────────────────────────────────────
with tab_score:
    profile = _profile_form("main")
    if st.button("CALCULATE MY RISK", type="primary"):
        outcome = score_patient(profile)
        from src.shared.result import Fail, Ok

        if isinstance(outcome, Fail):
            st.error(f"{outcome.error.code}: {outcome.error.message}")
        else:
            assert isinstance(outcome, Ok)
            st.session_state["last_score"] = outcome.value

    if "last_score" in st.session_state:
        _show_score(st.session_state["last_score"], title="Your patient")

# ── Tab 2: What-if ───────────────────────────────────────────────────────────
with tab_whatif:
    st.markdown(
        "**See how one change affects risk.** Start with a baseline profile, "
        "then flip a lifestyle or clinical factor and compare percentages side by side."
    )
    base = _profile_form("whatif_base")
    change = st.selectbox(
        "What if the patient…",
        [
            "Quits smoking",
            "Develops diabetes",
            "Has a heart attack (prior MI)",
            "Lowers systolic BP by 20 mmHg",
            "Loses weight (BMI −5)",
        ],
    )

    if st.button("COMPARE BEFORE / AFTER", type="primary"):
        modified = base
        if change == "Quits smoking":
            modified = replace(base, smoker=False)
            before_label, after_label = "Smoker", "Non-smoker"
        elif change == "Develops diabetes":
            modified = replace(base, diabetes=True)
            before_label, after_label = "No diabetes", "Diabetes"
        elif change == "Has a heart attack (prior MI)":
            modified = replace(base, prior_mi=True)
            before_label, after_label = "No prior MI", "Prior heart attack"
        elif change == "Lowers systolic BP by 20 mmHg":
            modified = replace(base, systolic_bp=max(90.0, base.systolic_bp - 20.0))
            before_label, after_label = f"BP {int(base.systolic_bp)}", f"BP {int(modified.systolic_bp)}"
        else:
            modified = replace(base, bmi=max(16.0, base.bmi - 5.0))
            before_label, after_label = f"BMI {base.bmi:.1f}", f"BMI {modified.bmi:.1f}"

        from src.shared.result import Fail, Ok

        before_out = score_patient(base)
        after_out = score_patient(modified)
        if isinstance(before_out, Fail) or isinstance(after_out, Fail):
            st.error("Scoring failed — is exp_034 checkpoint available?")
        else:
            assert isinstance(before_out, Ok) and isinstance(after_out, Ok)
            b, a = before_out.value, after_out.value
            delta = a.risk_percent - b.risk_percent
            c1, c2, c3 = st.columns(3)
            c1.metric(before_label, f"{b.risk_percent:.1f}%")
            c2.metric(after_label, f"{a.risk_percent:.1f}%")
            c3.metric("Change", f"{delta:+.1f} pp", delta_color="inverse")
            st.info(
                f"Changing **{change.lower()}** moved risk from **{b.risk_percent:.1f}%** "
                f"to **{a.risk_percent:.1f}%** ({delta:+.1f} percentage points)."
            )

# ── Tab 3: Mini game ─────────────────────────────────────────────────────────
with tab_game:
    st.markdown(
        "### 🎮 Doctor's challenge\n"
        "Two patients walk in. **You** pick who has the **higher** 12-month cardiovascular risk. "
        "Then the AI scores both — you see if your clinical intuition matches the model."
    )

    presets = {
        "Easy — young vs elderly smoker": (
            PatientProfile(age_years=32, sex_male=True, bmi=24, systolic_bp=118, smoker=False),
            PatientProfile(
                age_years=78, sex_male=True, bmi=31, systolic_bp=165,
                smoker=True, diabetes=True, prior_mi=True,
            ),
        ),
        "Medium — same age, different lifestyle": (
            PatientProfile(age_years=55, sex_male=False, bmi=23, systolic_bp=120, hdl=65, smoker=False),
            PatientProfile(
                age_years=55, sex_male=True, bmi=34, systolic_bp=152,
                hdl=38, smoker=True, diabetes=True,
            ),
        ),
        "Hard — both look sick": (
            PatientProfile(
                age_years=67, sex_male=True, bmi=29, systolic_bp=145,
                diabetes=True, prior_stroke=True,
            ),
            PatientProfile(
                age_years=71, sex_male=False, bmi=27, systolic_bp=138,
                prior_mi=True, afib=True, family_history_cvd=True,
            ),
        ),
    }

    scenario = st.selectbox("Scenario", list(presets.keys()))
    patient_a, patient_b = presets[scenario]

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Patient A**")
        st.write(profile_summary(patient_a))
    with cb:
        st.markdown("**Patient B**")
        st.write(profile_summary(patient_b))

    guess = st.radio("Who has HIGHER risk?", ["Patient A", "Patient B", "Equal / unsure"], horizontal=True)

    if st.button("REVEAL AI SCORES", type="primary"):
        from src.shared.result import Fail, Ok

        outcome = compare_profiles(patient_a, patient_b)
        if isinstance(outcome, Fail):
            st.error(outcome.error.message)
        else:
            assert isinstance(outcome, Ok)
            ra, rb = outcome.value
            higher = "A" if ra.probability > rb.probability else "B" if rb.probability > ra.probability else "Equal"
            correct = (
                (guess == "Patient A" and higher == "A")
                or (guess == "Patient B" and higher == "B")
                or (guess.startswith("Equal") and higher == "Equal")
            )

            c1, c2 = st.columns(2)
            with c1:
                _show_score(ra, title="Patient A")
            with c2:
                _show_score(rb, title="Patient B")

            diff_pp = abs(ra.risk_percent - rb.risk_percent)
            if correct:
                st.success(
                    f"✅ Correct! Patient **{higher}** is higher risk "
                    f"({ra.risk_percent:.1f}% vs {rb.risk_percent:.1f}%, Δ={diff_pp:.1f} pp)."
                )
            else:
                st.error(
                    f"❌ Not quite — Patient **{higher}** is higher risk "
                    f"({ra.risk_percent:.1f}% vs {rb.risk_percent:.1f}%, Δ={diff_pp:.1f} pp)."
                )

# ── Tab 4: Random from dataset ───────────────────────────────────────────────
with tab_random:
    st.markdown(
        "Load a **real row** from the Synthea validation set, see the human profile, "
        "and check what the model predicts vs what actually happened in the synthetic record."
    )
    if st.button("LOAD RANDOM PATIENT", type="primary"):
        rows, labels = load_open_split_labeled(
            "synthea_cv_risk_v1",
            PROJECT_ROOT,
            split="val",
            n_rows=1,
            random_state=random.randint(0, 999_999),
        )
        profile = features_to_profile(rows[0])
        actual = labels[0]
        outcome = score_patient(profile)
        from src.shared.result import Fail, Ok

        if isinstance(outcome, Fail):
            st.error(outcome.error.message)
        else:
            assert isinstance(outcome, Ok)
            r = outcome.value
            st.session_state["random_demo"] = (r, actual)

    if "random_demo" in st.session_state:
        r, actual = st.session_state["random_demo"]
        _show_score(r, title="Model prediction")
        actual_text = "CV event within 12 months" if actual == 1 else "No CV event"
        predicted_text = "event predicted" if r.probability >= 0.5 else "no event predicted"
        match = (r.probability >= 0.5) == (actual == 1)
        st.markdown(f"**What actually happened (synthetic record):** {actual_text}")
        icon = "✅" if match else "⚠️"
        st.markdown(
            f"{icon} Model threshold (50%): **{predicted_text}** — "
            f"{'matches' if match else 'differs from'} the recorded outcome."
        )

# ── Tab 5: Science validation (exp_041) ───────────────────────────────────────
with tab_science:
    st.markdown(
        "### 📋 Literature-backed cases (EXP 041)\n"
        "Eight fixed profiles ranked by cardiovascular epidemiology. "
        "Run `make exp-041-publication` to regenerate."
    )
    from src.application.clinical_validation_cases import CLINICAL_VALIDATION_CASES

    if st.button("RUN ALL 8 SCIENCE CASES", type="primary"):
        from experiments.exp_041_human_cv_clinical_cases.run import run_exp_041

        st.session_state["science_result"] = run_exp_041(verbose=False)

    if "science_result" in st.session_state:
        res = st.session_state["science_result"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Spearman ρ", f"{res.spearman_rho:.3f}")
        c2.metric("Tier separation", f"{res.separation_pp:+.2f} pp")
        c3.metric("Verdict", "PASS" if res.passed else "FAIL")

        import pandas as pd

        rows = [
            {
                "ID": c.case_id,
                "Rank": c.expected_rank,
                "Tier": c.expected_tier,
                "Risk %": round(c.risk_percent, 2),
                "Title": c.title,
            }
            for c in sorted(res.case_scores, key=lambda x: x.expected_rank)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Full report: `experiments/exp_041_human_cv_clinical_cases/results.md`")

st.markdown("---")
st.caption(
    "Model: `exp_034` · LargeNanoMLP · 700K Synthea CV train rows · RTX 4060 · "
    "[Model Lab](/Model_Lab) for technical metrics"
)
