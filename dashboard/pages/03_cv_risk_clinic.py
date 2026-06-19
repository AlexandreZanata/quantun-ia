"""CV Risk Clinic — human-friendly demo of the trained nano model."""

from __future__ import annotations

import os
import random
import sys
from dataclasses import replace
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.application.data_provenance_check import (
    check_dataset_provenance,
    validate_patient_profile,
)
from src.application.evaluate_serve_model import load_open_split_labeled
from src.application.human_cv_scorer import (
    RESEARCH_DISCLAIMER,
    PatientProfile,
    compare_profiles,
    features_to_profile,
    profile_summary,
    score_patient,
)
from src.shared.result import Fail, Ok

st.set_page_config(page_title="CV Risk Clinic", page_icon="❤️", layout="wide")

os.environ.setdefault("MLFLOW_DISABLE", "1")
os.environ.setdefault("QML_DEVICE", "cuda")

DATASET_ID = "synthea_cv_risk_v1"
BAND_COLORS = {"low": "#33ff66", "moderate": "#ffb000", "high": "#ff3366"}


def _provenance_banner() -> None:
    prov = check_dataset_provenance(DATASET_ID, root=PROJECT_ROOT)
    badge_color = "#4a2000" if prov.is_synthetic else "#0a2a10"
    st.markdown(
        f"""
        <div style="background:{badge_color};border:2px solid #ffb000;border-radius:8px;
        padding:12px 16px;margin-bottom:12px;">
        <strong>{prov.badge_label}</strong> — {prov.badge_help}<br/>
        <small>Manifest verified: {'✅ yes' if prov.verified else '⚠️ no'} ·
        source: {prov.source_mode or 'n/a'} · license {prov.license_name}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _how_to_read_box() -> None:
    with st.expander("How to read this clinic (30 seconds)", expanded=True):
        st.markdown(
            """
            **This is a research demo — not a doctor.**

            1. **Enter a patient profile** (age, BP, smoking, etc.) — same fields a nurse would chart.
            2. **The AI returns a risk %** — probability of a heart/stroke event in 12 months *in the Synthea simulator*.
            3. **Ignore the absolute number** — the training cohort is ~99% positive, so almost everyone scores high.
            4. **Use comparisons** — who is *higher* vs *lower* risk (mini game, what-if, science cases).

            ✅ **Good use:** "Patient B looks riskier than Patient A" · "Quitting smoking lowered the score"  
            ❌ **Bad use:** "This person has exactly 97% chance of a heart attack in real life"
            """
        )


def _show_profile_validation(profile: PatientProfile) -> None:
    validation = validate_patient_profile(profile)
    if validation.trust_level == "implausible":
        st.error(f"⚠️ Input check: {validation.summary}")
        for w in validation.warnings:
            st.caption(f"• {w}")
    elif validation.trust_level == "review":
        st.warning(f"Input check: {validation.summary}")
        for w in validation.warnings:
            st.caption(f"• {w}")
    else:
        st.success(f"✅ Input check: {validation.summary} ({validation.checks_passed}/{validation.checks_total})")


def _risk_gauge(percent: float, band: str, *, subtitle: str = "") -> go.Figure:
    color = BAND_COLORS.get(band, "#888")
    title = {"text": "Relative risk score" + (f"<br><sup>{subtitle}</sup>" if subtitle else "")}
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percent,
            number={"suffix": "%", "font": {"size": 42}},
            title=title,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#0a2010"},
                    {"range": [40, 70], "color": "#2a2008"},
                    {"range": [70, 100], "color": "#2a0810"},
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


def _profile_form(key: str, defaults: PatientProfile | None = None) -> PatientProfile:
    d = defaults or PatientProfile()
    st.caption("Fill in values as you would on a clinic form.")
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
    st.plotly_chart(
        _risk_gauge(result.risk_percent, result.risk_band, subtitle="compare patients, not absolute %"),
        use_container_width=True,
    )
    st.markdown(f"**{title}** — band: **{result.risk_label}**")
    st.markdown(result.human_summary)
    st.caption(
        f"Model probability = {result.probability:.4f} · "
        f"checkpoint `{Path(result.checkpoint_path).name}` · **synthetic cohort**"
    )


# ── Page header ───────────────────────────────────────────────────────────────
st.title("❤️ Cardiovascular Risk Clinic")
_provenance_banner()
_how_to_read_box()
st.caption(RESEARCH_DISCLAIMER)

tab_score, tab_whatif, tab_game, tab_random, tab_science = st.tabs(
    [
        "1 · Score a patient",
        "2 · What-if (before/after)",
        "3 · Mini game",
        "4 · Random record",
        "5 · Science check",
    ]
)

# ── Tab 1: Score ─────────────────────────────────────────────────────────────
with tab_score:
    st.markdown("Enter a profile and see how the model ranks this patient **relative to others**.")
    profile = _profile_form("main")
    _show_profile_validation(profile)
    if st.button("Calculate risk score", type="primary"):
        outcome = score_patient(profile)
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
        "Pick one change (quit smoking, new diabetes, etc.) and see **how many percentage points** "
        "the model score moves — that's the meaningful number here."
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

    if st.button("Compare before / after", type="primary"):
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
            if abs(delta) < 0.05:
                st.info("Almost no change — both profiles sit near the model ceiling on this synthetic cohort.")
            else:
                st.success(
                    f"**{change}** moved the score **{delta:+.1f} percentage points** "
                    f"({b.risk_percent:.1f}% → {a.risk_percent:.1f}%)."
                )

# ── Tab 3: Mini game ─────────────────────────────────────────────────────────
with tab_game:
    st.markdown(
        "**Who is higher risk?** Pick A or B, then reveal the model. "
        "You're testing whether your clinical intuition matches the AI **ranking**."
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
                prior_mi=True, atrial_fibrillation=True, family_history_cvd=True,
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

    if st.button("Reveal AI scores", type="primary"):
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
                    f"Correct — Patient **{higher}** ranks higher "
                    f"({ra.risk_percent:.1f}% vs {rb.risk_percent:.1f}%, gap {diff_pp:.1f} pp)."
                )
            else:
                st.error(
                    f"Not quite — Patient **{higher}** ranks higher "
                    f"({ra.risk_percent:.1f}% vs {rb.risk_percent:.1f}%, gap {diff_pp:.1f} pp)."
                )

# ── Tab 4: Random from dataset ───────────────────────────────────────────────
with tab_random:
    st.markdown(
        "Load one row from the **Synthea validation set** (synthetic EHR, not a real person). "
        "See whether the model prediction matches the **simulated** outcome label."
    )
    if st.button("Load random synthetic record", type="primary"):
        rows, labels = load_open_split_labeled(
            DATASET_ID,
            PROJECT_ROOT,
            split="val",
            n_rows=1,
            random_state=random.randint(0, 999_999),
        )
        profile = features_to_profile(rows[0])
        actual = labels[0]
        outcome = score_patient(profile)
        if isinstance(outcome, Fail):
            st.error(outcome.error.message)
        else:
            assert isinstance(outcome, Ok)
            r = outcome.value
            st.session_state["random_demo"] = (r, actual, profile)

    if "random_demo" in st.session_state:
        r, actual, profile = st.session_state["random_demo"]
        st.info("🧪 **Synthetic record** — generated by Synthea, not from a real hospital.")
        _show_profile_validation(profile)
        _show_score(r, title="Model prediction")
        actual_text = "Simulated CV event in 12 months" if actual == 1 else "No simulated CV event"
        predicted_text = "event predicted" if r.probability >= 0.5 else "no event predicted"
        match = (r.probability >= 0.5) == (actual == 1)
        st.markdown(f"**Simulated outcome in dataset:** {actual_text}")
        icon = "✅" if match else "⚠️"
        st.markdown(
            f"{icon} At 50% threshold: model **{predicted_text}** — "
            f"{'matches' if match else 'differs from'} the synthetic label."
        )

# ── Tab 5: Science validation (exp_041) ───────────────────────────────────────
with tab_science:
    st.markdown(
        "Eight profiles from cardiovascular epidemiology (Framingham / ACC-AHA). "
        "We check whether the model **orders** low-risk below high-risk cases."
    )
    from src.application.clinical_validation_cases import CLINICAL_VALIDATION_CASES

    with st.expander("Case list (L = low expected risk, H = high)"):
        for case in CLINICAL_VALIDATION_CASES:
            st.caption(f"**{case.case_id}** ({case.expected_tier}): {case.title}")

    if st.button("Run all 8 science cases", type="primary"):
        from experiments.exp_041_human_cv_clinical_cases.run import run_exp_041

        st.session_state["science_result"] = run_exp_041(verbose=False)

    if "science_result" in st.session_state:
        res = st.session_state["science_result"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Rank agreement (Spearman ρ)", f"{res.spearman_rho:.3f}", help="≥ 0.85 = pass")
        c2.metric("Low vs high separation", f"{res.separation_pp:+.2f} pp")
        c3.metric("Verdict", "PASS ✅" if res.passed else "FAIL ❌")

        import pandas as pd

        rows = [
            {
                "Case": c.case_id,
                "Expected order": c.expected_rank,
                "Epidemiology tier": c.expected_tier.replace("_", " "),
                "Model score %": round(c.risk_percent, 2),
                "Profile": c.title,
            }
            for c in sorted(res.case_scores, key=lambda x: x.expected_rank)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Full report: `experiments/exp_041_human_cv_clinical_cases/results.md`")

st.markdown("---")
if st.button("Verify dataset is synthetic (checksum + manifest)"):
    prov = check_dataset_provenance(DATASET_ID, root=PROJECT_ROOT)
    if prov.verified and prov.is_synthetic:
        st.success(
            f"✅ **{prov.badge_label}** — checksums match manifest. "
            f"Source mode: `{prov.source_mode or 'n/a'}`. No real patient data."
        )
    elif prov.is_synthetic:
        st.warning(f"Synthetic dataset identified but verification issues: {', '.join(prov.issues) or 'unknown'}")
    else:
        st.error(f"Unexpected origin: {prov.origin}")

st.caption(
    "Model: exp_034 LargeNanoMLP · Synthea CV v1 · "
    "Technical metrics → open **Model Lab** from the sidebar"
)
