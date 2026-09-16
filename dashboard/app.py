"""Read-only monitor for the retained scientific evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research" / "evidence_registry.json"


@st.cache_data
def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(
        page_title="QUANTUN-IA — Evidências",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    registry = load_registry()
    studies = registry["active_experiments"]

    st.title("QUANTUN-IA — EVIDÊNCIAS AUDITADAS")
    st.warning(
        "Nenhum estudo ativo demonstra vantagem quântica ou validação clínica. "
        "Os itens abaixo são evidências internas reproduzíveis, com limitações."
    )
    left, middle, right = st.columns(3)
    left.metric("Estudos ativos", len(studies))
    middle.metric("Experimentos arquivados", registry["excluded_from_active_claims"]["count"])
    right.metric("Alegações de vantagem", 0)

    rows = [
        {
            "Experimento": item["exp_id"],
            "Dataset": item["dataset"],
            "Seeds": item["seeds"],
            "Δ primário (pp)": item["delta_pp"],
            "p": item["p_value"],
            "Cohen d": item["effect_size_cohens_d"],
        }
        for item in studies
    ]
    st.subheader("Registro ativo")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Conclusões permitidas")
    for item in studies:
        with st.expander(f"{item['exp_id']} — {item['dataset']}"):
            st.write(item["supported_statement"])
            st.caption("Limitações: " + "; ".join(item["limitations"]))

    st.info(
        "Próxima etapa: pré-registrar o programa Q-Shift descrito em "
        "docs/RESEARCH_RESET.md. Este painel não treina, publica nem altera modelos."
    )


if __name__ == "__main__":
    main()
