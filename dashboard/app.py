"""Retro 90s benchmark dashboard — Quantum ML Lab."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.benchmark_data import best_row, load_records, to_benchmark_rows
from dashboard.terminal_report import print_benchmark_report

RETRO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    background-color: #050805 !important;
    color: #33ff66 !important;
}

.stApp {
    background: radial-gradient(ellipse at center, #0a120a 0%, #020402 100%);
}

/* CRT scanlines */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 9999;
    background: repeating-linear-gradient(
        0deg,
        rgba(0, 0, 0, 0.12) 0px,
        rgba(0, 0, 0, 0.12) 1px,
        transparent 1px,
        transparent 3px
    );
}

h1, h2, h3 {
    font-family: 'VT323', monospace !important;
    color: #33ff66 !important;
    text-shadow: 0 0 8px #33ff6644;
    letter-spacing: 2px;
}

.monitor-header {
    border: 1px solid #33ff66;
    box-shadow: 0 0 24px rgba(51, 255, 102, 0.15), inset 0 0 40px rgba(51, 255, 102, 0.03);
    background: linear-gradient(180deg, #0c140c 0%, #060a06 100%);
    padding: 0;
    margin-bottom: 1.5rem;
    overflow: hidden;
}

.monitor-titlebar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 1rem;
    background: #0f1a0f;
    border-bottom: 1px solid rgba(51, 255, 102, 0.35);
    font-family: 'VT323', monospace;
    font-size: 1.05rem;
    letter-spacing: 1px;
}

.monitor-titlebar .title {
    color: #33ff66;
    text-shadow: 0 0 10px rgba(51, 255, 102, 0.4);
}

.monitor-titlebar .badge {
    color: #050805;
    background: #33ff66;
    padding: 0.1rem 0.55rem;
    font-size: 0.85rem;
    letter-spacing: 2px;
}

.monitor-body {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0;
    border-bottom: 1px solid rgba(51, 255, 102, 0.2);
}

.monitor-stat {
    padding: 0.9rem 1rem;
    border-right: 1px solid rgba(51, 255, 102, 0.15);
}
.monitor-stat:last-child { border-right: none; }

.monitor-stat .label {
    font-size: 0.65rem;
    color: #1a992a;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

.monitor-stat .value {
    font-family: 'VT323', monospace;
    font-size: 1.35rem;
    color: #ffb000;
    text-shadow: 0 0 8px rgba(255, 176, 0, 0.3);
}

.monitor-console {
    padding: 0.75rem 1rem 1rem;
    font-size: 0.78rem;
    line-height: 1.65;
    color: #2dcc52;
}

.monitor-console .line-ok  { color: #33ff66; }
.monitor-console .line-dim { color: #1a7a28; }
.monitor-console .prompt  { color: #00e5ff; }

.retro-amber { color: #ffb000 !important; }
.retro-dim   { color: #1a992a !important; }
.retro-cyan  { color: #00e5ff !important; }

.stat-value {
    font-family: 'VT323', monospace;
    font-size: 2.4rem;
    color: #ffb000;
    text-shadow: 0 0 10px #ffb00055;
}

.stat-label {
    font-size: 0.75rem;
    color: #1a992a;
    text-transform: uppercase;
    letter-spacing: 3px;
}

div[data-testid="stMetric"] {
    background: #080f08;
    border: 1px solid #33ff6644;
    padding: 0.5rem;
}

.stButton > button {
    font-family: 'VT323', monospace !important;
    font-size: 1.2rem !important;
    background: #080f08 !important;
    color: #33ff66 !important;
    border: 2px solid #33ff66 !important;
    border-radius: 0 !important;
    letter-spacing: 2px;
}
.stButton > button:hover {
    background: #33ff66 !important;
    color: #050805 !important;
    box-shadow: 0 0 16px #33ff66;
}

[data-testid="stDataFrame"] {
    border: 1px solid #33ff6644;
}

hr { border-color: #33ff6633 !important; }
"""

PLOT_LAYOUT = dict(
    paper_bgcolor="#050805",
    plot_bgcolor="#080f08",
    font=dict(family="Share Tech Mono, monospace", color="#33ff66", size=12),
    xaxis=dict(gridcolor="#1a3a1a", zerolinecolor="#1a3a1a"),
    yaxis=dict(gridcolor="#1a3a1a", zerolinecolor="#1a3a1a"),
    legend=dict(bgcolor="#080f08", bordercolor="rgba(51, 255, 102, 0.27)"),
    margin=dict(l=40, r=20, t=50, b=40),
)

NEON_COLORS = ["#33ff66", "#ffb000", "#00e5ff", "#ff3366", "#cc66ff", "#ffff33"]


def retro_header(*, run_count: int, best_model: str | None, best_acc: float | None) -> None:
    best_label = f"{best_model} ({best_acc:.1f}%)" if best_model and best_acc is not None else "—"
    st.markdown(
        f"""
<div class="monitor-header">
  <div class="monitor-titlebar">
    <span class="title">◈ QUANTUN-IA BENCHMARK MONITOR</span>
    <span class="badge">v1.0 · ONLINE</span>
  </div>
  <div class="monitor-body">
    <div class="monitor-stat">
      <div class="label">Data source</div>
      <div class="value">experiments.jsonl</div>
    </div>
    <div class="monitor-stat">
      <div class="label">Runs loaded</div>
      <div class="value">{run_count}</div>
    </div>
    <div class="monitor-stat">
      <div class="label">Leader</div>
      <div class="value" style="font-size:1.05rem">{best_label}</div>
    </div>
  </div>
  <div class="monitor-console">
    <span class="prompt">quantun@lab:~$</span> load logs/experiments.jsonl<br>
    <span class="line-ok">  ✓ parsed {run_count} benchmark record{"s" if run_count != 1 else ""}</span><br>
    <span class="line-ok">  ✓ charts ready</span><br>
    <span class="line-dim">  ── retro terminal mode · quantum ml lab · 2026</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def accuracy_bar_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, exp in enumerate(df["exp_id"].unique()):
        sub = df[df["exp_id"] == exp]
        fig.add_trace(
            go.Bar(
                name=exp,
                x=sub["model"],
                y=sub["accuracy"],
                marker_color=NEON_COLORS[i % len(NEON_COLORS)],
                marker_line=dict(color="#33ff66", width=1),
            )
        )
    fig.update_layout(
        title=dict(text=">> ACCURACY BENCHMARKS", font=dict(size=16)),
        barmode="group",
        yaxis_title="ACC %",
        **PLOT_LAYOUT,
    )
    return fig


def efficiency_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, exp in enumerate(df["exp_id"].unique()):
        sub = df[df["exp_id"] == exp]
        fig.add_trace(
            go.Scatter(
                name=exp,
                x=sub["elapsed_s"],
                y=sub["accuracy"],
                mode="markers+text",
                text=sub["model"],
                textposition="top center",
                textfont=dict(size=9, color="#ffb000"),
                marker=dict(size=12, color=NEON_COLORS[i % len(NEON_COLORS)], line=dict(width=1, color="#fff")),
            )
        )
    fig.update_layout(
        title=dict(text=">> EFFICIENCY MAP  (time vs accuracy)", font=dict(size=16)),
        xaxis_title="TRAINING TIME (s)",
        yaxis_title="ACC %",
        **PLOT_LAYOUT,
    )
    return fig


def learning_curves(records: list[dict], selected: list[str]) -> go.Figure:
    fig = go.Figure()
    color_idx = 0
    for r in records:
        name = r["model_name"]
        if name not in selected:
            continue
        history = r.get("history", [])
        if not history or "accuracy" not in history[0]:
            continue
        epochs = [h["epoch"] for h in history]
        accs = [h.get("accuracy", 0) * 100 for h in history]
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=accs,
                name=name,
                mode="lines",
                line=dict(color=NEON_COLORS[color_idx % len(NEON_COLORS)], width=2),
            )
        )
        color_idx += 1
    fig.update_layout(
        title=dict(text=">> LEARNING CURVES", font=dict(size=16)),
        xaxis_title="EPOCH",
        yaxis_title="ACC %",
        **PLOT_LAYOUT,
    )
    return fig


def main() -> None:
    st.set_page_config(
        page_title="QUANTUN-IA // BENCHMARKS",
        page_icon="💾",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(f"<style>{RETRO_CSS}</style>", unsafe_allow_html=True)

    records = load_records()
    rows = to_benchmark_rows(records)
    best = best_row(rows)

    retro_header(
        run_count=len(rows),
        best_model=best["model"] if best else None,
        best_acc=best["accuracy"] if best else None,
    )
    st.markdown("## ◈ BENCHMARK RESULTS")

    if not rows:
        st.markdown(
            '<p class="retro-amber">&gt; ERROR: no benchmark data found.</p>',
            unsafe_allow_html=True,
        )
        st.code("python experiments/exp_001_quantum_vs_classical/run.py", language="bash")
        st.stop()

    df = pd.DataFrame(rows)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="stat-label">runs logged</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{len(rows)}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat-label">best accuracy</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="stat-value">{best["accuracy"]:.1f}%</div>' if best else '<div class="stat-value">—</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown('<div class="stat-label">top model</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="stat-value" style="font-size:1.4rem">{best["model"]}</div>' if best else "—",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown('<div class="stat-label">fastest run</div>', unsafe_allow_html=True)
        fastest = df.loc[df["elapsed_s"].idxmin()]
        st.markdown(
            f'<div class="stat-value" style="font-size:1.4rem">{fastest["elapsed_s"]:.2f}s</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(accuracy_bar_chart(df), use_container_width=True)
    with right:
        st.plotly_chart(efficiency_scatter(df), use_container_width=True)

    st.markdown("### ◈ LEARNING CURVES")
    selected = st.multiselect(
        "SELECT MODELS",
        options=df["model"].unique(),
        default=list(df["model"].unique()[:6]),
        label_visibility="collapsed",
    )
    st.plotly_chart(learning_curves(records, selected), use_container_width=True)

    st.markdown("### ◈ FULL BENCHMARK TABLE")
    table_cols = ["exp_id", "model", "accuracy", "loss", "elapsed_s", "epochs", "started_at"]
    display_df = (
        df[table_cols]
        .sort_values("accuracy", ascending=False)
        .rename(
            columns={
                "exp_id": "EXPERIMENT",
                "model": "MODEL",
                "accuracy": "ACC %",
                "loss": "LOSS",
                "elapsed_s": "TIME(s)",
                "epochs": "EPOCHS",
                "started_at": "DATE",
            }
        )
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown(
        '<p class="retro-dim">[F5] refresh &nbsp;|&nbsp; logs/experiments.jsonl &nbsp;|&nbsp; quantun-ia v0.1.0</p>',
        unsafe_allow_html=True,
    )

    if st.button("[ REFRESH DATA ]"):
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    print_benchmark_report()
    main()
