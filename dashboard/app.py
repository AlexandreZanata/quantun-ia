import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Quantum ML Lab", layout="wide")
st.title("Quantum ML Lab — Progress Dashboard")

LOGS = Path("logs/experiments.jsonl")


@st.cache_data(ttl=5)
def load_data():
    records = []
    if LOGS.exists():
        with open(LOGS) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


records = load_data()

if not records:
    st.warning("No experiments yet. Run: python experiments/exp_001_quantum_vs_classical/run.py")
    st.stop()

df = pd.DataFrame([
    {
        "Experiment": r["exp_id"],
        "Model": r["model_name"],
        "Accuracy": round(r.get("final_acc", 0) * 100, 2),
        "Final Loss": round(r.get("final_loss", 0), 4),
        "Time (s)": round(r.get("elapsed_s", 0), 1),
        "Epochs": r["n_epochs"],
        "Date": r["started_at"][:16],
    }
    for r in records
])

col1, col2, col3 = st.columns(3)
col1.metric("Total Experiments", len(records))
col2.metric("Best Accuracy", f"{df['Accuracy'].max():.1f}%")
col3.metric("Top Model", df.loc[df["Accuracy"].idxmax(), "Model"])

st.subheader("Model Comparison")
fig = px.bar(
    df,
    x="Model",
    y="Accuracy",
    color="Experiment",
    title="Accuracy by Model and Experiment",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Accuracy vs Training Time")
fig2 = px.scatter(
    df,
    x="Time (s)",
    y="Accuracy",
    color="Experiment",
    text="Model",
    size_max=20,
    title="Efficiency: best accuracy with lowest training time?",
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Learning Curves")
selected = st.multiselect(
    "Select models:",
    df["Model"].unique(),
    default=list(df["Model"].unique()[:4]),
)

fig3 = go.Figure()
for r in records:
    if r["model_name"] in selected:
        epochs = [h["epoch"] for h in r["history"]]
        accs = [h.get("accuracy", 0) for h in r["history"]]
        fig3.add_trace(go.Scatter(x=epochs, y=accs, name=r["model_name"], mode="lines"))

fig3.update_layout(title="Accuracy per Epoch", xaxis_title="Epoch", yaxis_title="Accuracy")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("All Results")
st.dataframe(df.sort_values("Accuracy", ascending=False), use_container_width=True)

if st.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()
