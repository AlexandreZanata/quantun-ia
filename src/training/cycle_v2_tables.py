"""Export Research Cycle v2 LaTeX tables from curated publication metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "config" / "cycle_v2_paper.yaml"
DEFAULT_OUT = ROOT / "paper" / "tables"


def load_cycle_v2_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"invalid cycle_v2 registry: {path}"
        raise ValueError(msg)
    return payload


def _fmt_auc(value: float) -> str:
    return f"{value:.4f}"


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "---"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}"


def boosting_frontier_latex(registry: dict[str, Any]) -> str:
    block = registry["boosting_frontier"]
    lines = [
        "% Auto-generated Cycle v2 boosting frontier (exp_084 / exp_092)",
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{block['caption']}}}",
        f"\\label{{{block['label']}}}",
        "\\begin{tabular}{lrl}",
        "\\toprule",
        "Model & Val ROC-AUC & Notes \\\\",
        "\\midrule",
    ]
    for row in block["rows"]:
        lines.append(
            f"{row['model']} & {_fmt_auc(float(row['roc_auc']))} & {row['notes']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def sample_efficiency_latex(registry: dict[str, Any]) -> str:
    block = registry["sample_efficiency"]
    lines = [
        "% Auto-generated Cycle v2 sample-efficiency (exp_085)",
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{block['caption']} "
        f"AULC distill={_fmt_auc(float(block['aulc_distill']))}, "
        f"HistGB={_fmt_auc(float(block['aulc_histgb']))}.}}",
        f"\\label{{{block['label']}}}",
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "Budget (\\%) & Train rows & HistGB & Hard nano & Distill nano \\\\",
        "\\midrule",
    ]
    for row in block["rows"]:
        lines.append(
            f"{int(row['budget_pct'])} & {int(row['train_rows'])} & "
            f"{_fmt_auc(float(row['histgb']))} & "
            f"{_fmt_auc(float(row['hard_nano']))} & "
            f"{_fmt_auc(float(row['distill_nano']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def quantum_v2_latex(registry: dict[str, Any]) -> str:
    block = registry["quantum_v2"]
    lines = [
        "% Auto-generated Cycle v2 quantum recipes (exp_086 / exp_087 / exp_088 / exp_091 / exp_093)",
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{block['caption']}}}",
        f"\\label{{{block['label']}}}",
        "\\begin{tabular}{llrl}",
        "\\toprule",
        "Exp & Arm & Val ROC-AUC & $\\Delta$ pp \\\\",
        "\\midrule",
    ]
    for row in block["rows"]:
        delta = row.get("delta_pp")
        delta_f = float(delta) if delta is not None else None
        lines.append(
            f"{row['experiment']} & {row['arm']} & "
            f"{_fmt_auc(float(row['roc_auc']))} & {_fmt_delta(delta_f)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def multicrop_latex(registry: dict[str, Any]) -> str:
    block = registry["multicrop"]
    lines = [
        "% Auto-generated Cycle v2 multi-crop (exp_090)",
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{block['caption']}}}",
        f"\\label{{{block['label']}}}",
        "\\begin{tabular}{lrl}",
        "\\toprule",
        "Model & Val ROC-AUC & Notes \\\\",
        "\\midrule",
    ]
    for row in block["rows"]:
        lines.append(
            f"{row['model']} & {_fmt_auc(float(row['roc_auc']))} & {row['notes']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def hard_drift_latex(registry: dict[str, Any]) -> str:
    block = registry["hard_drift"]
    lines = [
        "% Auto-generated Cycle v2 hard temporal drift (exp_094)",
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{block['caption']}}}",
        f"\\label{{{block['label']}}}",
        "\\begin{tabular}{lrl}",
        "\\toprule",
        "Model & Val ROC-AUC & Notes \\\\",
        "\\midrule",
    ]
    for row in block["rows"]:
        lines.append(
            f"{row['model']} & {_fmt_auc(float(row['roc_auc']))} & {row['notes']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def export_cycle_v2_tables(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    out_dir: Path = DEFAULT_OUT,
) -> list[Path]:
    """Write Cycle v2 paper tables; returns created paths."""
    registry = load_cycle_v2_registry(registry_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "boosting_frontier.tex": boosting_frontier_latex,
        "sample_efficiency.tex": sample_efficiency_latex,
        "quantum_v2.tex": quantum_v2_latex,
        "multicrop.tex": multicrop_latex,
        "hard_drift.tex": hard_drift_latex,
    }
    created: list[Path] = []
    for name, fn in writers.items():
        path = out_dir / name
        path.write_text(fn(registry), encoding="utf-8")
        created.append(path)
    return created
