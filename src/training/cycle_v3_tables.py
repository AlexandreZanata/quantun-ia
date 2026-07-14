"""Export Research Cycle v3 LaTeX tables from curated publication metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "config" / "cycle_v3_paper.yaml"
DEFAULT_OUT = ROOT / "paper" / "tables"


def load_cycle_v3_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"invalid cycle_v3 registry: {path}"
        raise ValueError(msg)
    return payload


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def image_nano_i2i_latex(registry: dict[str, Any]) -> str:
    block = registry["image_nano_i2i"]
    lines = [
        "% Auto-generated Cycle v3 I2I table (Phase K)",
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{block['caption']}}}",
        f"\\label{{{block['label']}}}",
        "\\begin{tabular}{lrrll}",
        "\\toprule",
        "Model & FID & Floor & Verdict & Notes \\\\",
        "\\midrule",
    ]
    for row in block["rows"]:
        lines.append(
            f"{_escape(row['model'])} & {float(row['fid']):.2f} & "
            f"{_escape(row['reference'])} & {row['verdict']} & {_escape(row['notes'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def image_nano_t2i_latex(registry: dict[str, Any]) -> str:
    block = registry["image_nano_t2i"]
    lines = [
        "% Auto-generated Cycle v3 T2I table (Phase K)",
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{block['caption']}}}",
        f"\\label{{{block['label']}}}",
        "\\begin{tabular}{lrlrl}",
        "\\toprule",
        "Model & CLIP & Floor & Verdict & Notes \\\\",
        "\\midrule",
    ]
    for row in block["rows"]:
        lines.append(
            f"{_escape(row['model'])} & {float(row['clip']):.2f} & "
            f"{_escape(row['reference'])} & {row['verdict']} & {_escape(row['notes'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def quantum_v3_latex(registry: dict[str, Any]) -> str:
    block = registry["quantum_v3"]
    lines = [
        "% Auto-generated Cycle v3 quantum recipes table (Phase K)",
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{block['caption']}}}",
        f"\\label{{{block['label']}}}",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Model & Metric & Floor & Verdict & Notes \\\\",
        "\\midrule",
    ]
    for row in block["rows"]:
        lines.append(
            f"{_escape(row['model'])} & {_escape(row['metric'])} & "
            f"{_escape(row['reference'])} & {row['verdict']} & {_escape(row['notes'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def export_cycle_v3_tables(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    out_dir: Path = DEFAULT_OUT,
) -> dict[str, Path]:
    registry = load_cycle_v3_registry(registry_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "image_nano_i2i.tex": image_nano_i2i_latex,
        "image_nano_t2i.tex": image_nano_t2i_latex,
        "quantum_v3.tex": quantum_v3_latex,
    }
    written: dict[str, Path] = {}
    for name, fn in writers.items():
        path = out_dir / name
        path.write_text(fn(registry), encoding="utf-8")
        written[name] = path
    return written
