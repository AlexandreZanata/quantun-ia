"""Build C1–C4 classical vs quantum grand comparison from curated publication metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "config" / "grand_comparison_registry.yaml"


@dataclass(frozen=True)
class GrandComparisonCell:
    domain: str
    recipe: str
    delta_pp: float | None
    source_exp: str | None
    metric: str


@dataclass(frozen=True)
class GrandComparisonResult:
    domains: tuple[str, ...]
    recipes: tuple[str, ...]
    cells: tuple[GrandComparisonCell, ...]
    claim_win_delta_pp: float
    quantum_recipe_wins: dict[str, int]
    hypothesis_confirmed: bool
    pending_domains: dict[str, list[str]]


def load_grand_comparison_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _recipe_keys(registry: dict[str, Any]) -> list[str]:
    return list(registry.get("recipes", {}).keys())


def _domain_keys(registry: dict[str, Any]) -> list[str]:
    return list(registry.get("domains", {}).keys())


def build_grand_comparison_matrix(
    registry: dict[str, Any],
    *,
    claim_win_delta_pp: float = 0.5,
) -> GrandComparisonResult:
    domains = tuple(_domain_keys(registry))
    recipes = tuple(_recipe_keys(registry))
    cells: list[GrandComparisonCell] = []
    pending: dict[str, list[str]] = {}
    quantum_wins: dict[str, int] = {key: 0 for key in recipes}

    for recipe_key in recipes:
        recipe = registry["recipes"][recipe_key]
        source_map: dict[str, str | None] = recipe.get("source_exp", {})
        delta_map: dict[str, float | None] = recipe.get("delta_pp", {})
        is_quantum = bool(recipe.get("quantum", False))

        for domain in domains:
            metric = str(registry["domains"][domain]["metric"])
            if domain not in delta_map:
                continue
            raw_delta = delta_map.get(domain)
            source = source_map.get(domain)
            if raw_delta is None:
                pending.setdefault(recipe_key, []).append(domain)
                cells.append(
                    GrandComparisonCell(
                        domain=domain,
                        recipe=recipe_key,
                        delta_pp=None,
                        source_exp=source,
                        metric=metric,
                    )
                )
                continue

            delta = float(raw_delta)
            cells.append(
                GrandComparisonCell(
                    domain=domain,
                    recipe=recipe_key,
                    delta_pp=delta,
                    source_exp=source,
                    metric=metric,
                )
            )
            if is_quantum and delta >= claim_win_delta_pp:
                quantum_wins[recipe_key] += 1

    min_domains = 3
    hypothesis_confirmed = all(
        wins < min_domains for key, wins in quantum_wins.items() if registry["recipes"][key].get("quantum")
    )

    return GrandComparisonResult(
        domains=domains,
        recipes=recipes,
        cells=tuple(cells),
        claim_win_delta_pp=claim_win_delta_pp,
        quantum_recipe_wins=quantum_wins,
        hypothesis_confirmed=hypothesis_confirmed,
        pending_domains=pending,
    )


def grand_comparison_to_dict(
    result: GrandComparisonResult,
    registry: dict[str, Any],
) -> dict[str, Any]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for cell in result.cells:
        recipe_meta = registry["recipes"][cell.recipe]
        matrix.setdefault(cell.recipe, {})[cell.domain] = {
            "label": recipe_meta.get("label", cell.recipe),
            "kind": recipe_meta.get("kind"),
            "quantum": bool(recipe_meta.get("quantum", False)),
            "metric": cell.metric,
            "delta_pp": cell.delta_pp,
            "source_exp": cell.source_exp,
            "pending": cell.delta_pp is None,
        }

    return {
        "bench_id": "nano_grand_comparison",
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domains": registry.get("domains", {}),
        "claim_win_delta_pp": result.claim_win_delta_pp,
        "hypothesis": "No single quantum recipe beats classical anchor on >=3/4 domains",
        "hypothesis_confirmed": result.hypothesis_confirmed,
        "quantum_recipe_wins": result.quantum_recipe_wins,
        "pending_domains": result.pending_domains,
        "matrix": matrix,
    }


def export_grand_comparison_json(
    payload: dict[str, Any],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def _format_delta(value: float | None) -> str:
    if value is None:
        return "---"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}"


def export_grand_comparison_latex(
    result: GrandComparisonResult,
    registry: dict[str, Any],
    out_path: Path,
) -> Path:
    cross_domain = ("large_nano_vs_logistic", "large_nano_vs_conventional", "qnn_head_4q")
    lines = [
        "% Auto-generated grand comparison (exp_068)",
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Cross-domain comparison: $\\Delta$ pp vs reference (C1--C4 publication runs).}",
        "\\label{tab:grand_comparison}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Recipe & HIGGS & NIHR & GoBug & ACYD \\\\",
        "\\midrule",
    ]

    for recipe_key in cross_domain:
        recipe = registry["recipes"][recipe_key]
        row = [_latex_escape(str(recipe.get("label", recipe_key)))]
        delta_map = recipe.get("delta_pp", {})
        for domain in result.domains:
            row.append(_format_delta(delta_map.get(domain)))
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")
