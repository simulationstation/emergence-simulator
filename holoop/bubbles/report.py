"""Reporting helpers for bubble sweeps."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional


def _format_row(entry: Dict) -> str:
    return (
        f"| R (m) | ΔE (J) | τ (s) | bits | ops | F | log10P |\n"
        f"| --- | --- | --- | --- | --- | --- | --- |\n"
        f"| {entry['R']:.3e} | {entry['E']:.3e} | {entry['tau']:.3e} | {entry['bits_max']:.3e} | {entry['ops_max']:.3e} | {entry['F']:.2f} | {entry['log10P']:.2f} |\n"
    )


def _top_entries(results: List[Dict], k: int = 8) -> List[Dict]:
    if not results:
        return []
    sorted_results = sorted(results, key=lambda r: (r["F"] if r.get("F") is not None else float("-inf")), reverse=True)
    return sorted_results[:k]


def write_report(
    results: Dict,
    path: str,
    terminality_info: Optional[Dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entries = results.get("results", [])
    summary = results.get("summary", {})
    lines = []
    lines.append("# Bubble model report (toy)\n")
    lines.append("This report summarizes a toy model of localized quantum fluctuations (‘bubbles’). It provides\n")
    lines.append("upper bounds on memory/ops, rarity estimates, and optional linkage to operational terminality metrics.\n\n")

    lines.append("## Complexity ceilings\n")
    lines.append("- Bits ceiling: I_max_bits = (2π R E) / (ħ c ln2)\n")
    lines.append("- Ops ceiling: ops/sec <= 2E / (π ħ); total_ops <= ops/sec * τ\n")
    lines.append("- Complexity proxy: min(log10 ops, log10(2)*bits)\n\n")

    lines.append("## Rarity models\n")
    lines.append("- Thermal-like: log P = -ΔE / (k_B T_env)\n")
    lines.append("- Instanton-like: log P = -B / ħ with B either α R ΔE / c or α ΔE² (R/c) / E_scale\n\n")

    lines.append("## Sweep highlights\n")
    lines.append(f"Entries evaluated: {summary.get('count', 0)}\n\n")
    lines.append("Top ops / bits / F:\n")
    for key in ["top_ops", "top_bits", "top_F"]:
        if key in summary:
            lines.append(f"- {key}: R={summary[key]['R']:.3e} m, ΔE={summary[key]['E']:.3e} J, τ={summary[key]['tau']:.3e} s, bits={summary[key]['bits_max']:.3e}, ops={summary[key]['ops_max']:.3e}, F={summary[key]['F']:.2f}\n")
    lines.append("\nRepresentative table (best F first):\n")
    lines.append("| R (m) | ΔE (J) | τ (s) | bits | ops | F | log10P |\n")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |\n")
    for entry in _top_entries(entries, k=8):
        lines.append(
            f"| {entry['R']:.3e} | {entry['E']:.3e} | {entry['tau']:.3e} | {entry['bits_max']:.3e} | {entry['ops_max']:.3e} | {entry['F']:.2f} | {entry['rarity'].get('log10P', float('-inf')):.2f} |\n"
        )
    lines.append("\nRarity drastically suppresses even large bubbles unless parameters are extreme. All values are ceilings, not capabilities.\n\n")

    if terminality_info:
        lines.append("## Terminality coupling\n")
        lines.append(
            f"A sample bubble activity trace was fed through the terminality estimator. Classification: {terminality_info.get('classification', 'n/a')}.\n"
        )
        lines.append("The activity proxy is based on ops/sec ceilings and is purely illustrative.\n\n")

    lines.append("## Limitations\n")
    lines.append("This is a toy model and not a prediction of cosmology or vacuum decay.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def append_main_report(main_report_path: str, bubble_results: Dict, bubble_json_path: str) -> None:
    if not os.path.exists(main_report_path):
        return
    entries = _top_entries(bubble_results.get("results", []), k=6)
    if not entries:
        return
    lines = []
    lines.append("\n## Bubble model summary\n")
    lines.append("A compact summary of the toy bubble sweep.\n\n")
    lines.append("| R (m) | ΔE (J) | τ (s) | bits | ops | F | log10P |\n")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |\n")
    for entry in entries:
        lines.append(
            f"| {entry['R']:.3e} | {entry['E']:.3e} | {entry['tau']:.3e} | {entry['bits_max']:.3e} | {entry['ops_max']:.3e} | {entry['F']:.2f} | {entry['rarity'].get('log10P', float('-inf')):.2f} |\n"
        )
    lines.append(f"\nDetailed JSON: `{bubble_json_path}`\n")
    with open(main_report_path, "a", encoding="utf-8") as f:
        f.writelines(lines)
