import math
from typing import Dict, Any, List


def generate_report(results: Dict[str, Any], report_path: str):
    """Generate markdown report from sweep results."""
    data = results["results"]
    metadata = results["metadata"]

    lines = []
    lines.append("# Bubble Sweep Report")
    lines.append("")

    # Definitions
    lines.append("## Parameter Definitions")
    lines.append("")
    lines.append("- **R_m**: Bubble radius in meters")
    lines.append("- **dE_J**: Excess energy in joules")
    lines.append("- **tau_s**: Bubble lifetime in seconds")
    lines.append("- **bits**: Bekenstein bound on information content")
    lines.append("- **ops**: Lloyd limit on computational operations")
    lines.append("- **logP**: Natural log of formation probability")
    lines.append("- **F**: Toy figure of merit = log10(ops) + log10(P)")
    lines.append("")

    # Equations
    lines.append("## Equations Used")
    lines.append("")
    lines.append("### Bekenstein Bound")
    lines.append("```")
    lines.append("bits = (2 * pi * R * E) / (hbar * c * ln2)")
    lines.append("```")
    lines.append("")
    lines.append("### Lloyd Limit")
    lines.append("```")
    lines.append("ops_per_s = (2 * E) / (pi * hbar)")
    lines.append("ops = ops_per_s * tau")
    lines.append("```")
    lines.append("")
    lines.append("### Thermal Rarity Model")
    lines.append("```")
    lines.append("logP = -dE / (kB * T)")
    lines.append("```")
    lines.append("")
    lines.append("### Instanton A Rarity Model")
    lines.append("```")
    lines.append("logP = -(alpha * R * dE) / (hbar * c)")
    lines.append("```")
    lines.append("")

    # Sweep config
    cfg = metadata["config"]
    lines.append("## Sweep Configuration")
    lines.append("")
    lines.append(f"- Fast mode: {cfg['fast']}")
    lines.append(f"- R grid: {cfg['nR']} points from {cfg['R_range'][0]:.1e} to {cfg['R_range'][1]:.1e} m")
    lines.append(f"- dE grid: {cfg['nE']} points from {cfg['dE_range'][0]:.1e} to {cfg['dE_range'][1]:.1e} J")
    lines.append(f"- tau grid: {cfg['n_tau']} points from {cfg['tau_range'][0]:.1e} to {cfg['tau_range'][1]:.1e} s")
    lines.append("")

    # Filter finite results for ranking
    valid_data = [d for d in data if math.isfinite(d["F"])]

    # Top 10 by F
    lines.append("## Top 10 by F (expected ops weighted by rarity)")
    lines.append("")
    top_F = sorted(valid_data, key=lambda x: x["F"], reverse=True)[:10]
    lines.append(_make_table(top_F, ["R_m", "dE_J", "tau_s", "rarity_model", "log10_ops", "log10P", "F"]))
    lines.append("")

    # Top 5 by ops
    lines.append("## Top 5 by Operations")
    lines.append("")
    top_ops = sorted(valid_data, key=lambda x: x["log10_ops"], reverse=True)[:5]
    lines.append(_make_table(top_ops, ["R_m", "dE_J", "tau_s", "log10_ops", "ops"]))
    lines.append("")

    # Top 5 by bits
    lines.append("## Top 5 by Bits")
    lines.append("")
    top_bits = sorted(valid_data, key=lambda x: x["log10_bits"], reverse=True)[:5]
    lines.append(_make_table(top_bits, ["R_m", "dE_J", "log10_bits", "bits"]))
    lines.append("")

    # Limitations
    lines.append("## Limitations")
    lines.append("")
    lines.append("- This is a toy model for exploratory purposes only")
    lines.append("- Physical constants and formulas are simplified approximations")
    lines.append("- Rarity models are illustrative, not derived from first principles")
    lines.append("- The figure of merit F is not physically meaningful")
    lines.append("- Grid resolution may miss important parameter regimes")
    lines.append("- No error propagation or uncertainty quantification")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _make_table(rows: List[Dict], cols: List[str]) -> str:
    """Create a markdown table from rows."""
    if not rows:
        return "*No data*"

    lines = []
    # Header
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    # Rows
    for row in rows:
        cells = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                if abs(val) > 1e6 or (abs(val) < 1e-3 and val != 0):
                    cells.append(f"{val:.3e}")
                else:
                    cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)
