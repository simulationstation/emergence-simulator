"""Report generation for comprehensive sweep results."""

import math
from typing import Dict, Any, List
from collections import Counter


def generate_sweep_report(results: Dict[str, Any], report_path: str):
    """Generate markdown report for comprehensive sweep."""
    data = results["results"]
    metadata = results["metadata"]
    config = metadata["config"]

    lines = []
    lines.append("# Comprehensive Bubble Sweep Report")
    lines.append("")

    # Summary of sweep
    lines.append("## Sweep Summary")
    lines.append("")
    lines.append(f"- Total grid points: {metadata['total_points']}")
    lines.append(f"- R0 grid: {config['nR']} points from {config['R0_range'][0]:.1e} to {config['R0_range'][1]:.1e} m")
    lines.append(f"- dE grid: {config['nE']} points from {config['dE_range'][0]:.1e} to {config['dE_range'][1]:.1e} J")
    lines.append(f"- tau grid: {config['nTau']} points from {config['tau_range'][0]:.1e} to {config['tau_range'][1]:.1e} s")

    feed_mode = config.get('feed_mode', 'constant')
    lines.append(f"- Feed mode: {feed_mode}")

    if feed_mode == "decay":
        q_vals = config.get('q_vals', [1.0])
        t0_fracs = config.get('t0_fracs', [0.1])
        eta0_vals = config.get('eta0_vals', [1e-4])
        lines.append(f"- q values (power-law exponent): {q_vals}")
        lines.append(f"- t0_fracs (t0/tau): {t0_fracs}")
        lines.append(f"- eta0 values (initial coupling): {eta0_vals}")
    else:
        eta_vals = config.get('eta_vals', [0.0])
        lines.append(f"- eta values: {eta_vals}")

    lines.append(f"- Fast mode: {config['fast']}")
    lines.append(f"- Random seed: {config['seed']}")
    lines.append("")

    # Rarity models
    lines.append("## Rarity Models")
    lines.append("")
    lines.append(f"1. **Thermal**: T_env = {config['T_env_K']} K")
    lines.append(f"2. **Instanton A**: alpha = {config['alpha']}")
    lines.append(f"3. **Instanton B**: alpha = {config['alpha']}, E_scale = {config['E_scale_J']:.1e} J")
    lines.append("")

    # Filter valid data
    valid_F = [d for d in data if math.isfinite(d["F_inst_a"])]

    # Top 10 by F_inst_a
    lines.append("## Top 10 by F_inst_a (Weighted Compute)")
    lines.append("")
    top_F = sorted(valid_F, key=lambda x: x["F_inst_a"], reverse=True)[:10]
    lines.append(_make_table(top_F, [
        "R0_m", "dE_J", "tau_s", "log10_ops", "log10P_inst_a", "F_inst_a",
        "persistence_class", "activity_class"
    ]))
    lines.append("")

    # Top 10 by ops
    lines.append("## Top 10 by Operations")
    lines.append("")
    top_ops = sorted(valid_F, key=lambda x: x["log10_ops"], reverse=True)[:10]
    lines.append(_make_table(top_ops, [
        "R0_m", "dE_J", "tau_s", "log10_ops", "log10P_inst_a", "log10P_thermal",
        "persistence_class"
    ]))
    lines.append("")

    # Persistence class counts
    lines.append("## Persistence Classification Distribution")
    lines.append("")

    persistence_counts = Counter(d["persistence_class"] for d in data)
    activity_counts = Counter(d["activity_class"] for d in data)

    lines.append("### Persistence Classes")
    lines.append("")
    lines.append("| Class | Count | Fraction |")
    lines.append("| --- | --- | --- |")
    total = len(data)
    for cls in ["Persistent", "LongTailTerminal", "Terminal"]:
        count = persistence_counts.get(cls, 0)
        frac = count / total if total > 0 else 0
        lines.append(f"| {cls} | {count} | {frac:.2%} |")
    lines.append("")

    lines.append("### Activity Classes")
    lines.append("")
    lines.append("| Class | Count | Fraction |")
    lines.append("| --- | --- | --- |")
    for cls in ["ContinuousActive", "IntermittentActive", "InstantaneousTerminal"]:
        count = activity_counts.get(cls, 0)
        frac = count / total if total > 0 else 0
        lines.append(f"| {cls} | {count} | {frac:.2%} |")
    lines.append("")

    # Persistence counts per feed parameter (eta for constant, q/eta0 for decay)
    feed_mode = config.get('feed_mode', 'constant')

    if feed_mode == "decay":
        # Decay mode: show q_summary and eta0_summary
        q_summary = metadata.get("q_summary", {})
        if q_summary:
            lines.append("### Persistence Counts by q (Power-Law Exponent)")
            lines.append("")
            lines.append("q controls how quickly the background feed decays: g(t) = 1/(1 + t/t0)^q")
            lines.append("")
            lines.append("| q | Persistent | LongTailTerminal | Terminal | Total |")
            lines.append("| --- | --- | --- | --- | --- |")
            for q_str, counts in sorted(q_summary.items(), key=lambda x: float(x[0])):
                lines.append(f"| {float(q_str):.1f} | {counts['Persistent']} | {counts['LongTailTerminal']} | {counts['Terminal']} | {counts['total']} |")
            lines.append("")

        eta0_summary = metadata.get("eta0_summary", {})
        if eta0_summary:
            lines.append("### Persistence Counts by eta0 (Initial Feed Coupling)")
            lines.append("")
            lines.append("eta0 sets the initial amplitude of the decaying background feed.")
            lines.append("")
            lines.append("| eta0 | Persistent | LongTailTerminal | Terminal | Total |")
            lines.append("| --- | --- | --- | --- | --- |")
            for eta0_str, counts in sorted(eta0_summary.items(), key=lambda x: float(x[0])):
                lines.append(f"| {float(eta0_str):.0e} | {counts['Persistent']} | {counts['LongTailTerminal']} | {counts['Terminal']} | {counts['total']} |")
            lines.append("")
    else:
        # Constant mode: show eta_summary
        eta_summary = metadata.get("eta_summary", {})
        if eta_summary:
            lines.append("### Persistence Counts by Eta (Background Feed)")
            lines.append("")
            lines.append("Eta represents the background feed coupling constant. Higher eta sustains activity longer.")
            lines.append("")
            lines.append("| eta | Persistent | LongTailTerminal | Terminal | Total |")
            lines.append("| --- | --- | --- | --- | --- |")
            for eta_str, counts in sorted(eta_summary.items(), key=lambda x: float(x[0])):
                eta_label = f"{float(eta_str):.0e}" if float(eta_str) > 0 else "0"
                lines.append(f"| {eta_label} | {counts['Persistent']} | {counts['LongTailTerminal']} | {counts['Terminal']} | {counts['total']} |")
            lines.append("")

    # Cross-tabulation: persistence vs rarity regime
    lines.append("## Persistence by Rarity Regime")
    lines.append("")
    lines.append("Binning points by log10P_inst_a into low/medium/high rarity:")
    lines.append("")

    # Bin by log10P
    log10P_vals = [d["log10P_inst_a"] for d in data if math.isfinite(d["log10P_inst_a"])]
    if log10P_vals:
        p33 = sorted(log10P_vals)[len(log10P_vals) // 3]
        p66 = sorted(log10P_vals)[2 * len(log10P_vals) // 3]

        bins = {"High Rarity (low P)": [], "Medium Rarity": [], "Low Rarity (high P)": []}
        for d in data:
            if not math.isfinite(d["log10P_inst_a"]):
                continue
            if d["log10P_inst_a"] < p33:
                bins["High Rarity (low P)"].append(d)
            elif d["log10P_inst_a"] < p66:
                bins["Medium Rarity"].append(d)
            else:
                bins["Low Rarity (high P)"].append(d)

        lines.append("| Rarity Regime | Persistent | LongTailTerminal | Terminal |")
        lines.append("| --- | --- | --- | --- |")
        for regime, points in bins.items():
            counts = Counter(d["persistence_class"] for d in points)
            lines.append(f"| {regime} | {counts.get('Persistent', 0)} | {counts.get('LongTailTerminal', 0)} | {counts.get('Terminal', 0)} |")
        lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")

    # Analyze the data to make observations
    obs_lines = _generate_interpretation(data, persistence_counts, valid_F)
    lines.extend(obs_lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _generate_interpretation(data: List[Dict], persistence_counts: Counter, valid_F: List[Dict]) -> List[str]:
    """Generate interpretation based on observed data."""
    lines = []

    # Observation 1: Rarity vs complexity at extremes
    lines.append("### Rarity Dominates at Extremes")
    lines.append("")

    # Check if high-ops points have very low P
    if valid_F:
        top_ops = sorted(valid_F, key=lambda x: x["log10_ops"], reverse=True)[:10]
        avg_logP_top_ops = sum(d["log10P_inst_a"] for d in top_ops) / len(top_ops)

        bottom_ops = sorted(valid_F, key=lambda x: x["log10_ops"])[:10]
        avg_logP_bottom_ops = sum(d["log10P_inst_a"] for d in bottom_ops) / len(bottom_ops)

        lines.append(f"- Points with highest ops (avg log10P = {avg_logP_top_ops:.1f}) tend to have ")
        if avg_logP_top_ops < avg_logP_bottom_ops:
            lines.append("  lower probability, confirming that rarity dominates at high complexity.")
        else:
            lines.append("  similar or higher probability than low-ops points.")
        lines.append("")

    # Observation 2: Persistence separability
    lines.append("### Persistence is Separable from Rarity/Complexity")
    lines.append("")

    # Check if persistence class correlates with rarity
    persistent_pts = [d for d in data if d["persistence_class"] == "Persistent" and math.isfinite(d["log10P_inst_a"])]
    terminal_pts = [d for d in data if d["persistence_class"] == "Terminal" and math.isfinite(d["log10P_inst_a"])]

    if persistent_pts and terminal_pts:
        avg_P_persistent = sum(d["log10P_inst_a"] for d in persistent_pts) / len(persistent_pts)
        avg_P_terminal = sum(d["log10P_inst_a"] for d in terminal_pts) / len(terminal_pts)

        avg_ops_persistent = sum(d["log10_ops"] for d in persistent_pts) / len(persistent_pts)
        avg_ops_terminal = sum(d["log10_ops"] for d in terminal_pts) / len(terminal_pts)

        lines.append(f"- Persistent points: avg log10P = {avg_P_persistent:.1f}, avg log10_ops = {avg_ops_persistent:.1f}")
        lines.append(f"- Terminal points: avg log10P = {avg_P_terminal:.1f}, avg log10_ops = {avg_ops_terminal:.1f}")

        # Check overlap
        p_diff = abs(avg_P_persistent - avg_P_terminal)
        ops_diff = abs(avg_ops_persistent - avg_ops_terminal)

        if p_diff < 50 and ops_diff < 10:
            lines.append("- Persistence classification shows substantial overlap across rarity/complexity regimes,")
            lines.append("  suggesting it captures an independent dimension of bubble behavior.")
        else:
            lines.append("- There is some correlation between persistence and rarity/complexity,")
            lines.append("  but persistence still provides additional classification information.")
    else:
        lines.append("- Insufficient data to compare persistence classes.")

    lines.append("")

    # Summary
    lines.append("### Summary")
    lines.append("")
    total = sum(persistence_counts.values())
    if total > 0:
        persistent_frac = persistence_counts.get("Persistent", 0) / total
        terminal_frac = persistence_counts.get("Terminal", 0) / total
        lines.append(f"- {persistent_frac:.1%} of grid points classified as Persistent")
        lines.append(f"- {terminal_frac:.1%} of grid points classified as Terminal")
        lines.append("- The F metric (log10_ops + log10P) provides a useful single-number summary")
        lines.append("  but loses information about persistence behavior.")

    lines.append("")
    return lines


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
                if not math.isfinite(val):
                    cells.append("-inf")
                elif abs(val) > 1e6 or (abs(val) < 1e-3 and val != 0):
                    cells.append(f"{val:.2e}")
                else:
                    cells.append(f"{val:.2f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)
