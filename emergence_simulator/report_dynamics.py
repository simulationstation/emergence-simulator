from typing import Dict, Any


def generate_dynamics_report(results: Dict[str, Any], report_path: str):
    """Generate markdown report for dynamics simulation."""
    sim = results["simulation"]
    params = sim["params"]
    metrics = results["metrics"]
    classification = metrics["classification"]
    summary = metrics["summary"]

    lines = []
    lines.append("# Bubble Dynamics Report")
    lines.append("")

    # Simulation parameters
    lines.append("## Simulation Parameters")
    lines.append("")
    lines.append(f"- Initial energy E0: {params['E0']:.3e} J")
    lines.append(f"- Initial radius R0: {params['R0']:.3e} m")
    lines.append(f"- Maximum radius Rmax: {params['Rmax']:.3e} m")
    lines.append(f"- Leak rate: {params['leak_rate']:.3e} s^-1")
    lines.append(f"- Growth timescale tgrow: {params['tgrow']:.3e} s")
    lines.append(f"- Simulation duration: {params['t_end']:.3e} s")
    lines.append("")

    # Dynamics equations
    lines.append("## Dynamics Equations")
    lines.append("")
    lines.append("### Energy Decay")
    lines.append("```")
    lines.append("E(t) = E0 * exp(-leak_rate * t)")
    lines.append("```")
    lines.append("")
    lines.append("### Radius Growth")
    lines.append("```")
    lines.append("R(t) = R0 + (Rmax - R0) * (1 - exp(-t / tgrow))")
    lines.append("```")
    lines.append("")
    lines.append("### Activity Proxy")
    lines.append("```")
    lines.append("f(t) = E(t) / E0  (normalized activity)")
    lines.append("ops_per_s(t) = 2 * E(t) / (pi * hbar)")
    lines.append("```")
    lines.append("")

    # Window capacity
    lines.append("## Window Capacity Analysis")
    lines.append("")
    lines.append("Window capacity measures integrated activity over [T, 2T]:")
    lines.append("```")
    lines.append("C_win(T) = integral from T to 2T of f(t) dt")
    lines.append("```")
    lines.append("")
    lines.append(f"- Tail slope (log-log): {classification['slope']:.3f}")
    lines.append(f"- Final C_win: {classification['final_C_win']:.3e}")
    lines.append("")

    # Classification
    lines.append("## Persistence Classification")
    lines.append("")
    lines.append(f"- **Activity Class**: {classification['activity_class']}")
    lines.append(f"- **Persistence Class**: {classification['persistence_class']}")
    lines.append(f"- **Rationale**: {classification['rationale']}")
    lines.append("")

    # Classification definitions
    lines.append("### Classification Definitions")
    lines.append("")
    lines.append("**Activity Classes:**")
    lines.append("- ContinuousActive: activity stays high (mean f > 0.5)")
    lines.append("- IntermittentActive: moderate activity (mean f > 0.1)")
    lines.append("- InstantaneousTerminal: activity drops quickly")
    lines.append("")
    lines.append("**Persistence Classes:**")
    lines.append("- Persistent: slope >= -1 (slow or no decay)")
    lines.append("- LongTailTerminal: -2 < slope < -1 (power-law decay)")
    lines.append("- Terminal: slope <= -2 (fast exponential-like decay)")
    lines.append("")

    # Summary statistics
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"- Mean normalized activity: {summary['mean_f']:.4f}")
    lines.append(f"- Final normalized activity: {summary['final_f']:.4e}")
    lines.append(f"- Total integrated activity: {summary['total_activity']:.3e}")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
