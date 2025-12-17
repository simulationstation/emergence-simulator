"""Master report generation consolidating all artifacts."""

import json
import math
import os
from typing import Dict, Any, List, Optional


def generate_master_report(outdir: str, report_path: str):
    """Generate consolidated master report from existing artifacts."""
    # Load available artifacts
    dynamics_data = _load_json(os.path.join(outdir, "dynamics", "dynamics_results.json"))
    sweep_data = _load_json(os.path.join(outdir, "sweep", "sweep_results.json"))
    bubble_data = _load_json(os.path.join(outdir, "bubbles", "bubble_results.json"))

    lines = []

    # Title
    lines.append("# Emergence Simulator: Master Report")
    lines.append("")

    # Executive Summary
    lines.extend(_section_executive_summary(dynamics_data, sweep_data))

    # Knobs
    lines.extend(_section_knobs())

    # Complexity Ceilings
    lines.extend(_section_complexity_ceilings())

    # Rarity Models
    lines.extend(_section_rarity_models())

    # Persistence Taxonomy
    lines.extend(_section_persistence_taxonomy())

    # Results Tables
    lines.extend(_section_results_tables(dynamics_data, sweep_data, bubble_data))

    # How to Interpret Results
    lines.extend(_section_interpretation_guide())

    # Limitations
    lines.extend(_section_limitations())

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load JSON file if it exists."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _section_executive_summary(dynamics_data: Optional[Dict], sweep_data: Optional[Dict]) -> List[str]:
    """Generate executive summary section."""
    lines = []
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This report consolidates results from the emergence-simulator bubble modeling framework.")
    lines.append("The simulator explores hypothetical vacuum bubble configurations across three dimensions:")
    lines.append("")
    lines.append("1. **Complexity ceilings**: Physical limits on information storage (Bekenstein) and computation (Lloyd)")
    lines.append("2. **Formation rarity**: Probability estimates under thermal and instanton models")
    lines.append("3. **Persistence**: Classification of bubble lifetime behavior based on activity decay")
    lines.append("")

    if sweep_data:
        meta = sweep_data.get("metadata", {}).get("config", {})
        total = sweep_data.get("metadata", {}).get("total_points", 0)
        lines.append(f"The comprehensive sweep evaluated {total} parameter combinations.")
        lines.append("")

    if dynamics_data:
        classification = dynamics_data.get("metrics", {}).get("classification", {})
        if classification:
            lines.append(f"Representative bubble classification: **{classification.get('persistence_class', 'N/A')}** ({classification.get('activity_class', 'N/A')})")
            lines.append("")

    return lines


def _section_knobs() -> List[str]:
    """Generate knobs definition section."""
    lines = []
    lines.append("## Parameter Definitions (Knobs)")
    lines.append("")
    lines.append("| Parameter | Symbol | Units | Description |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| Initial radius | R0 | m | Bubble radius at formation |")
    lines.append("| Excess energy | dE | J | Energy above ambient vacuum |")
    lines.append("| Lifetime | tau | s | Characteristic timescale |")
    lines.append("| Leak rate | leak_rate | s^-1 | Energy dissipation rate (derived from tau) |")
    lines.append("| Maximum radius | Rmax | m | Asymptotic radius after expansion |")
    lines.append("| Growth timescale | tgrow | s | Characteristic time for radius growth |")
    lines.append("")
    lines.append("**Derived relationships:**")
    lines.append("")
    lines.append("- `leak_rate = -ln(f_end) / tau` where f_end is target final activity (default 1e-3)")
    lines.append("- `Rmax = 10 * R0` (default)")
    lines.append("- `tgrow = 0.1 * tau` (default)")
    lines.append("")
    return lines


def _section_complexity_ceilings() -> List[str]:
    """Generate complexity ceilings section."""
    lines = []
    lines.append("## Complexity Ceilings")
    lines.append("")
    lines.append("### Bekenstein Bound (Information Storage)")
    lines.append("")
    lines.append("Maximum bits that can be stored in a region of radius R with energy E:")
    lines.append("")
    lines.append("```")
    lines.append("bits = (2 * pi * R * E) / (hbar * c * ln(2))")
    lines.append("```")
    lines.append("")
    lines.append("This represents the fundamental limit on information content imposed by quantum mechanics and general relativity.")
    lines.append("")
    lines.append("### Lloyd Limit (Computation)")
    lines.append("")
    lines.append("Maximum operations per second given energy E:")
    lines.append("")
    lines.append("```")
    lines.append("ops_per_s = (2 * E) / (pi * hbar)")
    lines.append("ops_total = ops_per_s * tau")
    lines.append("```")
    lines.append("")
    lines.append("This bounds the rate of logical operations based on available energy.")
    lines.append("")
    lines.append("### Constants Used")
    lines.append("")
    lines.append("| Constant | Value | Units |")
    lines.append("| --- | --- | --- |")
    lines.append("| c (speed of light) | 2.998e8 | m/s |")
    lines.append("| hbar (reduced Planck) | 1.055e-34 | J·s |")
    lines.append("| kB (Boltzmann) | 1.381e-23 | J/K |")
    lines.append("")
    return lines


def _section_rarity_models() -> List[str]:
    """Generate rarity models section."""
    lines = []
    lines.append("## Rarity Models")
    lines.append("")
    lines.append("Formation probability is estimated using log-probability to avoid numerical underflow.")
    lines.append("")
    lines.append("### Thermal Model")
    lines.append("")
    lines.append("Boltzmann suppression at ambient temperature T:")
    lines.append("")
    lines.append("```")
    lines.append("logP = -dE / (kB * T)")
    lines.append("```")
    lines.append("")
    lines.append("Default: T = 2.7 K (CMB temperature)")
    lines.append("")
    lines.append("### Instanton Model A")
    lines.append("")
    lines.append("Semiclassical tunneling with linear R-E coupling:")
    lines.append("")
    lines.append("```")
    lines.append("logP = -(alpha * R * dE) / (hbar * c)")
    lines.append("```")
    lines.append("")
    lines.append("Default: alpha = 1.0")
    lines.append("")
    lines.append("### Instanton Model B")
    lines.append("")
    lines.append("Quadratic energy dependence:")
    lines.append("")
    lines.append("```")
    lines.append("logP = -(alpha * dE^2 * R / (c * E_scale)) / hbar")
    lines.append("```")
    lines.append("")
    lines.append("Default: alpha = 1.0, E_scale = 1e-9 J")
    lines.append("")
    lines.append("### Figure of Merit (F)")
    lines.append("")
    lines.append("A toy metric combining computational capacity and formation probability:")
    lines.append("")
    lines.append("```")
    lines.append("F = log10(ops) + log10(P)")
    lines.append("```")
    lines.append("")
    lines.append("Higher F indicates configurations that balance high compute with non-negligible formation probability.")
    lines.append("")
    return lines


def _section_persistence_taxonomy() -> List[str]:
    """Generate persistence taxonomy section."""
    lines = []
    lines.append("## Persistence Taxonomy")
    lines.append("")
    lines.append("Bubbles are classified along two independent dimensions based on their activity decay behavior.")
    lines.append("")
    lines.append("### Activity Classes")
    lines.append("")
    lines.append("Based on how normalized activity f(t) = E(t)/E0 evolves:")
    lines.append("")
    lines.append("| Class | Criterion | Interpretation |")
    lines.append("| --- | --- | --- |")
    lines.append("| ContinuousActive | mean(f) > 0.5, final f > 0.3 | Sustained high activity throughout lifetime |")
    lines.append("| IntermittentActive | mean(f) > 0.1 | Moderate activity with significant decay |")
    lines.append("| InstantaneousTerminal | mean(f) <= 0.1 | Rapid collapse to low activity |")
    lines.append("")
    lines.append("### Persistence Classes")
    lines.append("")
    lines.append("Based on the decay slope of window-integrated capacity C_win(T) = integral of f(t) over [T, 2T]:")
    lines.append("")
    lines.append("| Class | Slope Criterion | Interpretation |")
    lines.append("| --- | --- | --- |")
    lines.append("| Persistent | slope >= -1 | Activity decays slowly; significant capacity at late times |")
    lines.append("| LongTailTerminal | -2 < slope < -1 | Power-law decay; diminishing but non-negligible late activity |")
    lines.append("| Terminal | slope <= -2 | Fast exponential decay; activity effectively ends |")
    lines.append("")
    lines.append("### Combined Classification")
    lines.append("")
    lines.append("The two dimensions are largely independent, yielding a 3x3 classification space:")
    lines.append("")
    lines.append("```")
    lines.append("                    Persistent    LongTailTerminal    Terminal")
    lines.append("ContinuousActive       [1,1]           [1,2]            [1,3]")
    lines.append("IntermittentActive     [2,1]           [2,2]            [2,3]")
    lines.append("InstantaneousTerminal  [3,1]           [3,2]            [3,3]")
    lines.append("```")
    lines.append("")
    lines.append("Most physically interesting configurations fall in the upper-left (sustained activity with persistence)")
    lines.append("or along the diagonal (activity and persistence decay together).")
    lines.append("")
    return lines


def _section_results_tables(
    dynamics_data: Optional[Dict],
    sweep_data: Optional[Dict],
    bubble_data: Optional[Dict],
) -> List[str]:
    """Generate results tables from JSON artifacts."""
    lines = []
    lines.append("## Results Summary")
    lines.append("")

    # Dynamics results
    if dynamics_data:
        lines.append("### Dynamics Simulation")
        lines.append("")
        params = dynamics_data.get("simulation", {}).get("params", {})
        classification = dynamics_data.get("metrics", {}).get("classification", {})
        summary = dynamics_data.get("metrics", {}).get("summary", {})

        lines.append("**Parameters:**")
        lines.append("")
        lines.append(f"- E0 = {params.get('E0', 'N/A'):.3e} J")
        lines.append(f"- R0 = {params.get('R0', 'N/A'):.3e} m")
        lines.append(f"- tau (from leak_rate) = {1/params.get('leak_rate', 1):.3e} s")
        lines.append("")

        lines.append("**Classification:**")
        lines.append("")
        lines.append(f"- Activity: {classification.get('activity_class', 'N/A')}")
        lines.append(f"- Persistence: {classification.get('persistence_class', 'N/A')}")
        lines.append(f"- Slope: {classification.get('slope', 'N/A'):.3f}")
        lines.append("")

        lines.append("**Summary Statistics:**")
        lines.append("")
        lines.append(f"- Mean activity: {summary.get('mean_f', 0):.4f}")
        lines.append(f"- Final activity: {summary.get('final_f', 0):.2e}")
        lines.append(f"- Total integrated activity: {summary.get('total_activity', 0):.3e}")
        lines.append("")

    # Sweep results
    if sweep_data:
        lines.append("### Comprehensive Sweep")
        lines.append("")
        meta = sweep_data.get("metadata", {})
        config = meta.get("config", {})
        results = sweep_data.get("results", [])

        lines.append(f"**Grid:** {config.get('nR', 0)} x {config.get('nE', 0)} x {config.get('nTau', 0)} = {meta.get('total_points', 0)} points")
        lines.append("")

        # Top 5 by F_inst_a
        valid = [r for r in results if math.isfinite(r.get("F_inst_a", float("-inf")))]
        if valid:
            top_F = sorted(valid, key=lambda x: x["F_inst_a"], reverse=True)[:5]
            lines.append("**Top 5 by F (instanton_a):**")
            lines.append("")
            lines.append("| R0 (m) | dE (J) | tau (s) | F | Persistence |")
            lines.append("| --- | --- | --- | --- | --- |")
            for r in top_F:
                lines.append(f"| {r['R0_m']:.2e} | {r['dE_J']:.2e} | {r['tau_s']:.2e} | {r['F_inst_a']:.1f} | {r['persistence_class']} |")
            lines.append("")

        # Persistence distribution
        from collections import Counter
        persistence_counts = Counter(r.get("persistence_class") for r in results)
        total = len(results)

        lines.append("**Persistence Distribution:**")
        lines.append("")
        lines.append("| Class | Count | Fraction |")
        lines.append("| --- | --- | --- |")
        for cls in ["Persistent", "LongTailTerminal", "Terminal"]:
            count = persistence_counts.get(cls, 0)
            frac = count / total if total > 0 else 0
            lines.append(f"| {cls} | {count} | {frac:.1%} |")
        lines.append("")

    # Bubble demo results (if no sweep)
    if bubble_data and not sweep_data:
        lines.append("### Bubble Parameter Sweep")
        lines.append("")
        results = bubble_data.get("results", [])
        valid = [r for r in results if math.isfinite(r.get("F", float("-inf")))]
        if valid:
            top_F = sorted(valid, key=lambda x: x["F"], reverse=True)[:5]
            lines.append("**Top 5 by F:**")
            lines.append("")
            lines.append("| R (m) | dE (J) | tau (s) | log10_ops | F |")
            lines.append("| --- | --- | --- | --- | --- |")
            for r in top_F:
                lines.append(f"| {r['R_m']:.2e} | {r['dE_J']:.2e} | {r['tau_s']:.2e} | {r['log10_ops']:.1f} | {r['F']:.1f} |")
            lines.append("")

    if not dynamics_data and not sweep_data and not bubble_data:
        lines.append("*No simulation results found. Run --bubble-dynamics, --bubble-demo, or --sweep-all first.*")
        lines.append("")

    return lines


def _section_interpretation_guide() -> List[str]:
    """Generate interpretation guide section."""
    lines = []
    lines.append("## How to Interpret Results")
    lines.append("")
    lines.append("### Reading the Metrics")
    lines.append("")
    lines.append("1. **log10_bits / log10_ops**: These represent the theoretical maximum information storage and computation.")
    lines.append("   Values above ~80 indicate cosmologically interesting scales; values below ~20 are subatomic.")
    lines.append("")
    lines.append("2. **logP / log10P**: Formation probability in natural/base-10 logarithm.")
    lines.append("   Values near 0 are common; values below -100 are astronomically rare.")
    lines.append("")
    lines.append("3. **F (figure of merit)**: Balances capacity against rarity.")
    lines.append("   - F > 0: Formation probability exceeds 10^(-ops), potentially observable")
    lines.append("   - F < -100: Effectively impossible to form with significant compute")
    lines.append("")
    lines.append("4. **Persistence class**: Indicates whether late-time activity matters.")
    lines.append("   - Persistent: Relevant for long-duration processes")
    lines.append("   - Terminal: Only early-time behavior matters")
    lines.append("")
    lines.append("### Typical Patterns")
    lines.append("")
    lines.append("- **High R, High E**: Maximum complexity ceiling, but rarity dominates (very negative logP)")
    lines.append("- **Low R, Low E**: High formation probability, but negligible compute capacity")
    lines.append("- **Optimal F region**: Intermediate scales where capacity and probability balance")
    lines.append("- **Persistence independence**: Classification is largely orthogonal to rarity/complexity")
    lines.append("")
    lines.append("### Using the Plots")
    lines.append("")
    lines.append("- **Heatmaps**: Identify parameter regions of interest; look for gradients and boundaries")
    lines.append("- **Scatter plots**: Understand correlations (or lack thereof) between metrics")
    lines.append("- **Pareto frontier**: Find optimal trade-offs between competing objectives")
    lines.append("")
    return lines


def _section_limitations() -> List[str]:
    """Generate limitations section."""
    lines = []
    lines.append("## Limitations")
    lines.append("")
    lines.append("This simulator is a toy model for exploratory and educational purposes. Key limitations:")
    lines.append("")
    lines.append("1. **Not physically predictive**: Equations are illustrative, not derived from quantum gravity or field theory")
    lines.append("")
    lines.append("2. **Simplified dynamics**: Exponential decay and asymptotic growth are placeholders for complex physics")
    lines.append("")
    lines.append("3. **Rarity models are speculative**: Thermal and instanton forms capture qualitative behavior only")
    lines.append("")
    lines.append("4. **No uncertainty quantification**: Results are point estimates without error bars")
    lines.append("")
    lines.append("5. **Grid resolution**: Coarse grids may miss important parameter regimes")
    lines.append("")
    lines.append("6. **Independence assumptions**: Real bubble formation likely couples parameters in complex ways")
    lines.append("")
    lines.append("7. **No environmental effects**: Interactions with surrounding spacetime are ignored")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by emergence-simulator*")
    lines.append("")
    return lines
