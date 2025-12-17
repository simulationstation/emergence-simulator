import argparse
import json
import os
from .sweeps import run_sweep, SweepConfig
from .plot import generate_plots
from .report import generate_report


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="emergence-simulator CLI")
    parser.add_argument(
        "--bubble-demo",
        action="store_true",
        help="Run bubble parameter sweep demo",
    )
    parser.add_argument(
        "--bubble-dynamics",
        action="store_true",
        help="Run bubble dynamics simulation",
    )
    parser.add_argument(
        "--sweep-all",
        action="store_true",
        help="Run comprehensive sweep with complexity, rarity, and persistence",
    )
    parser.add_argument(
        "--report-master",
        action="store_true",
        help="Generate consolidated master report from existing artifacts",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs_emergence",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced grid size for faster execution",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Background feed coupling constant (default 0.0)",
    )
    parser.add_argument(
        "--E_bg_J",
        type=float,
        default=0.0,
        help="Background energy scale in Joules (default 0.0)",
    )
    return parser.parse_args(args)


def run_bubble_demo(outdir: str, fast: bool = False):
    """Run the bubble sweep demo and generate all artifacts."""
    bubble_dir = os.path.join(outdir, "bubbles")
    os.makedirs(bubble_dir, exist_ok=True)

    config = SweepConfig(fast=fast)
    results = run_sweep(config)

    # Save JSON results
    json_path = os.path.join(bubble_dir, "bubble_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate plots
    generate_plots(results, bubble_dir)

    # Generate report
    report_path = os.path.join(bubble_dir, "bubble_report.md")
    generate_report(results, report_path)

    print(f"Bubble demo complete. Artifacts written to {bubble_dir}")
    return results


def run_bubble_dynamics(
    outdir: str,
    fast: bool = False,
    sweep_results: dict = None,
    eta: float = 0.0,
    E_bg_J: float = 0.0,
):
    """Run bubble dynamics simulation for a representative bubble."""
    from .dynamics import simulate_bubble_dynamics
    from .metrics import compute_dynamics_metrics
    from .plot_dynamics import generate_dynamics_plots
    from .report_dynamics import generate_dynamics_report
    import math

    dynamics_dir = os.path.join(outdir, "dynamics")
    os.makedirs(dynamics_dir, exist_ok=True)

    # Pick representative bubble params
    if sweep_results and sweep_results.get("results"):
        # Find best-F bubble
        valid = [r for r in sweep_results["results"] if math.isfinite(r["F"])]
        if valid:
            best = max(valid, key=lambda x: x["F"])
            E0 = best["dE_J"]
            R0 = best["R_m"]
            tau = best["tau_s"]
        else:
            E0, R0, tau = 1e-10, 1e-3, 1e3
    else:
        # Default representative values
        E0 = 1e-10  # 0.1 nJ
        R0 = 1e-3   # 1 mm
        tau = 1e3   # 1000 s

    # Derive leak rate from tau (e-folding time)
    leak_rate = 1.0 / tau

    # Simulation parameters
    Rmax = R0 * 10
    tgrow = tau / 5
    t_end = tau * 5
    n_points = 100 if fast else 500

    # Use E0 as E_bg if not specified
    E_bg = E_bg_J if E_bg_J > 0 else E0

    # Run simulation
    sim_result = simulate_bubble_dynamics(
        E0=E0,
        R0=R0,
        Rmax=Rmax,
        leak_rate=leak_rate,
        tgrow=tgrow,
        t_end=t_end,
        n_points=n_points,
        eta=eta,
        E_bg=E_bg,
    )

    # Compute metrics
    metrics = compute_dynamics_metrics(sim_result)

    # Combine results
    results = {
        "simulation": {
            "params": sim_result["params"],
            "ts": sim_result["ts"].tolist(),
            "E_t": sim_result["E_t"].tolist(),
            "R_t": sim_result["R_t"].tolist(),
            "f_t": sim_result["f_t"].tolist(),
            "ops_per_s": sim_result["ops_per_s"].tolist(),
        },
        "metrics": metrics,
    }

    # Save JSON
    json_path = os.path.join(dynamics_dir, "dynamics_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate plots
    generate_dynamics_plots(sim_result, metrics, dynamics_dir)

    # Generate report
    report_path = os.path.join(dynamics_dir, "dynamics_report.md")
    generate_dynamics_report(results, report_path)

    print(f"Bubble dynamics complete. Artifacts written to {dynamics_dir}")
    return results


def run_sweep_all_cmd(outdir: str, fast: bool = False):
    """Run comprehensive sweep with complexity, rarity, and persistence."""
    from .sweep_all import run_sweep_all, SweepAllConfig
    from .plot_sweep import generate_sweep_plots
    from .report_sweep import generate_sweep_report

    sweep_dir = os.path.join(outdir, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)

    config = SweepAllConfig(fast=fast)
    results = run_sweep_all(config)

    # Save JSON results
    json_path = os.path.join(sweep_dir, "sweep_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate plots
    generate_sweep_plots(results, sweep_dir)

    # Generate report
    report_path = os.path.join(sweep_dir, "sweep_report.md")
    generate_sweep_report(results, report_path)

    print(f"Comprehensive sweep complete. Artifacts written to {sweep_dir}")
    return results


def run_report_master_cmd(outdir: str):
    """Generate consolidated master report from existing artifacts."""
    from .report_master import generate_master_report

    report_path = os.path.join(outdir, "MASTER_REPORT.md")
    generate_master_report(outdir, report_path)

    print(f"Master report generated: {report_path}")
    return report_path


def main(args=None):
    parsed = parse_args(args)

    sweep_results = None

    if parsed.bubble_demo:
        sweep_results = run_bubble_demo(parsed.outdir, parsed.fast)

    if parsed.bubble_dynamics:
        run_bubble_dynamics(
            parsed.outdir,
            parsed.fast,
            sweep_results,
            eta=parsed.eta,
            E_bg_J=parsed.E_bg_J,
        )

    if parsed.sweep_all:
        run_sweep_all_cmd(parsed.outdir, parsed.fast)

    if parsed.report_master:
        run_report_master_cmd(parsed.outdir)
        return 0

    if parsed.bubble_demo or parsed.bubble_dynamics or parsed.sweep_all:
        return 0

    # Default behavior: print ok message
    print("emergence-simulator ok")
    return 0
