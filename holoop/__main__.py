"""Command-line entry point for holoop."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List
import statistics

from .bubbles import dynamics, experiments, plot, report

try:
    from .ops import terminality
except Exception:  # pragma: no cover - optional dependency
    terminality = None


DEFAULT_OUTDIR = "outputs_holoop"


def _placeholder_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("plot unavailable (matplotlib not installed)\n")


def run_bubble_demo(args: argparse.Namespace) -> Dict:
    rarity_model = args.rarity_model
    params = {
        "rarity_model": rarity_model,
        "R_min": args.R_min,
        "R_max": args.R_max,
        "nR": args.nR,
        "E_min": args.E_min,
        "E_max": args.E_max,
        "nE": args.nE,
        "tau_min": args.tau_min,
        "tau_max": args.tau_max,
        "nTau": args.nTau,
        "T_env": args.T_env,
        "alpha": args.alpha,
        "E_scale": args.E_scale,
        "f_end": args.f_end,
        "seed": args.seed,
    }

    results = experiments.run_bubble_sweep(**params)

    bubble_dir = Path(args.outdir) / "bubbles"
    bubble_dir.mkdir(parents=True, exist_ok=True)
    results_path = bubble_dir / "bubble_results.json"
    experiments.save_results(results, str(results_path))

    # plots
    plots_generated: List[str] = []
    for plotting_func, filename in [
        (plot.plot_complexity_vs_rarity, plot.PLOT_FILENAMES["complexity_bits_vs_rarity"]),
        (plot.plot_ops_vs_rarity, plot.PLOT_FILENAMES["ops_vs_rarity"]),
        (plot.plot_bits_vs_ops, plot.PLOT_FILENAMES["bits_vs_ops"]),
        (plot.plot_phase_heatmap, plot.PLOT_FILENAMES["phase_heatmap"]),
    ]:
        try:
            out_path = plotting_func(results["results"], str(bubble_dir))
        except Exception:
            out_path = bubble_dir / filename
            _placeholder_plot(out_path)
        plots_generated.append(str(out_path))

    terminality_info: Dict | None = None
    if terminality is not None:
        # Build a representative activity trace using median parameters
        tau_values = [r["tau"] for r in results["results"]]
        E_values = [r["E"] for r in results["results"]]
        tau_sample = float(statistics.median(tau_values)) if tau_values else 1.0
        E_sample = float(statistics.median(E_values)) if E_values else 1.0
        lam_sample = math.log(1.0 / args.f_end) / tau_sample
        if args.nsteps <= 1:
            times = [args.tmin]
        else:
            times = [
                10
                ** (
                    math.log10(args.tmin)
                    + (math.log10(args.tmax) - math.log10(args.tmin)) * i / (args.nsteps - 1)
                )
                for i in range(args.nsteps)
            ]
        activity = dynamics.activity_series(E_sample, lam_sample, times, mode="ops")
        capacities = terminality.window_capacity(times, activity)
        classification = terminality.classify_activity(capacities)
        terminality_info = {
            "classification": classification,
            "activity_times": list(times),
            "activity": activity,
            "capacities": capacities,
        }
        try:
            plots_generated.append(plot.plot_activity(times, activity, str(bubble_dir)))
        except Exception:
            out_path = bubble_dir / plot.PLOT_FILENAMES["activity"]
            _placeholder_plot(out_path)
            plots_generated.append(str(out_path))
        try:
            plots_generated.append(
                plot.plot_cwin([t for t, _ in capacities], [c for _, c in capacities], str(bubble_dir))
            )
        except Exception:
            out_path = bubble_dir / plot.PLOT_FILENAMES["cwin"]
            _placeholder_plot(out_path)
            plots_generated.append(str(out_path))

    report_path = bubble_dir / "bubble_report.md"
    report.write_report(results, str(report_path), terminality_info=terminality_info)

    main_report_path = Path(args.outdir) / "report.md"
    report.append_main_report(str(main_report_path), results, str(results_path))

    return {
        "results_path": str(results_path),
        "report_path": str(report_path),
        "plots": plots_generated,
    }


def run_suite(args: argparse.Namespace) -> int:
    results = {}

    # Placeholder for other suite components
    try:
        bubble_output = run_bubble_demo(args)
        results["bubble_demo"] = {"status": "ok", **bubble_output}
    except Exception as exc:  # pragma: no cover - robustness
        results["bubble_demo"] = {"status": "failed", "error": str(exc)}

    results_path = Path(args.outdir) / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="holoop CLI")
    parser.add_argument("--bubble_demo", action="store_true", help="run bubble parameter sweep")
    parser.add_argument("--suite", action="store_true", help="run full suite including bubble demo")
    parser.add_argument("--fast", action="store_true", help="use reduced sweep sizes")

    parser.add_argument("--R_min", type=float, default=1e-6)
    parser.add_argument("--R_max", type=float, default=1e3)
    parser.add_argument("--nR", type=int, default=16)
    parser.add_argument("--E_min", type=float, default=1e-20)
    parser.add_argument("--E_max", type=float, default=1e10)
    parser.add_argument("--nE", type=int, default=16)
    parser.add_argument("--tau_min", type=float, default=1e-9)
    parser.add_argument("--tau_max", type=float, default=1e9)
    parser.add_argument("--nTau", type=int, default=4)
    parser.add_argument("--rarity_model", type=str, default="thermal", choices=["thermal", "instanton_a", "instanton_b"])
    parser.add_argument("--T_env", type=float, default=300.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--E_scale", type=float, default=1e-9)
    parser.add_argument("--f_end", type=float, default=1e-3)
    parser.add_argument("--tmin", type=float, default=1e-6)
    parser.add_argument("--tmax", type=float, default=1e3)
    parser.add_argument("--nsteps", type=int, default=64)
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fast:
        args.nR = min(args.nR, 8)
        args.nE = min(args.nE, 8)
        args.nTau = 1
    if args.bubble_demo:
        run_bubble_demo(args)
        return 0
    if args.suite:
        return run_suite(args)
    # default: print guidance
    print("No action requested. Use --bubble_demo or --suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
