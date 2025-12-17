import os
import math
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_plots(results: Dict[str, Any], outdir: str):
    """Generate all plots from sweep results."""
    data = results["results"]
    metadata = results["metadata"]

    _plot_ops_vs_logP(data, outdir)
    _plot_bits_vs_ops(data, outdir)
    _plot_F_heatmap(data, metadata, outdir)


def _plot_ops_vs_logP(data: List[Dict], outdir: str):
    """Scatter plot of log10_ops vs log10P."""
    log10_ops = []
    log10P = []

    for d in data:
        if math.isfinite(d["log10_ops"]) and math.isfinite(d["log10P"]):
            log10_ops.append(d["log10_ops"])
            log10P.append(d["log10P"])

    plt.figure()
    plt.scatter(log10_ops, log10P, alpha=0.5, s=10)
    plt.xlabel("log10(ops)")
    plt.ylabel("log10(P)")
    plt.title("Operations vs Rarity")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ops_vs_logP.png"), dpi=100)
    plt.close()


def _plot_bits_vs_ops(data: List[Dict], outdir: str):
    """Scatter plot of bits vs ops."""
    log10_bits = []
    log10_ops = []

    for d in data:
        if math.isfinite(d["log10_bits"]) and math.isfinite(d["log10_ops"]):
            log10_bits.append(d["log10_bits"])
            log10_ops.append(d["log10_ops"])

    plt.figure()
    plt.scatter(log10_bits, log10_ops, alpha=0.5, s=10)
    plt.xlabel("log10(bits)")
    plt.ylabel("log10(ops)")
    plt.title("Bekenstein Bits vs Lloyd Ops")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bits_vs_ops.png"), dpi=100)
    plt.close()


def _plot_F_heatmap(data: List[Dict], metadata: Dict, outdir: str):
    """Heatmap of F for fixed tau (first tau value), one per rarity model."""
    R_vals = metadata["R_vals"]
    dE_vals = metadata["dE_vals"]
    tau_vals = metadata["tau_vals"]

    # Use first tau value
    fixed_tau = tau_vals[0]

    for model in ["thermal", "instanton_a"]:
        # Build 2D grid
        F_grid = np.full((len(dE_vals), len(R_vals)), np.nan)

        for d in data:
            if d["rarity_model"] != model:
                continue
            if not np.isclose(d["tau_s"], fixed_tau, rtol=1e-9):
                continue

            # Find indices
            try:
                i_R = R_vals.index(d["R_m"])
            except ValueError:
                i_R = np.argmin(np.abs(np.array(R_vals) - d["R_m"]))
            try:
                i_dE = dE_vals.index(d["dE_J"])
            except ValueError:
                i_dE = np.argmin(np.abs(np.array(dE_vals) - d["dE_J"]))

            F_val = d["F"]
            if math.isfinite(F_val):
                F_grid[i_dE, i_R] = F_val

        # Plot
        plt.figure()
        log_R = np.log10(R_vals)
        log_dE = np.log10(dE_vals)

        plt.imshow(
            F_grid,
            aspect="auto",
            origin="lower",
            extent=[log_R[0], log_R[-1], log_dE[0], log_dE[-1]],
        )
        plt.colorbar(label="F = log10(ops) + log10(P)")
        plt.xlabel("log10(R_m)")
        plt.ylabel("log10(dE_J)")
        plt.title(f"F heatmap ({model}, tau={fixed_tau:.1e}s)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"F_heatmap_{model}.png"), dpi=100)
        plt.close()
