"""Plotting functions for comprehensive sweep results."""

import os
import math
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Persistence class encoding
PERSISTENCE_CLASSES = ["Persistent", "LongTailTerminal", "Terminal"]
PERSISTENCE_TO_INT = {cls: i for i, cls in enumerate(PERSISTENCE_CLASSES)}


def generate_sweep_plots(results: Dict[str, Any], outdir: str):
    """Generate all sweep-related plots."""
    data = results["results"]
    metadata = results["metadata"]

    _plot_persistence_heatmap(data, metadata, outdir)
    _plot_ops_vs_rarity_by_persistence(data, outdir)
    _plot_F_inst_a_heatmap(data, metadata, outdir)
    _plot_pareto_frontier(data, outdir)


def _plot_persistence_heatmap(data: List[Dict], metadata: Dict, outdir: str):
    """Heatmap showing persistence class across (R0, dE) for fixed tau slice."""
    grids = metadata["grids"]
    R0_vals = grids["R0_vals"]
    dE_vals = grids["dE_vals"]
    tau_vals = grids["tau_vals"]

    # Use middle tau value
    fixed_tau = tau_vals[len(tau_vals) // 2]

    # Build 2D grid
    grid = np.full((len(dE_vals), len(R0_vals)), np.nan)

    for d in data:
        if not np.isclose(d["tau_s"], fixed_tau, rtol=1e-9):
            continue

        i_R = np.argmin(np.abs(np.array(R0_vals) - d["R0_m"]))
        i_dE = np.argmin(np.abs(np.array(dE_vals) - d["dE_J"]))

        cls = d["persistence_class"]
        grid[i_dE, i_R] = PERSISTENCE_TO_INT.get(cls, -1)

    fig, ax = plt.subplots(figsize=(8, 6))

    log_R0 = np.log10(R0_vals)
    log_dE = np.log10(dE_vals)

    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        extent=[log_R0[0], log_R0[-1], log_dE[0], log_dE[-1]],
        cmap="viridis",
        vmin=0,
        vmax=2,
    )

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(PERSISTENCE_CLASSES)
    cbar.set_label("Persistence Class")

    ax.set_xlabel("log10(R0_m)")
    ax.set_ylabel("log10(dE_J)")
    ax.set_title(f"Persistence Classification (tau={fixed_tau:.1e}s)")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "persistence_heatmap.png"), dpi=100)
    plt.close()


def _plot_ops_vs_rarity_by_persistence(data: List[Dict], outdir: str):
    """Scatter plot of log10_ops vs log10P colored by persistence class."""
    markers = {"Persistent": "o", "LongTailTerminal": "s", "Terminal": "^"}

    fig, ax = plt.subplots(figsize=(8, 6))

    for cls in PERSISTENCE_CLASSES:
        subset = [d for d in data if d["persistence_class"] == cls]
        if not subset:
            continue

        log10_ops = [d["log10_ops"] for d in subset if math.isfinite(d["log10_ops"]) and math.isfinite(d["log10P_inst_a"])]
        log10P = [d["log10P_inst_a"] for d in subset if math.isfinite(d["log10_ops"]) and math.isfinite(d["log10P_inst_a"])]

        if log10_ops:
            ax.scatter(log10_ops, log10P, marker=markers[cls], alpha=0.6, s=20, label=cls)

    ax.set_xlabel("log10(ops)")
    ax.set_ylabel("log10(P) [instanton_a]")
    ax.set_title("Operations vs Rarity by Persistence Class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ops_vs_rarity_persistence.png"), dpi=100)
    plt.close()


def _plot_F_inst_a_heatmap(data: List[Dict], metadata: Dict, outdir: str):
    """Heatmap of F_inst_a across (R0, dE) for fixed tau."""
    grids = metadata["grids"]
    R0_vals = grids["R0_vals"]
    dE_vals = grids["dE_vals"]
    tau_vals = grids["tau_vals"]

    # Use middle tau value
    fixed_tau = tau_vals[len(tau_vals) // 2]

    # Build 2D grid
    grid = np.full((len(dE_vals), len(R0_vals)), np.nan)

    for d in data:
        if not np.isclose(d["tau_s"], fixed_tau, rtol=1e-9):
            continue

        i_R = np.argmin(np.abs(np.array(R0_vals) - d["R0_m"]))
        i_dE = np.argmin(np.abs(np.array(dE_vals) - d["dE_J"]))

        F_val = d["F_inst_a"]
        if math.isfinite(F_val):
            grid[i_dE, i_R] = F_val

    fig, ax = plt.subplots(figsize=(8, 6))

    log_R0 = np.log10(R0_vals)
    log_dE = np.log10(dE_vals)

    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        extent=[log_R0[0], log_R0[-1], log_dE[0], log_dE[-1]],
    )

    plt.colorbar(im, ax=ax, label="F_inst_a = log10(ops) + log10(P)")

    ax.set_xlabel("log10(R0_m)")
    ax.set_ylabel("log10(dE_J)")
    ax.set_title(f"F (instanton_a) Heatmap (tau={fixed_tau:.1e}s)")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "F_inst_a_heatmap.png"), dpi=100)
    plt.close()


def _plot_pareto_frontier(data: List[Dict], outdir: str):
    """Plot points maximizing ops at fixed rarity bins (Pareto frontier)."""
    # Filter valid points
    valid = [d for d in data if math.isfinite(d["log10_ops"]) and math.isfinite(d["log10P_inst_a"])]

    if len(valid) < 2:
        return

    # Bin by log10P
    log10P_vals = np.array([d["log10P_inst_a"] for d in valid])
    log10_ops_vals = np.array([d["log10_ops"] for d in valid])

    # Create bins
    p_min, p_max = log10P_vals.min(), log10P_vals.max()
    n_bins = 20
    bin_edges = np.linspace(p_min, p_max, n_bins + 1)

    # Find max ops in each bin
    pareto_P = []
    pareto_ops = []

    for i in range(n_bins):
        mask = (log10P_vals >= bin_edges[i]) & (log10P_vals < bin_edges[i + 1])
        if np.sum(mask) > 0:
            idx = np.argmax(log10_ops_vals[mask])
            pareto_P.append(log10P_vals[mask][idx])
            pareto_ops.append(log10_ops_vals[mask][idx])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all points
    ax.scatter(log10P_vals, log10_ops_vals, alpha=0.3, s=10, label="All points")

    # Plot Pareto frontier
    if pareto_P:
        # Sort by P for line plot
        sorted_idx = np.argsort(pareto_P)
        pareto_P_sorted = np.array(pareto_P)[sorted_idx]
        pareto_ops_sorted = np.array(pareto_ops)[sorted_idx]

        ax.plot(pareto_P_sorted, pareto_ops_sorted, "r-", linewidth=2, label="Pareto frontier")
        ax.scatter(pareto_P_sorted, pareto_ops_sorted, c="red", s=50, zorder=5)

    ax.set_xlabel("log10(P) [instanton_a]")
    ax.set_ylabel("log10(ops)")
    ax.set_title("Pareto Frontier: Max Ops at Fixed Rarity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pareto_frontier.png"), dpi=100)
    plt.close()
