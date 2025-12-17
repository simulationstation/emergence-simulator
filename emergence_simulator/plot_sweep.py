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

    feed_mode = metadata["config"].get("feed_mode", "constant")

    _plot_persistence_heatmap(data, metadata, outdir)
    _plot_ops_vs_rarity_by_persistence(data, metadata, outdir)
    _plot_F_inst_a_heatmap(data, metadata, outdir)
    _plot_pareto_frontier(data, metadata, outdir)

    if feed_mode == "decay":
        _plot_fraction_by_class_vs_q(metadata, outdir)
        _plot_fraction_by_class_vs_eta0(metadata, outdir)
    else:
        _plot_fraction_terminal_vs_eta(metadata, outdir)


def _plot_persistence_heatmap(data: List[Dict], metadata: Dict, outdir: str):
    """Heatmap showing persistence class across (R0, dE) for fixed tau slice."""
    grids = metadata["grids"]
    config = metadata["config"]
    feed_mode = config.get("feed_mode", "constant")
    R0_vals = grids["R0_vals"]
    dE_vals = grids["dE_vals"]
    tau_vals = grids["tau_vals"]

    # Use middle tau value
    fixed_tau = tau_vals[len(tau_vals) // 2]

    # Filter function depends on feed mode
    if feed_mode == "decay":
        q_vals = grids.get("q_vals", [1.0])
        eta0_vals = grids.get("eta0_vals", [1e-4])
        t0_fracs = grids.get("t0_fracs", [0.1])
        # Use middle q, first t0_frac, first eta0
        fixed_q = q_vals[len(q_vals) // 2]
        fixed_eta0 = eta0_vals[0]
        fixed_t0_frac = t0_fracs[0]
        title_suffix = f"tau={fixed_tau:.1e}s, q={fixed_q}, eta0={fixed_eta0:.0e}"

        def matches_filter(d):
            return (
                np.isclose(d["tau_s"], fixed_tau, rtol=1e-9) and
                d.get("q", 1.0) == fixed_q and
                d.get("eta0", 1e-4) == fixed_eta0 and
                d.get("t0_frac", 0.1) == fixed_t0_frac
            )
    else:
        eta_vals = grids.get("eta_vals", [0.0])
        fixed_eta = max(eta_vals)
        title_suffix = f"tau={fixed_tau:.1e}s, eta={fixed_eta:.1e}"

        def matches_filter(d):
            return (
                np.isclose(d["tau_s"], fixed_tau, rtol=1e-9) and
                d.get("eta", 0.0) == fixed_eta
            )

    # Build 2D grid
    grid = np.full((len(dE_vals), len(R0_vals)), np.nan)

    for d in data:
        if not matches_filter(d):
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
    ax.set_title(f"Persistence Classification ({title_suffix})")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "persistence_heatmap.png"), dpi=100)
    plt.close()


def _plot_ops_vs_rarity_by_persistence(data: List[Dict], metadata: Dict, outdir: str):
    """Scatter plot of log10_ops vs log10P colored by persistence class."""
    markers = {"Persistent": "o", "LongTailTerminal": "s", "Terminal": "^"}
    config = metadata["config"]
    feed_mode = config.get("feed_mode", "constant")
    grids = metadata["grids"]

    # Filter based on feed mode
    if feed_mode == "decay":
        q_vals = grids.get("q_vals", [1.0])
        eta0_vals = grids.get("eta0_vals", [1e-4])
        fixed_q = q_vals[len(q_vals) // 2]
        fixed_eta0 = eta0_vals[0]
        filtered_data = [d for d in data if d.get("q") == fixed_q and d.get("eta0") == fixed_eta0]
        title_suffix = f"q={fixed_q}, eta0={fixed_eta0:.0e}"
    else:
        eta_vals = grids.get("eta_vals", [0.0])
        fixed_eta = max(eta_vals)
        filtered_data = [d for d in data if d.get("eta", 0.0) == fixed_eta]
        title_suffix = f"eta={fixed_eta:.1e}"

    fig, ax = plt.subplots(figsize=(8, 6))

    for cls in PERSISTENCE_CLASSES:
        subset = [d for d in filtered_data if d["persistence_class"] == cls]
        if not subset:
            continue

        log10_ops = [d["log10_ops"] for d in subset if math.isfinite(d["log10_ops"]) and math.isfinite(d["log10P_inst_a"])]
        log10P = [d["log10P_inst_a"] for d in subset if math.isfinite(d["log10_ops"]) and math.isfinite(d["log10P_inst_a"])]

        if log10_ops:
            ax.scatter(log10_ops, log10P, marker=markers[cls], alpha=0.6, s=20, label=cls)

    ax.set_xlabel("log10(ops)")
    ax.set_ylabel("log10(P) [instanton_a]")
    ax.set_title(f"Operations vs Rarity by Persistence Class ({title_suffix})")
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
    eta_vals = grids.get("eta_vals", [0.0])

    # Use middle tau value (F is independent of eta)
    fixed_tau = tau_vals[len(tau_vals) // 2]
    fixed_eta = eta_vals[0]  # F doesn't depend on eta, use first

    # Build 2D grid
    grid = np.full((len(dE_vals), len(R0_vals)), np.nan)

    for d in data:
        if not np.isclose(d["tau_s"], fixed_tau, rtol=1e-9):
            continue
        if d.get("eta", 0.0) != fixed_eta:
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


def _plot_pareto_frontier(data: List[Dict], metadata: Dict, outdir: str):
    """Plot points maximizing ops at fixed rarity bins (Pareto frontier)."""
    config = metadata["config"]
    feed_mode = config.get("feed_mode", "constant")
    grids = metadata["grids"]

    # Filter based on feed mode
    if feed_mode == "decay":
        q_vals = grids.get("q_vals", [1.0])
        eta0_vals = grids.get("eta0_vals", [1e-4])
        fixed_q = q_vals[len(q_vals) // 2]
        fixed_eta0 = eta0_vals[0]
        filtered_data = [d for d in data if d.get("q") == fixed_q and d.get("eta0") == fixed_eta0]
        title_suffix = f"q={fixed_q}, eta0={fixed_eta0:.0e}"
    else:
        eta_vals = grids.get("eta_vals", [0.0])
        fixed_eta = max(eta_vals)
        filtered_data = [d for d in data if d.get("eta", 0.0) == fixed_eta]
        title_suffix = f"eta={fixed_eta:.1e}"

    # Filter valid points
    valid = [d for d in filtered_data if math.isfinite(d["log10_ops"]) and math.isfinite(d["log10P_inst_a"])]

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
    ax.set_title(f"Pareto Frontier: Max Ops at Fixed Rarity ({title_suffix})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pareto_frontier.png"), dpi=100)
    plt.close()


def _plot_fraction_terminal_vs_eta(metadata: Dict, outdir: str):
    """Plot fraction of Terminal classifications vs eta."""
    eta_summary = metadata.get("eta_summary", {})

    if not eta_summary:
        return

    eta_vals = []
    frac_terminal = []
    frac_persistent = []
    frac_longtail = []

    for eta_str, counts in sorted(eta_summary.items(), key=lambda x: float(x[0])):
        eta = float(eta_str)
        total = counts["total"]
        if total == 0:
            continue

        eta_vals.append(eta)
        frac_terminal.append(counts["Terminal"] / total)
        frac_persistent.append(counts["Persistent"] / total)
        frac_longtail.append(counts["LongTailTerminal"] / total)

    if not eta_vals:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use log scale for x-axis if eta > 0 exists
    x_vals = np.arange(len(eta_vals))
    x_labels = [f"{e:.0e}" if e > 0 else "0" for e in eta_vals]

    width = 0.25
    ax.bar(x_vals - width, frac_persistent, width, label="Persistent")
    ax.bar(x_vals, frac_longtail, width, label="LongTailTerminal")
    ax.bar(x_vals + width, frac_terminal, width, label="Terminal")

    ax.set_xlabel("eta (background feed coupling)")
    ax.set_ylabel("Fraction of grid points")
    ax.set_title("Persistence Classification vs Background Feed (eta)")
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fraction_terminal_vs_eta.png"), dpi=100)
    plt.close()


def _plot_fraction_by_class_vs_q(metadata: Dict, outdir: str):
    """Plot fraction of each persistence class vs q (power-law exponent)."""
    q_summary = metadata.get("q_summary", {})

    if not q_summary:
        return

    q_vals = []
    frac_terminal = []
    frac_persistent = []
    frac_longtail = []

    for q_str, counts in sorted(q_summary.items(), key=lambda x: float(x[0])):
        q = float(q_str)
        total = counts["total"]
        if total == 0:
            continue

        q_vals.append(q)
        frac_terminal.append(counts["Terminal"] / total)
        frac_persistent.append(counts["Persistent"] / total)
        frac_longtail.append(counts["LongTailTerminal"] / total)

    if not q_vals:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    x_vals = np.arange(len(q_vals))
    x_labels = [f"{q:.1f}" for q in q_vals]

    width = 0.25
    ax.bar(x_vals - width, frac_persistent, width, label="Persistent", color="tab:green")
    ax.bar(x_vals, frac_longtail, width, label="LongTailTerminal", color="tab:orange")
    ax.bar(x_vals + width, frac_terminal, width, label="Terminal", color="tab:red")

    ax.set_xlabel("q (power-law exponent)")
    ax.set_ylabel("Fraction of grid points")
    ax.set_title("Persistence Classification vs Feed Decay Exponent (q)")
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fraction_by_class_vs_q.png"), dpi=100)
    plt.close()


def _plot_fraction_by_class_vs_eta0(metadata: Dict, outdir: str):
    """Plot fraction of each persistence class vs eta0 (initial feed coupling)."""
    eta0_summary = metadata.get("eta0_summary", {})

    if not eta0_summary:
        return

    eta0_vals = []
    frac_terminal = []
    frac_persistent = []
    frac_longtail = []

    for eta0_str, counts in sorted(eta0_summary.items(), key=lambda x: float(x[0])):
        eta0 = float(eta0_str)
        total = counts["total"]
        if total == 0:
            continue

        eta0_vals.append(eta0)
        frac_terminal.append(counts["Terminal"] / total)
        frac_persistent.append(counts["Persistent"] / total)
        frac_longtail.append(counts["LongTailTerminal"] / total)

    if not eta0_vals:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    x_vals = np.arange(len(eta0_vals))
    x_labels = [f"{e:.0e}" for e in eta0_vals]

    width = 0.25
    ax.bar(x_vals - width, frac_persistent, width, label="Persistent", color="tab:green")
    ax.bar(x_vals, frac_longtail, width, label="LongTailTerminal", color="tab:orange")
    ax.bar(x_vals + width, frac_terminal, width, label="Terminal", color="tab:red")

    ax.set_xlabel("eta0 (initial feed coupling)")
    ax.set_ylabel("Fraction of grid points")
    ax.set_title("Persistence Classification vs Initial Feed Coupling (eta0)")
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fraction_by_class_vs_eta0.png"), dpi=100)
    plt.close()
