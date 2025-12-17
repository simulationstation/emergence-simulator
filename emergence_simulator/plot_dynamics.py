import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_dynamics_plots(sim_result: dict, metrics: dict, outdir: str):
    """Generate dynamics-related plots."""
    _plot_activity_over_time(sim_result, outdir)
    _plot_window_capacity(metrics, outdir)


def _plot_activity_over_time(sim_result: dict, outdir: str):
    """Plot normalized activity f(t) and radius R(t) over time."""
    ts = sim_result["ts"]
    f_t = sim_result["f_t"]
    R_t = sim_result["R_t"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Activity plot
    ax1.plot(ts, f_t)
    ax1.set_ylabel("Normalized Activity f(t) = E(t)/E0")
    ax1.set_title("Bubble Dynamics: Expansion then Evaporation")
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)

    # Radius plot
    ax2.plot(ts, R_t)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Radius R(t) (m)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "activity_over_time.png"), dpi=100)
    plt.close()


def _plot_window_capacity(metrics: dict, outdir: str):
    """Plot window capacity C_win(T) vs T."""
    T_values = np.array(metrics["T_values"])
    C_win = np.array(metrics["C_win"])

    # Filter positive values for log plot
    valid = C_win > 0
    if np.sum(valid) < 2:
        # Not enough data for plot
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(T_values[valid], C_win[valid], "o-")
    ax.set_xlabel("Window Start T (s)")
    ax.set_ylabel("Window Capacity C_win = ∫[T,2T] f(t) dt")
    ax.set_title(f"Window-Integrated Activity (slope={metrics['classification']['slope']:.2f})")
    ax.grid(True, alpha=0.3, which="both")

    # Add classification annotation
    classification = metrics["classification"]
    text = f"{classification['persistence_class']}\n{classification['activity_class']}"
    ax.annotate(
        text,
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "window_capacity.png"), dpi=100)
    plt.close()
