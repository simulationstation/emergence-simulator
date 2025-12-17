"""Plotting utilities for bubble sweeps."""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = None
    plt = None


PLOT_FILENAMES = {
    "complexity_bits_vs_rarity": "complexity_bits_vs_rarity.png",
    "ops_vs_rarity": "ops_vs_rarity.png",
    "bits_vs_ops": "bits_vs_ops.png",
    "phase_heatmap": "phase_heatmap_log10_ops_minus_log10_rarity.png",
    "activity": "bubble_activity_f_vs_t.png",
    "cwin": "bubble_Cwin_vs_T.png",
}


def _ensure_matplotlib():
    if matplotlib is None or plt is None:
        raise RuntimeError("matplotlib is required for plotting")


def _extract_arrays(results: List[Dict]):
    log10_ops = [r["log10_ops"] for r in results]
    log10_bits = [r["log10_bits"] for r in results]
    log10P = [r["rarity"].get("log10P", float("-inf")) for r in results]
    R = [r["R"] for r in results]
    E = [r["E"] for r in results]
    tau = [r["tau"] for r in results]
    return log10_ops, log10_bits, log10P, R, E, tau


def plot_complexity_vs_rarity(results: List[Dict], outdir: str) -> str:
    _ensure_matplotlib()
    log10_ops, log10_bits, log10P, _, _, _ = _extract_arrays(results)
    fig, ax = plt.subplots()
    ax.scatter(log10P, log10_ops, label="ops ceiling", alpha=0.8)
    ax.scatter(log10P, log10_bits * math.log10(2.0), label="bits ceiling", marker="x", alpha=0.7)
    ax.set_xlabel(r"$\log_{10} P$")
    ax.set_ylabel(r"$\log_{10}$ complexity ceilings")
    ax.legend()
    path = os.path.join(outdir, PLOT_FILENAMES["complexity_bits_vs_rarity"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_ops_vs_rarity(results: List[Dict], outdir: str) -> str:
    _ensure_matplotlib()
    log10_ops, _, log10P, _, _, _ = _extract_arrays(results)
    fig, ax = plt.subplots()
    ax.scatter(log10P, log10_ops, alpha=0.8)
    ax.set_xlabel(r"$\log_{10} P$")
    ax.set_ylabel(r"$\log_{10}$ ops ceiling")
    path = os.path.join(outdir, PLOT_FILENAMES["ops_vs_rarity"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_bits_vs_ops(results: List[Dict], outdir: str) -> str:
    _ensure_matplotlib()
    log10_ops, log10_bits, _, _, _, _ = _extract_arrays(results)
    fig, ax = plt.subplots()
    ax.scatter(log10_bits, log10_ops, alpha=0.8)
    ax.set_xlabel(r"$\log_{10}$ bits")
    ax.set_ylabel(r"$\log_{10}$ ops")
    path = os.path.join(outdir, PLOT_FILENAMES["bits_vs_ops"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_phase_heatmap(results: List[Dict], outdir: str) -> str:
    _ensure_matplotlib()
    log10_ops, _, log10P, R, E, tau = _extract_arrays(results)
    unique_tau = sorted(set(tau))
    if len(unique_tau) == 0:
        return ""
    # pick first tau slice for 2D grid
    tau0 = unique_tau[0]
    # mask indices where tau==tau0
    mask_indices = [i for i, t in enumerate(tau) if math.isclose(t, tau0)]
    if not mask_indices:
        return ""
    Rs = sorted(set(R[i] for i in mask_indices))
    Es = sorted(set(E[i] for i in mask_indices))
    grid = [[math.nan for _ in Es] for _ in Rs]
    for i, r in enumerate(Rs):
        for j, e in enumerate(Es):
            candidates = [
                k
                for k in mask_indices
                if math.isclose(R[k], r) and math.isclose(E[k], e)
            ]
            if not candidates:
                continue
            idx = candidates[0]
            value = log10_ops[idx] - log10P[idx]
            if not math.isfinite(value):
                value = math.nan
            grid[i][j] = value

    fig, ax = plt.subplots()
    im = ax.imshow(grid, origin="lower", aspect="auto", extent=[min(Es), max(Es), min(Rs), max(Rs)])
    ax.set_xlabel("ΔE (J)")
    ax.set_ylabel("R (m)")
    ax.set_title("log10 ops - log10 rarity (tau slice)")
    fig.colorbar(im, ax=ax, label="score")
    path = os.path.join(outdir, PLOT_FILENAMES["phase_heatmap"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_activity(times: Iterable[float], f_values: Iterable[float], outdir: str) -> str:
    _ensure_matplotlib()
    fig, ax = plt.subplots()
    ax.plot(list(times), list(f_values))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("activity proxy")
    ax.set_title("bubble activity vs time")
    path = os.path.join(outdir, PLOT_FILENAMES["activity"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_cwin(Ts: Iterable[float], C_values: Iterable[float], outdir: str) -> str:
    _ensure_matplotlib()
    fig, ax = plt.subplots()
    ax.plot(list(Ts), list(C_values))
    ax.set_xlabel("window (s)")
    ax.set_ylabel("capacity proxy")
    ax.set_title("window capacity vs timescale")
    path = os.path.join(outdir, PLOT_FILENAMES["cwin"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
