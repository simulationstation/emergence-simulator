"""Comprehensive sweep over bubble parameters with complexity, rarity, and persistence."""

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

from .bubbles import (
    bekenstein_bits,
    lloyd_ops,
    thermal_logP,
    instanton_a_logP,
    instanton_b_logP,
)
from .dynamics import simulate_bubble_dynamics
from .metrics import classify_persistence


@dataclass
class SweepAllConfig:
    """Configuration for comprehensive sweep."""
    fast: bool = False
    seed: int = 42

    @property
    def nR(self) -> int:
        return 8 if self.fast else 20

    @property
    def nE(self) -> int:
        return 8 if self.fast else 20

    @property
    def nTau(self) -> int:
        return 2 if self.fast else 5

    @property
    def nsteps(self) -> int:
        return 1200 if self.fast else 4000

    # Grid ranges
    R0_range: tuple = (1e-6, 1e3)
    dE_range: tuple = (1e-20, 1e10)
    tau_range: tuple = (1e-9, 1e9)

    # Rarity model parameters
    T_env_K: float = 2.7
    alpha: float = 1.0
    E_scale_J: float = 1e-9

    # Dynamics parameters
    f_end: float = 1e-3  # Target final activity for leak rate derivation


def derive_leak_rate(tau: float, f_end: float = 1e-3) -> float:
    """Derive leak rate from tau such that f(tau) = f_end."""
    # f(t) = exp(-leak * t), so f(tau) = exp(-leak * tau) = f_end
    # leak = -ln(f_end) / tau
    return -math.log(f_end) / tau


def run_sweep_all(config: SweepAllConfig) -> Dict[str, Any]:
    """Run comprehensive sweep over all parameters."""
    np.random.seed(config.seed)

    # Generate grids
    R0_vals = np.logspace(
        math.log10(config.R0_range[0]),
        math.log10(config.R0_range[1]),
        config.nR,
    )
    dE_vals = np.logspace(
        math.log10(config.dE_range[0]),
        math.log10(config.dE_range[1]),
        config.nE,
    )
    tau_vals = np.logspace(
        math.log10(config.tau_range[0]),
        math.log10(config.tau_range[1]),
        config.nTau,
    )

    results_list: List[Dict[str, Any]] = []

    total_points = len(R0_vals) * len(dE_vals) * len(tau_vals)
    point_idx = 0

    for R0 in R0_vals:
        for dE in dE_vals:
            for tau in tau_vals:
                point_idx += 1

                # Complexity ceilings
                bits_max = bekenstein_bits(R0, dE)
                ops_max = lloyd_ops(dE, tau)

                log10_bits = math.log10(bits_max) if bits_max > 0 else float("-inf")
                log10_ops = math.log10(ops_max) if ops_max > 0 else float("-inf")

                # Rarity scores (natural log)
                logP_thermal = thermal_logP(dE, config.T_env_K)
                logP_inst_a = instanton_a_logP(R0, dE, config.alpha)
                logP_inst_b = instanton_b_logP(R0, dE, config.alpha, config.E_scale_J)

                # Convert to log10 for F computation
                ln_to_log10 = 1.0 / math.log(10)
                log10P_thermal = logP_thermal * ln_to_log10 if math.isfinite(logP_thermal) else float("-inf")
                log10P_inst_a = logP_inst_a * ln_to_log10 if math.isfinite(logP_inst_a) else float("-inf")
                log10P_inst_b = logP_inst_b * ln_to_log10 if math.isfinite(logP_inst_b) else float("-inf")

                # F = log10_ops + log10P (toy weighted compute)
                def compute_F(log10_ops: float, log10P: float) -> float:
                    if math.isfinite(log10_ops) and math.isfinite(log10P):
                        return log10_ops + log10P
                    return float("-inf")

                F_thermal = compute_F(log10_ops, log10P_thermal)
                F_inst_a = compute_F(log10_ops, log10P_inst_a)
                F_inst_b = compute_F(log10_ops, log10P_inst_b)

                # Run dynamics for persistence classification
                leak_rate = derive_leak_rate(tau, config.f_end)
                Rmax = 10 * R0
                tgrow = 0.1 * tau
                t_end = 10 * tau

                sim_result = simulate_bubble_dynamics(
                    E0=dE,
                    R0=R0,
                    Rmax=Rmax,
                    leak_rate=leak_rate,
                    tgrow=tgrow,
                    t_end=t_end,
                    n_points=config.nsteps,
                )

                classification = classify_persistence(sim_result["ts"], sim_result["f_t"])

                results_list.append({
                    "R0_m": float(R0),
                    "dE_J": float(dE),
                    "tau_s": float(tau),
                    "bits_max": float(bits_max),
                    "ops_max": float(ops_max),
                    "log10_bits": float(log10_bits),
                    "log10_ops": float(log10_ops),
                    "logP_thermal": float(logP_thermal),
                    "logP_inst_a": float(logP_inst_a),
                    "logP_inst_b": float(logP_inst_b),
                    "log10P_thermal": float(log10P_thermal),
                    "log10P_inst_a": float(log10P_inst_a),
                    "log10P_inst_b": float(log10P_inst_b),
                    "F_thermal": float(F_thermal),
                    "F_inst_a": float(F_inst_a),
                    "F_inst_b": float(F_inst_b),
                    "activity_class": classification.activity_class,
                    "persistence_class": classification.persistence_class,
                    "slope": float(classification.slope),
                })

    metadata = {
        "config": {
            "fast": config.fast,
            "seed": config.seed,
            "nR": config.nR,
            "nE": config.nE,
            "nTau": config.nTau,
            "nsteps": config.nsteps,
            "R0_range": config.R0_range,
            "dE_range": config.dE_range,
            "tau_range": config.tau_range,
            "T_env_K": config.T_env_K,
            "alpha": config.alpha,
            "E_scale_J": config.E_scale_J,
            "f_end": config.f_end,
        },
        "grids": {
            "R0_vals": R0_vals.tolist(),
            "dE_vals": dE_vals.tolist(),
            "tau_vals": tau_vals.tolist(),
        },
        "rarity_models": [
            {"name": "thermal", "T_env_K": config.T_env_K},
            {"name": "instanton_a", "alpha": config.alpha},
            {"name": "instanton_b", "alpha": config.alpha, "E_scale_J": config.E_scale_J},
        ],
        "total_points": total_points,
    }

    return {
        "metadata": metadata,
        "results": results_list,
    }
