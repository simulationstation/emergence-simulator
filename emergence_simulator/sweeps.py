import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np

from .bubbles import (
    bekenstein_bits,
    lloyd_ops,
    proxy_complexity,
    thermal_logP,
    instanton_a_logP,
    BubbleParams,
    RarityParams,
    bubble_logP,
)


@dataclass
class SweepConfig:
    fast: bool = False

    @property
    def nR(self) -> int:
        return 8 if self.fast else 20

    @property
    def nE(self) -> int:
        return 8 if self.fast else 20

    @property
    def n_tau(self) -> int:
        return 1 if self.fast else 3

    @property
    def R_range(self) -> tuple:
        return (1e-6, 1e3)

    @property
    def dE_range(self) -> tuple:
        return (1e-20, 1e10)

    @property
    def tau_range(self) -> tuple:
        return (1e-9, 1e9)


def run_sweep(config: SweepConfig) -> Dict[str, Any]:
    """Run parameter sweep over R, dE, tau grids."""
    R_vals = np.logspace(
        math.log10(config.R_range[0]),
        math.log10(config.R_range[1]),
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
        config.n_tau,
    )

    # Rarity models
    rarity_configs = [
        RarityParams(model="thermal", T_env_K=2.7),
        RarityParams(model="instanton_a", alpha=1.0),
    ]

    results_list: List[Dict[str, Any]] = []

    for R in R_vals:
        for dE in dE_vals:
            for tau in tau_vals:
                bits = bekenstein_bits(R, dE)
                ops = lloyd_ops(dE, tau)
                complexity = proxy_complexity(bits, ops)

                for rp in rarity_configs:
                    bubble = BubbleParams(R_m=R, dE_J=dE, tau_s=tau)
                    logP = bubble_logP(bubble, rp)
                    log10P = logP / math.log(10) if math.isfinite(logP) else float("-inf")
                    log10_ops = complexity["log10_ops"]

                    # F = log10_ops + log10P (toy expected ops weighted by rarity)
                    if math.isfinite(log10_ops) and math.isfinite(log10P):
                        F = log10_ops + log10P
                    else:
                        F = float("-inf")

                    results_list.append({
                        "R_m": R,
                        "dE_J": dE,
                        "tau_s": tau,
                        "bits": bits,
                        "ops": ops,
                        "log10_bits": complexity["log10_bits"],
                        "log10_ops": log10_ops,
                        "rarity_model": rp.model,
                        "logP": logP,
                        "log10P": log10P,
                        "F": F,
                    })

    metadata = {
        "config": {
            "fast": config.fast,
            "nR": config.nR,
            "nE": config.nE,
            "n_tau": config.n_tau,
            "R_range": config.R_range,
            "dE_range": config.dE_range,
            "tau_range": config.tau_range,
        },
        "rarity_models": [
            {"model": "thermal", "T_env_K": 2.7},
            {"model": "instanton_a", "alpha": 1.0},
        ],
        "R_vals": R_vals.tolist(),
        "dE_vals": dE_vals.tolist(),
        "tau_vals": tau_vals.tolist(),
    }

    return {
        "metadata": metadata,
        "results": results_list,
    }
