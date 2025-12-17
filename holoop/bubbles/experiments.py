"""Parameter sweeps for bubble rarity and complexity."""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Tuple

from . import bounds, rarity


def _logspace(min_val: float, max_val: float, n: int) -> List[float]:
    if n <= 1:
        return [min_val]
    log_min = math.log10(min_val)
    log_max = math.log10(max_val)
    return [10 ** (log_min + (log_max - log_min) * i / (n - 1)) for i in range(n)]


def sweep_parameters(
    R_min: float,
    R_max: float,
    nR: int,
    E_min: float,
    E_max: float,
    nE: int,
    tau_min: float,
    tau_max: float,
    nTau: int,
) -> Tuple[List[float], List[float], List[float]]:
    Rs = _logspace(R_min, R_max, nR)
    Es = _logspace(E_min, E_max, nE)
    taus = _logspace(tau_min, tau_max, nTau)
    return Rs, Es, taus


def run_bubble_sweep(
    rarity_model: str,
    R_min: float = 1e-6,
    R_max: float = 1e3,
    nR: int = 32,
    E_min: float = 1e-20,
    E_max: float = 1e10,
    nE: int = 32,
    tau_min: float = 1e-9,
    tau_max: float = 1e9,
    nTau: int = 8,
    T_env: float = 300.0,
    alpha: float = 1.0,
    E_scale: float = 1e-9,
    f_end: float = 1e-3,
    seed: int | None = None,
) -> Dict:
    """Compute grid of bubble metrics and return results dictionary."""
    Rs, Es, taus = sweep_parameters(R_min, R_max, nR, E_min, E_max, nE, tau_min, tau_max, nTau)
    results: List[Dict] = []

    for R in Rs:
        for E in Es:
            for tau in taus:
                lam = math.log(1.0 / f_end) / tau
                bits, log10_dim = bounds.bits_and_hilbert_dim(R, E)
                n_ops = bounds.N_ops_max(E, tau)
                log10_ops = math.log10(n_ops) if n_ops > 0 else float("-inf")
                log10_bits = math.log10(bits) if bits > 0 else float("-inf")
                rarity_metrics = rarity.rarity_from_model(
                    rarity_model, R, E, T_env=T_env, alpha=alpha, E_scale=E_scale
                )
                logP = rarity_metrics["logP"]
                log10P = rarity_metrics["log10P"]
                F = log10_ops + log10P if math.isfinite(log10_ops) else float("-inf")
                G = log10_bits + log10P if math.isfinite(log10_bits) else float("-inf")

                results.append(
                    {
                        "R": float(R),
                        "E": float(E),
                        "tau": float(tau),
                        "lambda": float(lam),
                        "bits_max": bits,
                        "log10_dim": log10_dim,
                        "ops_max": n_ops,
                        "log10_ops": log10_ops,
                        "log10_bits": log10_bits,
                        "rarity": rarity_metrics,
                        "F": F,
                        "G": G,
                    }
                )

    summary = summarize_results(results)
    return {
        "params": {
            "R_min": R_min,
            "R_max": R_max,
            "nR": nR,
            "E_min": E_min,
            "E_max": E_max,
            "nE": nE,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "nTau": nTau,
            "rarity_model": rarity_model,
            "T_env": T_env,
            "alpha": alpha,
            "E_scale": E_scale,
            "f_end": f_end,
            "seed": seed,
        },
        "results": results,
        "summary": summary,
    }


def summarize_results(results: List[Dict]) -> Dict:
    if not results:
        return {}
    by_ops = sorted(results, key=lambda r: r["ops_max"], reverse=True)
    by_bits = sorted(results, key=lambda r: r["bits_max"], reverse=True)
    by_F = sorted(results, key=lambda r: (r["F"] if math.isfinite(r["F"]) else -math.inf), reverse=True)

    def pick(entry):
        return {
            "R": entry["R"],
            "E": entry["E"],
            "tau": entry["tau"],
            "bits_max": entry["bits_max"],
            "ops_max": entry["ops_max"],
            "F": entry["F"],
            "log10P": entry["rarity"].get("log10P", float("-inf")),
        }

    return {
        "top_ops": pick(by_ops[0]),
        "top_bits": pick(by_bits[0]),
        "top_F": pick(by_F[0]),
        "count": len(results),
    }


def save_results(results: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
