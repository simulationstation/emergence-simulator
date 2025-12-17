"""Parametric rarity / event rate models for toy quantum bubbles."""

from __future__ import annotations

import math
from typing import Dict, Tuple

from .constants import c, hbar, kB

LOG10E = math.log10(math.e)


def safe_exp(log_value: float) -> float:
    """Exponentiate a log-value with underflow protection."""
    if log_value < -745:  # below typical double precision underflow threshold
        return 0.0
    return math.exp(log_value)


def thermal_logP(delta_E: float, T_env: float) -> Tuple[float, float]:
    """Thermal-like rarity: P ~ exp(-ΔE / (kB T)).

    Returns (logP, P). If T_env <= 0, returns (-inf, 0).
    """
    if delta_E <= 0:
        return 0.0, 1.0
    if T_env <= 0:
        return float("-inf"), 0.0
    logP = -delta_E / (kB * T_env)
    return logP, safe_exp(logP)


def instanton_logP_a(R: float, delta_E: float, alpha: float = 1.0) -> float:
    """Instanton-like rarity with action cost proportional to R * ΔE / c."""
    if R <= 0 or delta_E <= 0 or alpha <= 0:
        return float("-inf")
    B = alpha * R * delta_E / c
    return -B / hbar


def instanton_logP_b(
    R: float, delta_E: float, alpha: float = 1.0, E_scale: float = 1e-9
) -> float:
    """Instanton-like rarity with quadratic energy cost.

    B = alpha * (ΔE^2) * (R / c) / E_scale
    """
    if R <= 0 or delta_E <= 0 or alpha <= 0 or E_scale <= 0:
        return float("-inf")
    B = alpha * (delta_E ** 2) * (R / c) / E_scale
    return -B / hbar


def rarity_from_model(
    model: str,
    R: float,
    delta_E: float,
    T_env: float = 300.0,
    alpha: float = 1.0,
    E_scale: float = 1e-9,
) -> Dict[str, float]:
    """Compute rarity metrics for the chosen model.

    Returns a dictionary with keys: logP (natural), log10P, P.
    """
    model = model.lower()
    if model == "thermal":
        logP, P = thermal_logP(delta_E, T_env)
    elif model == "instanton_a":
        logP = instanton_logP_a(R, delta_E, alpha=alpha)
        P = safe_exp(logP)
    elif model == "instanton_b":
        logP = instanton_logP_b(R, delta_E, alpha=alpha, E_scale=E_scale)
        P = safe_exp(logP)
    else:
        raise ValueError(f"Unknown rarity model '{model}'")

    log10P = logP * LOG10E if math.isfinite(logP) else float("-inf")
    return {"logP": logP, "log10P": log10P, "P": P}


def expected_events(
    logP: float, rate0: float = 1e-30, volume: float = 1.0, duration: float = 1.0
) -> float:
    """Toy expected number of events in a space-time region.

    Parameters use arbitrary units; this is purely illustrative.
    """
    if rate0 <= 0 or volume <= 0 or duration <= 0:
        return 0.0
    rate = rate0 * safe_exp(logP)
    return rate * volume * duration
