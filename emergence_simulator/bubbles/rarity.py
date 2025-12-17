import math
from .constants import kB, hbar, c
from .types import BubbleParams, RarityParams


def thermal_logP(dE_J: float, T_env_K: float) -> float:
    """Thermal activation log-probability: ln(P) = -dE/(kB*T)."""
    if T_env_K <= 0:
        return float("-inf")
    return -dE_J / (kB * T_env_K)


def instanton_a_logP(R_m: float, dE_J: float, alpha: float) -> float:
    """Instanton model A log-probability: ln(P) = -(alpha * R * dE)/(hbar*c)."""
    return -(alpha * R_m * dE_J) / (hbar * c)


def instanton_b_logP(R_m: float, dE_J: float, alpha: float, E_scale_J: float) -> float:
    """Instanton model B log-probability: ln(P) = -(alpha * dE^2 * R/(c*E_scale))/hbar."""
    return -(alpha * (dE_J ** 2) * (R_m / (c * E_scale_J))) / hbar


def bubble_logP(bubble: BubbleParams, rp: RarityParams) -> float:
    """Compute log-probability for a bubble given rarity parameters."""
    if rp.model == "thermal":
        return thermal_logP(bubble.dE_J, rp.T_env_K)
    elif rp.model == "instanton_a":
        return instanton_a_logP(bubble.R_m, bubble.dE_J, rp.alpha)
    elif rp.model == "instanton_b":
        return instanton_b_logP(bubble.R_m, bubble.dE_J, rp.alpha, rp.E_scale_J)
    else:
        raise ValueError(f"Unknown rarity model: {rp.model}")
