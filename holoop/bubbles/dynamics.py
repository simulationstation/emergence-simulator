"""Phenomenological bubble dynamics."""

from __future__ import annotations

import math
from typing import Iterable, List

from .bounds import N_ops_max, ops_per_sec_max


def energy_decay(E0: float, lam: float, t: float) -> float:
    """Energy at time ``t`` for exponential leakiness with rate ``lam``."""
    if E0 <= 0:
        return 0.0
    if lam <= 0:
        return E0
    return E0 * math.exp(-lam * t)


def bubble_lifetime(f_end: float, lam: float, tau_override: float | None = None) -> float:
    """Effective bubble lifetime when energy falls below fraction ``f_end``."""
    if tau_override is not None and tau_override > 0:
        return tau_override
    if lam <= 0:
        return float("inf")
    if f_end <= 0:
        return float("inf")
    return math.log(1.0 / f_end) / lam


def activity_proxy(E0: float, lam: float, t: float, mode: str = "ops") -> float:
    """Return activity proxy at time ``t``.

    mode="ops" uses the Margolus–Levitin ceiling on ops/sec, while
    mode="normalized" returns E(t)/E0.
    """
    Et = energy_decay(E0, lam, t)
    if mode == "normalized":
        return Et / E0 if E0 > 0 else 0.0
    return ops_per_sec_max(Et)


def activity_series(
    E0: float, lam: float, times: Iterable[float], mode: str = "ops"
) -> List[float]:
    """Activity proxy evaluated over a time grid."""
    return [activity_proxy(E0, lam, t, mode=mode) for t in times]


def bubble_radius_growth(R0: float, Rmax: float, tgrow: float, t: float) -> float:
    """Phenomenological growth law for bubble radius."""
    if tgrow <= 0:
        return Rmax
    return R0 + (Rmax - R0) * (1.0 - math.exp(-t / tgrow))


def Rmax_from_energy(R0: float, delta_E: float, E_bg: float, beta_grow: float = 1.0) -> float:
    """Limit radius growth based on available excess energy."""
    if E_bg <= 0 or beta_grow <= 0:
        return R0
    scale = max(delta_E / E_bg, 0.0)
    return R0 * (1.0 + beta_grow * scale)


def ops_over_lifetime(E0: float, tau: float) -> float:
    """Total ops upper bound over provided lifetime."""
    return N_ops_max(E0, tau)
