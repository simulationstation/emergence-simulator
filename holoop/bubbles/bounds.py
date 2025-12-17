"""Complexity ceilings for toy quantum bubbles."""

from __future__ import annotations

import math

from .constants import hbar, hbar_c, ln2, pi


def bekenstein_bits(R: float, E: float) -> float:
    """Maximum bits storable in a region (toy Bekenstein-like bound).

    Parameters
    ----------
    R: float
        Bubble radius in meters.
    E: float
        Total energy available (joules).

    Returns
    -------
    float
        Upper bound on number of bits.
    """
    if R <= 0 or E <= 0:
        return 0.0
    return (2.0 * pi * R * E) / (hbar_c * ln2)


def bits_and_hilbert_dim(R: float, E: float) -> tuple[float, float]:
    """Return bits ceiling and log10 of Hilbert space dimension."""
    bits = bekenstein_bits(R, E)
    log10_dim = bits * math.log10(2.0)
    return bits, log10_dim


def ops_per_sec_max(E: float) -> float:
    """Margolus–Levitin/Lloyd-like bound on ops per second.

    Returns a conservative upper bound on logical operations per second
    for a device with energy ``E``. This is a ceiling, not a claim of
    achievable performance.
    """
    if E <= 0:
        return 0.0
    return 2.0 * E / (pi * hbar)


def N_ops_max(E: float, tau: float) -> float:
    """Upper bound on total operations over time window ``tau`` (seconds)."""
    if E <= 0 or tau <= 0:
        return 0.0
    return ops_per_sec_max(E) * tau


def complexity_proxy(bits: float, n_ops: float) -> float:
    """Toy ceiling complexity proxy in log-space.

    Returns
    -------
    float
        Minimum of log10(N_ops_max) and bits * log10(2). If either term
        is non-positive, negative infinity is returned.
    """
    if bits <= 0 or n_ops <= 0:
        return float("-inf")
    return min(math.log10(n_ops), bits * math.log10(2.0))
