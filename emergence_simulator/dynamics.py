import numpy as np
from .bubbles.constants import pi, hbar


def energy_decay(E0: float, ts: np.ndarray, leak_rate: float) -> np.ndarray:
    """Exponential energy decay: E(t) = E0 * exp(-leak_rate * t)."""
    return E0 * np.exp(-leak_rate * ts)


def radius_growth(R0: float, ts: np.ndarray, Rmax: float, tgrow: float) -> np.ndarray:
    """Asymptotic radius growth: R(t) = R0 + (Rmax - R0) * (1 - exp(-t/tgrow))."""
    return R0 + (Rmax - R0) * (1 - np.exp(-ts / tgrow))


def activity_ops_per_s(E_t: np.ndarray) -> np.ndarray:
    """Lloyd-limit activity rate: ops/s = 2*E/(pi*hbar)."""
    return (2 * E_t) / (pi * hbar)


def normalized_activity(E_t: np.ndarray, E0: float) -> np.ndarray:
    """Normalized activity: f(t) = E(t) / E0."""
    if E0 == 0:
        return np.zeros_like(E_t)
    return E_t / E0


def simulate_bubble_dynamics(
    E0: float,
    R0: float,
    Rmax: float,
    leak_rate: float,
    tgrow: float,
    t_end: float,
    n_points: int = 500,
) -> dict:
    """
    Simulate bubble expansion and evaporation dynamics.

    Returns dict with:
    - ts: time array
    - E_t: energy over time
    - R_t: radius over time
    - f_t: normalized activity (E/E0)
    - ops_per_s: activity in ops/s
    """
    ts = np.linspace(0, t_end, n_points)
    E_t = energy_decay(E0, ts, leak_rate)
    R_t = radius_growth(R0, ts, Rmax, tgrow)
    f_t = normalized_activity(E_t, E0)
    ops_s = activity_ops_per_s(E_t)

    return {
        "ts": ts,
        "E_t": E_t,
        "R_t": R_t,
        "f_t": f_t,
        "ops_per_s": ops_s,
        "params": {
            "E0": E0,
            "R0": R0,
            "Rmax": Rmax,
            "leak_rate": leak_rate,
            "tgrow": tgrow,
            "t_end": t_end,
        },
    }
