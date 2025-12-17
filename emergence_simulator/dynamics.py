import numpy as np
from .bubbles.constants import pi, hbar


def energy_decay(E0: float, ts: np.ndarray, leak_rate: float) -> np.ndarray:
    """Exponential energy decay: E(t) = E0 * exp(-leak_rate * t)."""
    return E0 * np.exp(-leak_rate * ts)


def energy_dynamics(
    E0: float,
    ts: np.ndarray,
    leak_rate: float,
    eta: float = 0.0,
    E_bg: float = 0.0,
) -> np.ndarray:
    """
    Energy dynamics with constant background feed term.

    ODE: dE/dt = -leak_rate * E + eta * E_bg

    Analytical solution:
    If leak_rate > 0:
        E(t) = E0 * exp(-λt) + (eta * E_bg / λ) * (1 - exp(-λt))
    If leak_rate == 0:
        E(t) = E0 + eta * E_bg * t

    Parameters:
    - E0: Initial energy (J)
    - ts: Time array (s)
    - leak_rate: Energy dissipation rate λ (s^-1)
    - eta: Background feed coupling constant (dimensionless)
    - E_bg: Background energy scale (J)

    Returns:
    - E(t): Energy array over time
    """
    if leak_rate > 0:
        exp_term = np.exp(-leak_rate * ts)
        steady_state = (eta * E_bg) / leak_rate
        return E0 * exp_term + steady_state * (1 - exp_term)
    else:
        # No decay: linear growth from background feed
        return E0 + eta * E_bg * ts


def energy_dynamics_decaying_feed(
    E0: float,
    ts: np.ndarray,
    leak_rate: float,
    eta0: float,
    E_bg: float,
    t0: float,
    q: float,
) -> np.ndarray:
    """
    Energy dynamics with decaying (power-law) background feed.

    ODE: dE/dt = -λ E + η0 * E_bg * g(t)
    where g(t) = 1 / (1 + t/t0)^q is a power-law decay.

    This creates intermediate tails where energy decays more slowly than
    pure exponential but does not plateau, enabling LongTailTerminal
    classification if the taxonomy is meaningful.

    Uses explicit Euler integration (stable for sufficiently dense ts).

    Parameters:
    - E0: Initial energy (J)
    - ts: Time array (s), assumed sorted ascending
    - leak_rate: Energy dissipation rate λ (s^-1)
    - eta0: Initial background feed coupling (dimensionless)
    - E_bg: Background energy scale (J)
    - t0: Characteristic decay timescale for feed (s)
    - q: Power-law exponent for feed decay (dimensionless)

    Returns:
    - E(t): Energy array over time, clamped to >= 0
    """
    n = len(ts)
    E = np.zeros(n)
    E[0] = E0

    for i in range(n - 1):
        t = ts[i]
        dt = ts[i + 1] - ts[i]

        # g(t) = 1 / (1 + t/t0)^q
        g_t = 1.0 / (1.0 + t / t0) ** q

        # dE/dt = -λ E + η0 * E_bg * g(t)
        dE_dt = -leak_rate * E[i] + eta0 * E_bg * g_t

        # Euler step, clamp to non-negative
        E[i + 1] = max(0.0, E[i] + dt * dE_dt)

    return E


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
    eta: float = 0.0,
    E_bg: float = 0.0,
    feed_mode: str = "constant",
    eta0: float = 0.0,
    t0: float = 1.0,
    q: float = 1.0,
) -> dict:
    """
    Simulate bubble expansion and evaporation dynamics.

    Supports two feed modes:
    - "constant": Constant background feed (analytic solution, uses eta)
    - "decay": Decaying power-law feed (numerical, uses eta0, t0, q)

    Parameters:
    - E0: Initial energy (J)
    - R0: Initial radius (m)
    - Rmax: Maximum radius (m)
    - leak_rate: Energy dissipation rate (s^-1)
    - tgrow: Radius growth timescale (s)
    - t_end: Simulation end time (s)
    - n_points: Number of time points
    - eta: Background feed coupling for constant mode (dimensionless)
    - E_bg: Background energy scale (J)
    - feed_mode: "constant" or "decay"
    - eta0: Initial feed coupling for decay mode (dimensionless)
    - t0: Feed decay timescale for decay mode (s)
    - q: Feed decay power-law exponent for decay mode

    Returns dict with:
    - ts: time array
    - E_t: energy over time
    - R_t: radius over time
    - f_t: normalized activity (E/E0)
    - ops_per_s: activity in ops/s
    - params: simulation parameters
    """
    ts = np.linspace(0, t_end, n_points)

    if feed_mode == "decay":
        E_t = energy_dynamics_decaying_feed(E0, ts, leak_rate, eta0, E_bg, t0, q)
    else:
        # Default: constant feed mode
        E_t = energy_dynamics(E0, ts, leak_rate, eta, E_bg)

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
            "feed_mode": feed_mode,
            "eta": eta,
            "E_bg": E_bg,
            "eta0": eta0,
            "t0": t0,
            "q": q,
        },
    }
