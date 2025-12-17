import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class PersistenceClassification:
    activity_class: str  # ContinuousActive / IntermittentActive / InstantaneousTerminal
    persistence_class: str  # Persistent / LongTailTerminal / Terminal
    slope: float
    final_C_win: float
    rationale: str


def window_capacity(ts: np.ndarray, f: np.ndarray, T_values: np.ndarray) -> np.ndarray:
    """
    Compute window-integrated activity C_win(T) = integral of f(t) over [T, 2T].

    Uses trapezoidal integration for each T value.
    """
    C_win = np.zeros(len(T_values))

    for i, T in enumerate(T_values):
        # Find indices in window [T, 2T]
        mask = (ts >= T) & (ts <= 2 * T)
        if np.sum(mask) < 2:
            C_win[i] = 0.0
            continue

        ts_window = ts[mask]
        f_window = f[mask]
        C_win[i] = np.trapezoid(f_window, ts_window)

    return C_win


def fit_tail_slope(T_values: np.ndarray, C_win: np.ndarray, tail_fraction: float = 0.5) -> float:
    """
    Fit slope on the tail (last tail_fraction) of log(C_win) vs log(T).

    Returns the slope (exponent in power-law decay).
    """
    # Use only positive C_win values
    valid = C_win > 0
    if np.sum(valid) < 2:
        return float("-inf")

    T_valid = T_values[valid]
    C_valid = C_win[valid]

    # Take tail fraction
    n_tail = max(2, int(len(T_valid) * tail_fraction))
    T_tail = T_valid[-n_tail:]
    C_tail = C_valid[-n_tail:]

    if len(T_tail) < 2:
        return float("-inf")

    # Linear fit in log-log space
    log_T = np.log(T_tail)
    log_C = np.log(C_tail)

    # Simple least squares: slope = cov(x,y) / var(x)
    slope = np.polyfit(log_T, log_C, 1)[0]
    return slope


def classify_persistence(
    ts: np.ndarray,
    f: np.ndarray,
    T_values: np.ndarray = None,
) -> PersistenceClassification:
    """
    Classify bubble persistence based on window-integrated activity.

    Activity classes:
    - ContinuousActive: activity stays high (f > 0.5 for most of time)
    - IntermittentActive: activity fluctuates
    - InstantaneousTerminal: activity drops quickly

    Persistence classes:
    - Persistent: C_win slope >= -1 (slow or no decay)
    - LongTailTerminal: -2 < slope < -1 (power-law decay)
    - Terminal: slope <= -2 or C_win goes to zero (fast decay)
    """
    if T_values is None:
        t_max = ts[-1]
        # Generate T values from small fraction to ~half of t_max
        T_values = np.logspace(np.log10(t_max / 100), np.log10(t_max / 2.5), 20)

    C_win = window_capacity(ts, f, T_values)
    slope = fit_tail_slope(T_values, C_win)

    # Get final C_win value (last valid)
    valid_C = C_win[C_win > 0]
    final_C_win = valid_C[-1] if len(valid_C) > 0 else 0.0

    # Activity classification based on f behavior
    mean_f = np.mean(f)
    final_f = f[-1] if len(f) > 0 else 0.0

    if mean_f > 0.5 and final_f > 0.3:
        activity_class = "ContinuousActive"
    elif mean_f > 0.1:
        activity_class = "IntermittentActive"
    else:
        activity_class = "InstantaneousTerminal"

    # Persistence classification based on C_win slope
    if slope >= -1.0 or (np.isfinite(slope) and final_C_win > 0.1 * T_values[-1]):
        persistence_class = "Persistent"
        rationale = f"Slope {slope:.2f} >= -1: window capacity decays slowly or not at all"
    elif slope > -2.0:
        persistence_class = "LongTailTerminal"
        rationale = f"Slope {slope:.2f} in (-2, -1): power-law decay with long tail"
    else:
        persistence_class = "Terminal"
        rationale = f"Slope {slope:.2f} <= -2: fast exponential-like decay"

    return PersistenceClassification(
        activity_class=activity_class,
        persistence_class=persistence_class,
        slope=slope,
        final_C_win=final_C_win,
        rationale=rationale,
    )


def compute_dynamics_metrics(sim_result: dict) -> dict:
    """
    Compute all metrics from a simulation result.

    Returns dict with T_values, C_win, classification, and summary stats.
    """
    ts = sim_result["ts"]
    f_t = sim_result["f_t"]

    t_max = ts[-1]
    T_values = np.logspace(np.log10(t_max / 100), np.log10(t_max / 2.5), 30)

    C_win = window_capacity(ts, f_t, T_values)
    classification = classify_persistence(ts, f_t, T_values)

    return {
        "T_values": T_values.tolist(),
        "C_win": C_win.tolist(),
        "classification": {
            "activity_class": classification.activity_class,
            "persistence_class": classification.persistence_class,
            "slope": classification.slope,
            "final_C_win": classification.final_C_win,
            "rationale": classification.rationale,
        },
        "summary": {
            "mean_f": float(np.mean(f_t)),
            "final_f": float(f_t[-1]),
            "total_activity": float(np.trapezoid(f_t, ts)),
        },
    }
