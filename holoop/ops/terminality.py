"""Toy operational terminality metrics."""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple


def window_capacity(times: Iterable[float], activity: Iterable[float]) -> List[Tuple[float, float]]:
    """Compute a simple integrated activity capacity over windows defined by times.

    Returns a list of (T, capacity) where capacity is the mean activity up to that time.
    """
    times_list = list(times)
    activity_list = list(activity)
    if not times_list or not activity_list:
        return []
    capacities = []
    cumulative = 0.0
    for i, (t, a) in enumerate(zip(times_list, activity_list)):
        cumulative += max(a, 0.0)
        window = max(times_list[i], 1e-30)
        capacities.append((t, cumulative / (i + 1)))
    return capacities


def classify_activity(capacities: List[Tuple[float, float]]) -> str:
    """Classify whether activity is negligible or persistent (toy)."""
    if not capacities:
        return "unknown"
    cap_values = [c for _, c in capacities]
    max_cap = max(cap_values)
    if max_cap <= 0:
        return "inactive"
    if max_cap < 1e5:
        return "flicker"
    return "sustained"
