import math
from .constants import hbar, c, ln2, pi


def bekenstein_bits(R_m: float, E_J: float) -> float:
    """Bekenstein bound: maximum bits in a region of radius R with energy E."""
    return (2 * pi * R_m * E_J) / (hbar * c * ln2)


def lloyd_ops(E_J: float, tau_s: float) -> float:
    """Lloyd limit: maximum operations given energy E over time tau."""
    ops_per_s = (2 * E_J) / (pi * hbar)
    return ops_per_s * tau_s


def proxy_complexity(bits: float, ops: float) -> dict:
    """Compute proxy complexity metrics."""
    return {
        "log10_bits": math.log10(bits) if bits > 0 else float("-inf"),
        "log10_ops": math.log10(ops) if ops > 0 else float("-inf"),
        "log10_hilbert_dim": bits * ln2 / math.log(10) if bits > 0 else float("-inf"),
    }
