from dataclasses import dataclass
from typing import Optional


@dataclass
class BubbleParams:
    R_m: float  # radius (m)
    dE_J: float  # excess energy (J)
    tau_s: float  # lifetime (s)
    leak_rate_1s: Optional[float] = None  # optional, if None derive from tau & f_end


@dataclass
class RarityParams:
    model: str  # "thermal" | "instanton_a" | "instanton_b"
    T_env_K: float = 0.0
    alpha: float = 1.0
    E_scale_J: float = 1e-9
