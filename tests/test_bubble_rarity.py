import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from holoop.bubbles import rarity


def test_thermal_logp_monotonic_with_energy():
    logp_low, _ = rarity.thermal_logP(delta_E=1.0, T_env=10.0)
    logp_high, _ = rarity.thermal_logP(delta_E=2.0, T_env=10.0)
    assert logp_high < logp_low


def test_instanton_logp_decreases_with_size_and_energy():
    logp_base = rarity.instanton_logP_a(R=1.0, delta_E=1.0)
    logp_big_R = rarity.instanton_logP_a(R=2.0, delta_E=1.0)
    logp_big_E = rarity.instanton_logP_a(R=1.0, delta_E=2.0)
    assert logp_big_R < logp_base
    assert logp_big_E < logp_base
