import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from holoop.bubbles import bounds


def test_bits_increase_with_radius_and_energy():
    bits_small = bounds.bekenstein_bits(1.0, 1.0)
    bits_large_R = bounds.bekenstein_bits(2.0, 1.0)
    bits_large_E = bounds.bekenstein_bits(1.0, 2.0)
    assert bits_large_R > bits_small
    assert bits_large_E > bits_small


def test_ops_increase_with_energy_and_tau():
    ops_small = bounds.N_ops_max(1.0, 1.0)
    ops_large_E = bounds.N_ops_max(2.0, 1.0)
    ops_large_tau = bounds.N_ops_max(1.0, 2.0)
    assert ops_large_E > ops_small
    assert ops_large_tau > ops_small


def test_bits_formula_matches_hand_calc():
    R = 0.5
    E = 2.0
    bits = bounds.bekenstein_bits(R, E)
    expected = (2 * math.pi * R * E) / (bounds.hbar_c * bounds.ln2)
    assert math.isclose(bits, expected, rel_tol=1e-12)
