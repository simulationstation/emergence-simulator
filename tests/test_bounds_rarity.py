from emergence_simulator.bubbles import (
    bekenstein_bits,
    lloyd_ops,
    thermal_logP,
    instanton_a_logP,
    instanton_b_logP,
)


class TestBounds:
    def test_bits_increases_with_R(self):
        E = 1e-10
        bits1 = bekenstein_bits(1.0, E)
        bits2 = bekenstein_bits(2.0, E)
        assert bits2 > bits1

    def test_bits_increases_with_E(self):
        R = 1.0
        bits1 = bekenstein_bits(R, 1e-10)
        bits2 = bekenstein_bits(R, 2e-10)
        assert bits2 > bits1

    def test_ops_increases_with_E(self):
        tau = 1.0
        ops1 = lloyd_ops(1e-10, tau)
        ops2 = lloyd_ops(2e-10, tau)
        assert ops2 > ops1

    def test_ops_increases_with_tau(self):
        E = 1e-10
        ops1 = lloyd_ops(E, 1.0)
        ops2 = lloyd_ops(E, 2.0)
        assert ops2 > ops1


class TestRarity:
    def test_thermal_decreases_with_dE(self):
        T = 300.0
        logP1 = thermal_logP(1e-20, T)
        logP2 = thermal_logP(2e-20, T)
        assert logP2 < logP1

    def test_thermal_zero_temp_returns_neginf(self):
        logP = thermal_logP(1e-20, 0.0)
        assert logP == float("-inf")

    def test_instanton_a_decreases_with_R(self):
        dE = 1e-10
        alpha = 1.0
        logP1 = instanton_a_logP(1.0, dE, alpha)
        logP2 = instanton_a_logP(2.0, dE, alpha)
        assert logP2 < logP1

    def test_instanton_a_decreases_with_dE(self):
        R = 1.0
        alpha = 1.0
        logP1 = instanton_a_logP(R, 1e-10, alpha)
        logP2 = instanton_a_logP(R, 2e-10, alpha)
        assert logP2 < logP1

    def test_instanton_b_decreases_with_R(self):
        dE = 1e-10
        alpha = 1.0
        E_scale = 1e-9
        logP1 = instanton_b_logP(1.0, dE, alpha, E_scale)
        logP2 = instanton_b_logP(2.0, dE, alpha, E_scale)
        assert logP2 < logP1

    def test_instanton_b_decreases_with_dE(self):
        R = 1.0
        alpha = 1.0
        E_scale = 1e-9
        logP1 = instanton_b_logP(R, 1e-10, alpha, E_scale)
        logP2 = instanton_b_logP(R, 2e-10, alpha, E_scale)
        assert logP2 < logP1
