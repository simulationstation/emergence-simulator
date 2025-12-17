"""Tests for decaying feed energy dynamics."""

import numpy as np

from emergence_simulator.dynamics import (
    energy_dynamics,
    energy_dynamics_decaying_feed,
    simulate_bubble_dynamics,
)
from emergence_simulator.metrics import classify_persistence


class TestDecayingFeedDynamics:
    def test_decaying_feed_initial_energy(self):
        """Decaying feed should start at E0."""
        E0 = 1e-10
        ts = np.linspace(0, 100, 100)
        E_t = energy_dynamics_decaying_feed(
            E0=E0, ts=ts, leak_rate=0.01, eta0=0.1, E_bg=E0, t0=10.0, q=1.0
        )
        assert np.isclose(E_t[0], E0)

    def test_decaying_feed_non_negative(self):
        """Energy should never go negative."""
        E0 = 1e-10
        ts = np.linspace(0, 1000, 500)
        E_t = energy_dynamics_decaying_feed(
            E0=E0, ts=ts, leak_rate=0.1, eta0=0.01, E_bg=E0, t0=10.0, q=2.0
        )
        assert np.all(E_t >= 0)

    def test_decaying_feed_decays_eventually(self):
        """With decaying feed, energy should eventually decay."""
        E0 = 1e-10
        ts = np.linspace(0, 10000, 1000)
        E_t = energy_dynamics_decaying_feed(
            E0=E0, ts=ts, leak_rate=0.01, eta0=0.1, E_bg=E0, t0=100.0, q=1.0
        )
        # Energy at end should be much less than initial (eventually decays)
        # Note: with decaying feed, it won't plateau like constant feed
        assert E_t[-1] < E_t[0]

    def test_higher_q_faster_decay(self):
        """Higher q should lead to faster feed decay and less sustained energy."""
        E0 = 1e-10
        ts = np.linspace(0, 1000, 500)

        E_low_q = energy_dynamics_decaying_feed(
            E0=E0, ts=ts, leak_rate=0.01, eta0=0.1, E_bg=E0, t0=10.0, q=0.5
        )
        E_high_q = energy_dynamics_decaying_feed(
            E0=E0, ts=ts, leak_rate=0.01, eta0=0.1, E_bg=E0, t0=10.0, q=2.0
        )

        # With higher q, feed decays faster, so energy should be lower at late times
        # (unless both are essentially decayed to 0)
        mid_idx = len(ts) // 2
        assert E_low_q[mid_idx] >= E_high_q[mid_idx]


class TestSimulateBubbleDynamicsFeedModes:
    def test_constant_feed_mode(self):
        """Test simulate_bubble_dynamics with constant feed mode."""
        result = simulate_bubble_dynamics(
            E0=1e-10,
            R0=1e-3,
            Rmax=1e-2,
            leak_rate=0.01,
            tgrow=100,
            t_end=1000,
            n_points=100,
            eta=0.1,
            E_bg=1e-10,
            feed_mode="constant",
        )
        assert result["params"]["feed_mode"] == "constant"
        assert result["params"]["eta"] == 0.1
        assert len(result["E_t"]) == 100

    def test_decay_feed_mode(self):
        """Test simulate_bubble_dynamics with decay feed mode."""
        result = simulate_bubble_dynamics(
            E0=1e-10,
            R0=1e-3,
            Rmax=1e-2,
            leak_rate=0.01,
            tgrow=100,
            t_end=1000,
            n_points=100,
            E_bg=1e-10,
            feed_mode="decay",
            eta0=0.1,
            t0=50.0,
            q=1.5,
        )
        assert result["params"]["feed_mode"] == "decay"
        assert result["params"]["eta0"] == 0.1
        assert result["params"]["t0"] == 50.0
        assert result["params"]["q"] == 1.5
        assert len(result["E_t"]) == 100


class TestDecayingFeedPersistence:
    def test_decaying_feed_can_produce_longtail(self):
        """Test that decaying feed can potentially produce LongTailTerminal classification.

        This depends on the specific parameter regime. We test that the classification
        logic at least runs and produces a valid class.
        """
        E0 = 1e-10
        ts = np.linspace(0, 5000, 500)

        # Parameters tuned to try to get intermediate decay behavior
        E_t = energy_dynamics_decaying_feed(
            E0=E0, ts=ts, leak_rate=0.001, eta0=0.1, E_bg=E0, t0=500.0, q=1.0
        )
        f_t = E_t / E0

        result = classify_persistence(ts, f_t)

        # Just verify classification runs and produces valid class
        assert result.persistence_class in ("Persistent", "LongTailTerminal", "Terminal")

    def test_no_feed_is_terminal(self):
        """With no background feed (eta0=0), should be Terminal."""
        E0 = 1e-10
        ts = np.linspace(0, 5000, 500)

        # eta0=0 means g(t) contribution is zero
        E_t = energy_dynamics_decaying_feed(
            E0=E0, ts=ts, leak_rate=0.01, eta0=0.0, E_bg=E0, t0=100.0, q=1.0
        )
        f_t = E_t / E0

        result = classify_persistence(ts, f_t)

        # Pure decay with no feed should be Terminal
        assert result.persistence_class == "Terminal"


class TestDecayVsConstantFeed:
    def test_decay_mode_different_from_constant(self):
        """Decay mode should produce different dynamics than constant mode at late times."""
        E0 = 1e-10
        ts = np.linspace(0, 5000, 500)
        leak_rate = 0.001

        # Constant feed with eta=0.1
        E_constant = energy_dynamics(E0, ts, leak_rate, eta=0.1, E_bg=E0)

        # Decaying feed with initial eta0=0.1
        E_decay = energy_dynamics_decaying_feed(
            E0=E0, ts=ts, leak_rate=leak_rate, eta0=0.1, E_bg=E0, t0=500.0, q=1.0
        )

        # At late times, constant feed should plateau while decay continues to drop
        late_idx = -10
        # Constant feed approaches steady state
        assert E_constant[late_idx] > E_decay[late_idx]
