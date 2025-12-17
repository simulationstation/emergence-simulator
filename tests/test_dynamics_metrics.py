import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from emergence_simulator.dynamics import energy_dynamics, simulate_bubble_dynamics
from emergence_simulator.metrics import classify_persistence, window_capacity


class TestPersistenceClassification:
    def test_constant_activity_is_persistent(self):
        """Constant f(t) = 1 should be classified as Persistent."""
        ts = np.linspace(0, 1000, 500)
        f = np.ones_like(ts)  # Constant activity

        result = classify_persistence(ts, f)

        assert result.persistence_class == "Persistent"

    def test_strong_decay_is_not_persistent(self):
        """Strongly decaying f(t) should be Terminal or LongTailTerminal."""
        ts = np.linspace(0, 1000, 500)
        # Strong exponential decay with short e-folding time
        f = np.exp(-ts / 10)  # Fast decay

        result = classify_persistence(ts, f)

        assert result.persistence_class in ("Terminal", "LongTailTerminal")
        assert result.persistence_class != "Persistent"

    def test_moderate_decay_classification(self):
        """Moderate decay should be LongTailTerminal or Terminal."""
        ts = np.linspace(0, 1000, 500)
        f = np.exp(-ts / 100)  # Moderate decay

        result = classify_persistence(ts, f)

        # Should not be persistent (decays significantly)
        assert result.slope < 0


class TestWindowCapacity:
    def test_window_capacity_constant(self):
        """Window capacity of constant f=1 over [T, 2T] should be ~T."""
        ts = np.linspace(0, 100, 1000)
        f = np.ones_like(ts)
        T_values = np.array([5.0, 10.0, 20.0])

        C_win = window_capacity(ts, f, T_values)

        # For f=1, integral over [T, 2T] = 2T - T = T
        for i, T in enumerate(T_values):
            assert abs(C_win[i] - T) < 0.1 * T


class TestDynamicsCLI:
    def test_bubble_dynamics_creates_artifacts(self):
        """Test that --bubble-dynamics creates expected artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "emergence_simulator",
                    "--bubble-dynamics",
                    "--fast",
                    "--outdir",
                    tmpdir,
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"

            dynamics_dir = os.path.join(tmpdir, "dynamics")

            # Check JSON exists
            json_path = os.path.join(dynamics_dir, "dynamics_results.json")
            assert os.path.exists(json_path), "dynamics_results.json not found"

            # Check report exists
            report_path = os.path.join(dynamics_dir, "dynamics_report.md")
            assert os.path.exists(report_path), "dynamics_report.md not found"

            # Check at least 2 plots exist
            plots = [f for f in os.listdir(dynamics_dir) if f.endswith(".png")]
            assert len(plots) >= 2, f"Expected at least 2 plots, found {len(plots)}"

            # Validate JSON structure
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "simulation" in data
            assert "metrics" in data
            assert "classification" in data["metrics"]


class TestEtaBackgroundFeed:
    def test_eta_zero_exponential_decay_terminal(self):
        """With eta=0, pure exponential decay should classify as Terminal."""
        E0 = 1e-10
        leak_rate = 0.01  # Fast decay
        ts = np.linspace(0, 1000, 500)

        # eta=0 means pure decay
        E_t = energy_dynamics(E0, ts, leak_rate, eta=0.0, E_bg=E0)
        f_t = E_t / E0

        result = classify_persistence(ts, f_t)

        # Pure exponential decay should be Terminal
        assert result.persistence_class in ("Terminal", "LongTailTerminal")

    def test_large_eta_sustains_activity_persistent(self):
        """With large eta, background feed sustains activity -> Persistent."""
        E0 = 1e-10
        leak_rate = 0.01
        eta = 0.1  # Strong background feed
        E_bg = E0
        ts = np.linspace(0, 1000, 500)

        # With large eta, E(t) approaches steady state = eta * E_bg / leak_rate
        E_t = energy_dynamics(E0, ts, leak_rate, eta=eta, E_bg=E_bg)
        f_t = E_t / E0

        # Check that activity plateaus (doesn't decay to near-zero)
        final_f = f_t[-1]
        assert final_f > 0.5, f"Expected sustained activity, got final_f={final_f}"

        result = classify_persistence(ts, f_t)

        # Sustained activity should classify as Persistent or at least not Terminal
        assert result.persistence_class == "Persistent", f"Expected Persistent, got {result.persistence_class}"

    def test_energy_dynamics_steady_state(self):
        """Verify energy dynamics reaches correct steady state."""
        E0 = 1e-10
        leak_rate = 0.01
        eta = 0.1
        E_bg = E0

        # Expected steady state: E_ss = eta * E_bg / leak_rate
        E_ss_expected = eta * E_bg / leak_rate

        ts = np.linspace(0, 10000, 1000)  # Long enough to reach steady state
        E_t = energy_dynamics(E0, ts, leak_rate, eta=eta, E_bg=E_bg)

        # Check final value is close to steady state
        assert abs(E_t[-1] - E_ss_expected) / E_ss_expected < 0.01

    def test_simulate_bubble_dynamics_with_eta(self):
        """Test that simulate_bubble_dynamics accepts eta parameters."""
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
        )

        assert "params" in result
        assert result["params"]["eta"] == 0.1
        assert result["params"]["E_bg"] == 1e-10
        assert len(result["f_t"]) == 100
