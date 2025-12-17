import json
import os
import subprocess
import sys
import tempfile

import numpy as np

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
