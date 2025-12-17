import json
import os
import subprocess
import sys
import tempfile


def test_sweep_eta_produces_nontrivial_distribution():
    """Test that sweep with eta > 0 produces non-trivial persistence distribution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "emergence_simulator",
                "--sweep-all",
                "--fast",
                "--outdir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        sweep_dir = os.path.join(tmpdir, "sweep")
        json_path = os.path.join(sweep_dir, "sweep_results.json")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check eta_summary exists
        eta_summary = data["metadata"].get("eta_summary", {})
        assert eta_summary, "eta_summary not found in metadata"

        # For eta > 0, should have some non-Terminal classifications
        non_terminal_count = 0
        for eta_str, counts in eta_summary.items():
            eta = float(eta_str)
            if eta > 0:
                non_terminal = counts["Persistent"] + counts["LongTailTerminal"]
                non_terminal_count += non_terminal

        # Assert that with eta > 0, we get at least some non-Terminal points
        assert non_terminal_count > 0, (
            f"Expected some Persistent or LongTailTerminal points for eta > 0, "
            f"but all were Terminal. eta_summary: {eta_summary}"
        )


def test_sweep_eta_zero_all_terminal():
    """Test that sweep with eta=0 produces mostly Terminal classifications."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "emergence_simulator",
                "--sweep-all",
                "--fast",
                "--outdir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        sweep_dir = os.path.join(tmpdir, "sweep")
        json_path = os.path.join(sweep_dir, "sweep_results.json")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        eta_summary = data["metadata"].get("eta_summary", {})

        # For eta=0, should have all Terminal
        eta_zero_counts = eta_summary.get("0.0", {})
        if eta_zero_counts:
            terminal_count = eta_zero_counts.get("Terminal", 0)
            total = eta_zero_counts.get("total", 1)
            terminal_frac = terminal_count / total

            # All eta=0 points should be Terminal
            assert terminal_frac == 1.0, (
                f"Expected all eta=0 points to be Terminal, got {terminal_frac:.1%}"
            )


def test_sweep_eta_creates_plot():
    """Test that sweep creates fraction_terminal_vs_eta.png plot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "emergence_simulator",
                "--sweep-all",
                "--fast",
                "--outdir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
        )

        sweep_dir = os.path.join(tmpdir, "sweep")
        plot_path = os.path.join(sweep_dir, "fraction_terminal_vs_eta.png")

        assert os.path.exists(plot_path), "fraction_terminal_vs_eta.png not found"
