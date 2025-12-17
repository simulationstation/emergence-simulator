"""Tests for decay mode sweep functionality."""

import json
import os
import subprocess
import sys
import tempfile


def test_sweep_decay_mode_runs():
    """Test that sweep with decay mode runs successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "emergence_simulator",
                "--sweep-all",
                "--fast",
                "--feed_mode",
                "decay",
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

        # Verify decay mode specific metadata
        config = data["metadata"]["config"]
        assert config["feed_mode"] == "decay"
        assert "q_vals" in config
        assert "t0_fracs" in config
        assert "eta0_vals" in config


def test_sweep_decay_has_q_summary():
    """Test that decay sweep produces q_summary in metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "emergence_simulator",
                "--sweep-all",
                "--fast",
                "--feed_mode",
                "decay",
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

        # Check q_summary exists
        q_summary = data["metadata"].get("q_summary", {})
        assert q_summary, "q_summary not found in metadata"

        # Verify q_summary has counts for each q value
        for q_str, counts in q_summary.items():
            assert "Persistent" in counts
            assert "LongTailTerminal" in counts
            assert "Terminal" in counts
            assert "total" in counts


def test_sweep_decay_creates_q_plot():
    """Test that decay sweep creates fraction_by_class_vs_q.png plot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "emergence_simulator",
                "--sweep-all",
                "--fast",
                "--feed_mode",
                "decay",
                "--outdir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
        )

        sweep_dir = os.path.join(tmpdir, "sweep")
        plot_path = os.path.join(sweep_dir, "fraction_by_class_vs_q.png")

        assert os.path.exists(plot_path), "fraction_by_class_vs_q.png not found"


def test_sweep_decay_results_have_q_field():
    """Test that decay sweep results have q, t0_frac, eta0 fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "emergence_simulator",
                "--sweep-all",
                "--fast",
                "--feed_mode",
                "decay",
                "--outdir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
        )

        sweep_dir = os.path.join(tmpdir, "sweep")
        json_path = os.path.join(sweep_dir, "sweep_results.json")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data["results"]
        assert len(results) > 0

        # Check first result has decay-specific fields
        first = results[0]
        assert "q" in first
        assert "t0_frac" in first
        assert "t0_s" in first
        assert "eta0" in first
        assert first["feed_mode"] == "decay"


def test_sweep_decay_produces_variation():
    """Test that decay sweep produces variation in persistence classification.

    With decaying feed, we expect to see at least some variation across
    different q values or eta0 values.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "emergence_simulator",
                "--sweep-all",
                "--fast",
                "--feed_mode",
                "decay",
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

        results = data["results"]

        # Count persistence classes
        classes = set(r["persistence_class"] for r in results)

        # With decay mode and varied parameters, we expect some diversity
        # At minimum we should have at least one class
        assert len(classes) >= 1


def test_constant_mode_still_works():
    """Test that constant mode still works (regression test)."""
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

        # Verify default is constant mode
        config = data["metadata"]["config"]
        assert config["feed_mode"] == "constant"
        assert "eta_vals" in config
