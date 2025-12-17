import os
import subprocess
import sys
import tempfile


def test_master_report_with_dynamics():
    """Test that --report-master generates MASTER_REPORT.md from dynamics artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First generate dynamics artifacts
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
        assert result.returncode == 0, f"dynamics failed: {result.stderr}"

        # Then generate master report
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "emergence_simulator",
                "--report-master",
                "--outdir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"report-master failed: {result.stderr}"

        # Check MASTER_REPORT.md exists
        report_path = os.path.join(tmpdir, "MASTER_REPORT.md")
        assert os.path.exists(report_path), "MASTER_REPORT.md not found"

        # Verify content has key sections
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Executive Summary" in content
        assert "Parameter Definitions" in content
        assert "Complexity Ceilings" in content
        assert "Rarity Models" in content
        assert "Persistence Taxonomy" in content
        assert "Limitations" in content


def test_master_report_with_all_artifacts():
    """Test master report with dynamics and sweep artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate dynamics
        subprocess.run(
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

        # Generate sweep
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

        # Generate master report
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "emergence_simulator",
                "--report-master",
                "--outdir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        report_path = os.path.join(tmpdir, "MASTER_REPORT.md")
        assert os.path.exists(report_path)

        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should include results from both
        assert "Dynamics Simulation" in content
        assert "Comprehensive Sweep" in content
