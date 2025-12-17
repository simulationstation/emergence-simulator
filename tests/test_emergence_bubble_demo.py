import json
import os
import subprocess
import sys
import tempfile


def test_emergence_bubble_demo_fast():
    """Test that bubble demo creates expected artifacts in fast mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, "-m", "emergence_simulator", "--bubble-demo", "--fast", "--outdir", tmpdir],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        bubble_dir = os.path.join(tmpdir, "bubbles")

        # Check JSON results exist
        json_path = os.path.join(bubble_dir, "bubble_results.json")
        assert os.path.exists(json_path), "bubble_results.json not found"

        # Check report exists
        report_path = os.path.join(bubble_dir, "bubble_report.md")
        assert os.path.exists(report_path), "bubble_report.md not found"

        # Check at least 2 plot files exist
        plots = [f for f in os.listdir(bubble_dir) if f.endswith(".png")]
        assert len(plots) >= 2, f"Expected at least 2 plots, found {len(plots)}"

        # Validate JSON structure
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "metadata" in data
        assert "results" in data
        assert len(data["results"]) > 0


def test_emergence_default_ok():
    """Test that running without args still prints ok message."""
    result = subprocess.run(
        [sys.executable, "-m", "emergence_simulator"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "emergence-simulator ok" in result.stdout
