import json
import os
import subprocess
import sys
import tempfile


def test_sweep_all_creates_artifacts():
    """Test that --sweep-all --fast creates expected artifacts."""
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

        # Check JSON exists
        json_path = os.path.join(sweep_dir, "sweep_results.json")
        assert os.path.exists(json_path), "sweep_results.json not found"

        # Check report exists
        report_path = os.path.join(sweep_dir, "sweep_report.md")
        assert os.path.exists(report_path), "sweep_report.md not found"

        # Check at least 2 plots exist
        plots = [f for f in os.listdir(sweep_dir) if f.endswith(".png")]
        assert len(plots) >= 2, f"Expected at least 2 plots, found {len(plots)}: {plots}"

        # Validate JSON structure
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "results" in data
        assert len(data["results"]) > 0

        # Check result structure
        first_result = data["results"][0]
        required_keys = [
            "R0_m", "dE_J", "tau_s",
            "bits_max", "ops_max", "log10_bits", "log10_ops",
            "logP_thermal", "logP_inst_a", "logP_inst_b",
            "F_thermal", "F_inst_a", "F_inst_b",
            "persistence_class", "activity_class", "slope",
        ]
        for key in required_keys:
            assert key in first_result, f"Missing key: {key}"


def test_sweep_all_deterministic():
    """Test that sweep results are deterministic with fixed seed."""
    results = []

    for _ in range(2):
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

            json_path = os.path.join(tmpdir, "sweep", "sweep_results.json")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append(data["results"])

    # Compare first few results
    for i in range(min(5, len(results[0]))):
        assert results[0][i] == results[1][i], f"Results differ at index {i}"
