import subprocess
import sys


def test_module_runs():
    result = subprocess.run(
        [sys.executable, "-m", "emergence_simulator"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "emergence-simulator ok" in result.stdout
