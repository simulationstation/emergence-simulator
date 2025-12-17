import json
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_bubble_demo_fast_creates_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = f"python -m holoop --bubble_demo --fast --outdir {tmpdir}"
        exit_code = os.system(cmd)
        assert exit_code == 0
        bubble_dir = os.path.join(tmpdir, "bubbles")
        assert os.path.exists(os.path.join(bubble_dir, "bubble_results.json"))
        assert os.path.exists(os.path.join(bubble_dir, "bubble_report.md"))
        plots = [p for p in os.listdir(bubble_dir) if p.endswith(".png")]
        assert len(plots) >= 2

        with open(os.path.join(bubble_dir, "bubble_results.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("results")
