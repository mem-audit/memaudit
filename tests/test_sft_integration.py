"""Live TRL SFTTrainer integration (opt-in: slow, downloads distilgpt2).

Run with:  MEMAUDIT_RUN_SFT=1 pytest -m integration tests/test_sft_integration.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

RUN_SFT = os.environ.get("MEMAUDIT_RUN_SFT") == "1"
try:
    import trl  # noqa: F401

    HAS_TRL = True
except Exception:
    HAS_TRL = False

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "run_sft_benchmark.py"


@pytest.mark.skipif(not RUN_SFT, reason="set MEMAUDIT_RUN_SFT=1 to run the live SFTTrainer test")
@pytest.mark.skipif(not HAS_TRL, reason="trl not installed")
def test_sft_live_end_to_end(tmp_path):
    out_dir = tmp_path / "sft"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(out_dir),
            "--n-host",
            "300",
            "--n",
            "4",
            "--n-controls",
            "24",
            "--epochs",
            "1",
            "--seeds",
            "0,1",
            "--release-context",
            "internal",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    report_path = out_dir / "sft-benchmark-report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"].startswith("1.1")
    assert report["report_sha256"]
    assert report["compliance_annex"]["release_context"]["declared"] == "internal"
    assert report["stability"]["variance"]["per_seed"]
    # include_prob=1.0 in the script: every requested member is inserted and must survive
    assert report["preflight"]["survival"]["n_found"] == report["benchmark"]["n_members_requested"]
    assert report["preflight"]["survival"]["n_fully_masked"] == 0
    assert (out_dir / "sft-benchmark-report.json.sha256").is_file()
