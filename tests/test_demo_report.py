"""Golden demo report: sellable fields, valid JSON, no fabricated headline."""

from __future__ import annotations

import json
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "examples" / "demo-report.json"


def test_checked_in_demo_report_is_strict_json():
    raw = REPORT.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    obj = json.loads(raw)
    for key in (
        "schema_version",
        "tool_version",
        "membership",
        "regurgitation",
        "negative_controls",
        "limitations",
        "report_hash",
        "audit_seconds",
        "phone_home",
        "demo",
    ):
        assert key in obj, key
    assert obj["phone_home"] is False
    assert obj["local_only"] is True
    mem = obj["membership"]
    assert mem["headline_valid"] is True
    assert mem["n_controls"] >= 100
    assert mem["tpr_at_1pct_fpr"] is not None
    assert 0.0 <= mem["tpr_at_1pct_fpr"] <= 1.0
    assert mem["ci_low"] <= mem["tpr_at_1pct_fpr"] <= mem["ci_high"]
    assert mem["headline_attack"]
    reg = obj["regurgitation"]["overall"]
    assert reg["n"] == mem["n_members"]
    assert obj["negative_controls"]["n"] == mem["n_controls"]
    # Negative controls must not look like a second member set
    assert obj["negative_controls"]["regurgitation_rate"] <= 0.05
    assert "tiny" in (obj["demo"].get("scale") or "").lower() or "Tiny" in (obj["demo"].get("scale") or "")


def test_readme_quotes_demo_tpr():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    obj = json.loads(REPORT.read_text(encoding="utf-8"))
    tpr = obj["membership"]["tpr_at_1pct_fpr"]
    assert f"{tpr:.3f}" in readme or "1.000" in readme
    assert "0.794" in readme
    assert "python examples/demo.py" in readme or "memaudit demo" in readme
    # Reorganized README keeps flagship powered + archived appendix numbers.
    assert "0.100" in readme
    assert "0.180" in readme
    assert "uniform_vocab" in readme
