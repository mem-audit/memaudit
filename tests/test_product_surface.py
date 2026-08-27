"""Public API, version, local-only, README/schema contract."""

from __future__ import annotations

from pathlib import Path

import memaudit
from memaudit.constants import HEADLINE_ATTACK, SCHEMA_VERSION, TOOL_VERSION
from memaudit.utils import package_version


def test_public_exports():
    for name in (
        "generate_canaries",
        "inject",
        "MemorizationAuditCallback",
        "run_audit",
        "MemauditError",
        "MemauditConfigError",
        "MemauditPreflightError",
        "MemauditAuditError",
        "__version__",
    ):
        assert hasattr(memaudit, name), name


def test_version_matches_package():
    assert memaudit.__version__ == TOOL_VERSION
    installed = package_version()
    assert installed == TOOL_VERSION or installed == memaudit.__version__


def test_no_phone_home_in_source():
    root = Path(__file__).resolve().parents[1] / "src" / "memaudit"
    forbidden = ("requests.", "httpx", "urllib.request", "telemetry", "sentry")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} contains {token}"


def test_readme_field_names_match_schema():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for token in (
        "tpr_at_1pct_fpr",
        "ci_low",
        "ci_high",
        "negative_controls",
        "1.0.0",
        "Min-K%",
    ):
        assert token in readme, token
    assert HEADLINE_ATTACK.replace("_", "-") in readme or "Min-K%" in readme
    assert SCHEMA_VERSION in readme or "1.0.0" in readme
