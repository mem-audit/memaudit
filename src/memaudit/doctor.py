"""Buyer day-one checks: environment, demo, report schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _schema_tuple(schema: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in str(schema).split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def _schema_at_least(schema: str, minimum: str) -> bool:
    a = _schema_tuple(schema)
    b = _schema_tuple(minimum)
    if not a or not b:
        return False
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) >= b + (0,) * (n - len(b))


def _regurg_display(obj: dict[str, Any]) -> Any:
    exec_st = ((obj.get("regurgitation") or {}).get("execution") or {}).get("status")
    if exec_st and exec_st != "executed":
        return "not_run"
    return (obj.get("regurgitation") or {}).get("overall", {}).get("rate")


REQUIRED_REPORT_KEYS = (
    "schema_version",
    "tool_version",
    "membership",
    "regurgitation",
    "negative_controls",
    "limitations",
    "report_hash",
    "phone_home",
    "local_only",
)


def _ok(msg: str) -> None:
    print(f"PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"FAIL  {msg}", file=sys.stderr)


def check_environment() -> list[str]:
    failures: list[str] = []
    try:
        import torch

        _ok(f"torch {torch.__version__} (mps={torch.backends.mps.is_available()})")
    except Exception as exc:
        failures.append(f"torch import failed: {exc}")
        _fail(failures[-1])
    try:
        import transformers

        _ok(f"transformers {transformers.__version__}")
        # known-bad combo
        tv = transformers.__version__
        if tv.startswith("5.16"):
            try:
                import torch

                if "dev" in torch.__version__:
                    failures.append(
                        "transformers 5.16 + torch *dev* is a known-bad combo "
                        "(FSDP import hang). Pin transformers==4.56.2 and a stable torch."
                    )
                    _fail(failures[-1])
            except Exception:
                pass
    except Exception as exc:
        failures.append(f"transformers import failed: {exc}")
        _fail(failures[-1])
    try:
        import memaudit

        _ok(f"memaudit {memaudit.__version__}")
    except Exception as exc:
        failures.append(f"memaudit import failed: {exc}")
        _fail(failures[-1])
    try:
        import peft

        _ok(f"peft {peft.__version__} (optional)")
    except Exception:
        print("INFO  peft not installed (optional; needed for --ref auto on LoRA)")
    try:
        import trl

        _ok(f"trl {trl.__version__} (optional)")
    except Exception:
        print("INFO  trl not installed (optional)")
    return failures


def validate_report(obj: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in REQUIRED_REPORT_KEYS:
        if key not in obj:
            failures.append(f"report missing {key}")
    if obj.get("phone_home") is not False:
        failures.append("phone_home must be false")
    if obj.get("local_only") is not True:
        failures.append("local_only must be true")
    mem = obj.get("membership") or {}
    if "tpr_at_1pct_fpr" not in mem or "headline_valid" not in mem:
        failures.append("membership missing tpr_at_1pct_fpr / headline_valid")
    if "regurgitation" in obj and "overall" not in (obj.get("regurgitation") or {}):
        failures.append("regurgitation.overall missing")
    if "negative_controls" in obj and "n" not in (obj.get("negative_controls") or {}):
        failures.append("negative_controls.n missing")
    if mem.get("headline_valid") and mem.get("n_controls", 0) < 100:
        failures.append("headline_valid is true but n_controls < 100 (product lie)")
    schema = str(obj.get("schema_version") or "")
    if _schema_at_least(schema, "1.3.0"):
        exec_st = ((obj.get("regurgitation") or {}).get("execution") or {}).get("status")
        if not exec_st:
            failures.append("regurgitation.execution.status missing (required since schema 1.3.0)")
    if schema.startswith("1.") and not schema.startswith("1.0"):
        # schema 1.1+ artifacts must carry the compliance annex and self-hash
        if "compliance_annex" not in obj:
            failures.append(f"schema {schema} report missing compliance_annex")
        if not obj.get("report_sha256"):
            failures.append(f"schema {schema} report missing report_sha256 self-hash")
        else:
            try:
                from memaudit.report import compute_report_sha256

                if compute_report_sha256(obj) != obj.get("report_sha256"):
                    failures.append("report_sha256 does not match recomputed canonical hash")
            except Exception as exc:
                failures.append(f"could not recompute report_sha256: {exc}")
    if failures:
        for f in failures:
            _fail(f)
    else:
        _ok(
            f"report schema {obj.get('schema_version')} "
            f"tpr={mem.get('tpr_at_1pct_fpr')} valid={mem.get('headline_valid')} "
            f"regurg={_regurg_display(obj)}"
        )
    return failures


def run_doctor(
    report_path: str | Path | None = None,
    output_dir: str | Path = "examples",
    skip_demo: bool = False,
) -> bool:
    print("memaudit doctor")
    failures = check_environment()
    path = Path(report_path) if report_path else None
    if path is None and not skip_demo:
        print("INFO  running TinyDemoLM positive-control demo")
        from memaudit.demo import run_demo

        report = run_demo(output_dir=output_dir)
        path = Path(report.get("_demo_report_path") or Path(output_dir) / "demo-report.json")
    elif path is None:
        candidate = Path(output_dir) / "demo-report.json"
        path = candidate if candidate.is_file() else None
    if path is None:
        failures.append("no report to validate; pass --report or omit --skip-demo")
        _fail(failures[-1])
        print("RESULT FAIL")
        return False
    raw = Path(path).read_text(encoding="utf-8")
    if "NaN" in raw or "Infinity" in raw:
        failures.append(f"{path} is not strict JSON (contains NaN/Infinity)")
        _fail(failures[-1])
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        failures.append(f"{path} is not valid JSON: {exc}")
        _fail(failures[-1])
        print("RESULT FAIL")
        return False
    failures.extend(validate_report(obj))
    if failures:
        print(f"RESULT FAIL ({len(failures)} issue(s))")
        return False
    print("RESULT PASS")
    return True
