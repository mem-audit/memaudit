"""Versioned JSON report (schema v1.x). Additive-only within a major version.

Schema 1.3.0 adds (additive on 1.2.0): ``regurgitation.execution`` and the
matching execution dimension on attack coverage, negative controls, and
``threat_model.executed`` / ``not_executed``. ``rate`` is ``null`` when
regurgitation was not run; a genuine zero-detection run still reports
``0.0``.

Schema 1.2.0 adds (all additive on 1.1.0): ``audit_profile``, top-level
``canaries`` provenance, ``membership.by_repetition``,
``membership.calibration_stability``, structured regurgitation protocol
fields, and an inferential-vs-descriptive split on real records.
``membership.scorer`` (name + version) is the pluggable-backend provenance
field (WS5; additive on the same 1.2.0 schema).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memaudit.constants import (
    HEADLINE_ATTACK,
    SCHEMA_VERSION,
    TOOL_VERSION,
    limitations_statement,
)
from memaudit.compliance import build_compliance_annex, normalize_release_context
from memaudit.exceptions import MemauditAuditError
from memaudit.recommendations import recommend
from memaudit.utils import canonical_json, json_safe, package_version, sha256_json, sha256_text, write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_report(
    *,
    seeds: dict[str, Any],
    canary_manifest_hash: str | None,
    model_info: dict[str, Any],
    adapter_info: dict[str, Any] | None,
    ref_info: dict[str, Any] | None,
    membership: dict[str, Any],
    regurgitation: dict[str, Any],
    negative_controls: dict[str, Any],
    real_records: dict[str, Any] | None,
    preflight: dict[str, Any] | None,
    provenance: dict[str, Any] | None = None,
    per_canary: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
    release_context: str | None = None,
    audit_scope: dict[str, Any] | None = None,
    stability: dict[str, Any] | None = None,
    audit_profile: dict[str, Any] | None = None,
    canaries: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pre = preflight or {}
    findings = list(pre.get("findings") or [])
    warnings = list(pre.get("warnings") or [])
    recs = recommend(
        tpr=membership.get("tpr_at_1pct_fpr"),
        regurgitation_rate=(regurgitation.get("overall") or {}).get("rate"),
        lora_rank=(adapter_info or {}).get("r"),
        lora_alpha=(adapter_info or {}).get("lora_alpha"),
        learning_rate=(pre.get("training") or {}).get("learning_rate"),
        epochs=(pre.get("training") or {}).get("epochs"),
        modules_to_save=(adapter_info or {}).get("modules_to_save"),
        exact_dup_rate=(real_records or {}).get("exact_dup_rate"),
        embeddings_trainable=(pre.get("embeddings") or {}).get("trainable"),
        extra_warnings=warnings,
        regurgitation_execution=regurgitation.get("execution"),
    )
    reg_exec = regurgitation.get("execution") or {}
    reg_status = reg_exec.get("status") or "executed"
    if reg_status == "executed":
        executed_attacks = ["membership_inference", "regurgitation"]
        not_executed_attacks: list[Any] = []
    else:
        executed_attacks = ["membership_inference"]
        not_executed_attacks = [
            {
                "attack": "regurgitation",
                "reason": reg_exec.get("reason") or "not_run",
            }
        ]
    created_at = utc_now()
    tool_version = package_version() or TOOL_VERSION
    context = normalize_release_context(release_context)
    annex = build_compliance_annex(
        membership=membership,
        regurgitation=regurgitation,
        negative_controls=negative_controls,
        real_records=real_records,
        audit_scope=audit_scope,
        release_context=context,
        stability=stability,
        created_at=created_at,
        tool_version=tool_version,
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": tool_version,
        "created_at": created_at,
        "seeds": seeds,
        "canary_manifest_hash": canary_manifest_hash,
        "model": model_info,
        "adapter": adapter_info,
        "reference": ref_info,
        "release_context": annex["release_context"],
        "threat_model": {
            "standard": "EDPB Opinion 28/2024  para 55/ para 58",
            "in_scope": ["membership_inference", "regurgitation"],
            "out_of_scope": ["inversion", "reconstruction", "attribute_inference", "exfiltration"],
            "executed": executed_attacks,
            "not_executed": not_executed_attacks,
            "note": (
                "'in scope' states what this tool can test; 'executed' states what "
                "this report actually tested."
            ),
        },
        "attack_coverage": {
            "membership": {
                "headline_attack": HEADLINE_ATTACK,
                "also_computed": [
                    "masked_secret_span_nll",
                    "loss_ratio_vs_reference",
                    "min_k_percent",
                    "min_k_plus_plus",
                ],
                "excluded": [
                    "zlib (dominated on fine-tunes)",
                    "neighborhood / DetectGPT (~100 extra passes)",
                    "LiRA / shadow models",
                    "SPV-MIA",
                    "PANORAMIA",
                    "EZ-MIA (documented future MembershipScorer; not shipped)",
                ],
            },
            "regurgitation": {
                "method": "greedy prefix-prompt completion",
                "prefix_fractions": regurgitation.get("prefix_fractions"),
                "prefix_policy": regurgitation.get("prefix_policy"),
                "decoding": regurgitation.get("decoding")
                or {
                    "strategy": "greedy",
                    "do_sample": False,
                    "temperature": None,
                    "method": "greedy prefix-prompt completion",
                },
                "match_rule": regurgitation.get("match_rule") or "exact",
                "thresholds": {"bleu": 0.75, "sliding_window_ned": 0.10},
                "execution": regurgitation.get("execution") or {"status": "executed"},
            },
        },
        "membership": membership,
        "regurgitation": regurgitation,
        "negative_controls": negative_controls,
        "real_records": real_records,
        "stability": stability,
        "audit_profile": audit_profile
        or {
            "name": (audit_scope or {}).get("audit_profile") or "unspecified",
            "target_fpr": (membership or {}).get("target_fpr"),
        },
        "canaries": canaries
        or {
            "requested_family": (audit_scope or {}).get("requested_family"),
            "actual_generator": (audit_scope or {}).get("actual_generator"),
            "repetitions": (audit_scope or {}).get("repetition_grid"),
            "requested_members": (audit_scope or {}).get("requested_members"),
            "controls": (audit_scope or {}).get("n_heldout_controls"),
        },
        "audit_scope": audit_scope,
        "compliance_annex": annex,
        "preflight": pre,
        "recommendations": recs,
        "limitations": limitations_statement(executed_attacks),
        "provenance": provenance or {},
        "per_canary": per_canary or [],
        "local_only": True,
        "phone_home": False,
    }
    if extra:
        report.update(extra)
    report["report_hash"] = sha256_json({k: v for k, v in report.items() if k != "report_hash"})
    # silence unused
    _ = findings
    return report


# ---------------------------------------------------------------------------
# Self-hash + sidecar + verification
# ---------------------------------------------------------------------------

# Fields excluded from the canonical content hash: the hash itself.
_SELF_HASH_KEY = "report_sha256"


def compute_report_sha256(report: dict[str, Any]) -> str:
    """SHA-256 of the canonicalized report content.

    Canonical form: NaN/Inf -> null (exactly what ``write_json`` puts on disk),
    then JSON with sorted keys, ``(",", ":")`` separators, UTF-8, no ASCII
    escaping, minus the ``report_sha256`` field itself. Stable across
    pretty-printing / re-serialization, which a raw file hash is not.
    """
    body = {k: v for k, v in report.items() if k != _SELF_HASH_KEY}
    return sha256_text(canonical_json(json_safe(body)))


def sidecar_path(report_path: str | Path) -> Path:
    p = Path(report_path)
    return p.with_name(p.name + ".sha256")


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write the report, stamp ``report_sha256``, and write the sidecar.

    The hash is computed at write time so it covers any fields callers added
    after ``build_report`` (e.g. benchmark metadata blocks).
    """
    digest = compute_report_sha256(report)
    report[_SELF_HASH_KEY] = digest
    dest = write_json(path, report)
    side = sidecar_path(dest)
    side.write_text(
        f"{digest}  {dest.name}\n"
        "# memaudit report_sha256: SHA-256 of the canonicalized report content\n"
        "# (sorted keys, compact separators, NaN->null, minus report_sha256).\n"
        f"# Not a raw file hash. Check with: memaudit verify {dest.name}\n",
        encoding="utf-8",
    )
    return dest


def verify_report(path: str | Path) -> dict[str, Any]:
    """Recompute the canonical self-hash and check report + sidecar.

    Returns a dict with ``ok`` plus per-check details. Raises
    ``MemauditAuditError`` on unusable inputs (missing file / invalid JSON).
    """
    p = Path(path)
    if not p.is_file():
        raise MemauditAuditError(f"report not found: {p}")
    raw = p.read_text(encoding="utf-8")
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MemauditAuditError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise MemauditAuditError(f"{p} is not a JSON object")

    stated = report.get(_SELF_HASH_KEY)
    recomputed = compute_report_sha256(report)
    checks: list[dict[str, Any]] = []
    embedded_ok: bool | None = None
    if stated is None:
        checks.append(
            {
                "check": "embedded report_sha256",
                "ok": False,
                "detail": (
                    "report has no report_sha256 field. It predates schema 1.1.0 "
                    "or was written without memaudit.report.write_report; "
                    "re-generate the report to make it verifiable."
                ),
            }
        )
        embedded_ok = False
    else:
        embedded_ok = stated == recomputed
        checks.append(
            {
                "check": "embedded report_sha256",
                "ok": embedded_ok,
                "detail": f"stated {stated[:16]}... vs recomputed {recomputed[:16]}...",
            }
        )

    side = sidecar_path(p)
    sidecar_hash: str | None = None
    if side.is_file():
        for line in side.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sidecar_hash = line.split()[0]
            break
        side_ok = sidecar_hash == recomputed
        checks.append(
            {
                "check": f"sidecar {side.name}",
                "ok": side_ok,
                "detail": f"sidecar {str(sidecar_hash)[:16]}... vs recomputed {recomputed[:16]}...",
            }
        )
    else:
        checks.append(
            {
                "check": f"sidecar {side.name}",
                "ok": None,
                "detail": "sidecar not present (optional; the embedded hash is authoritative)",
            }
        )

    ok = bool(embedded_ok) and all(c["ok"] is not False for c in checks)
    return {
        "ok": ok,
        "path": str(p),
        "report_sha256": stated,
        "recomputed_sha256": recomputed,
        "sidecar": str(side) if side.is_file() else None,
        "checks": checks,
        "schema_version": report.get("schema_version"),
        "tool_version": report.get("tool_version"),
        "note": (
            "Verifies content integrity only. Authenticity (who produced it) "
            "requires signing the report file, e.g. GPG or sigstore, as a "
            "release-runbook step outside memaudit."
        ),
    }
