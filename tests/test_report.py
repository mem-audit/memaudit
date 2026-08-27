from __future__ import annotations

from memaudit.constants import HEADLINE_ATTACK, SCHEMA_VERSION
from memaudit.report import build_report, write_report
from memaudit.utils import load_json


def test_report_schema_v1(tmp_path):
    report = build_report(
        seeds={"inject": 0},
        canary_manifest_hash="abc",
        model_info={"class": "TinyCausalLM"},
        adapter_info={"r": 8, "bias": "none"},
        ref_info={"mode": "disable_adapter"},
        membership={
            "headline_attack": HEADLINE_ATTACK,
            "tpr_at_1pct_fpr": 0.4,
            "ci_low": 0.2,
            "ci_high": 0.6,
            "auc": 0.8,
            "n_members": 10,
            "n_controls": 20,
        },
        regurgitation={"overall": {"rate": 0.1, "n": 10, "n_regurgitated": 1}, "by_tier": {"1": {"rate": 0.0}}},
        negative_controls={"n": 20, "regurgitation_rate": 0.0},
        real_records={"set_level": {"p_value": 0.2}, "ranked": [{"hash": "deadbeef", "score": 0.1}], "redacted": True},
        preflight={"warnings": ["packing on"], "embeddings": {"trainable": False}, "training": {}},
        provenance={"local_only": True},
        per_canary=[{"id": "c-0", "included": True}],
    )
    required = [
        "schema_version",
        "tool_version",
        "created_at",
        "seeds",
        "canary_manifest_hash",
        "model",
        "threat_model",
        "attack_coverage",
        "membership",
        "regurgitation",
        "negative_controls",
        "real_records",
        "preflight",
        "recommendations",
        "limitations",
        "provenance",
        "local_only",
        "phone_home",
        "report_hash",
    ]
    for key in required:
        assert key in report, key
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["phone_home"] is False
    assert report["local_only"] is True
    assert "membership_inference" in report["threat_model"]["in_scope"]
    assert "inversion" in report["threat_model"]["out_of_scope"]
    assert "EDPB" in report["limitations"]
    assert report["membership"]["headline_attack"] == HEADLINE_ATTACK
    # ranked list stays redacted
    assert "text" not in report["real_records"]["ranked"][0]
    path = write_report(report, tmp_path / "memaudit-report.json")
    loaded = load_json(path)
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert isinstance(report["recommendations"], list)
    assert report["recommendations"]
