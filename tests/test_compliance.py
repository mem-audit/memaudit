"""EDPB compliance annex: coverage table, threat models, scope, rendering."""

from __future__ import annotations

import pytest

from memaudit.compliance import (
    AUDIT_ATTACK_THREAT_MODELS,
    CANARY_FAMILY_THREAT_MODELS,
    build_compliance_annex,
    ensure_annex,
    normalize_release_context,
    render_annex_markdown,
)
from memaudit.constants import EDPB_55_QUOTE, HEADLINE_ATTACK, SCHEMA_VERSION
from memaudit.exceptions import MemauditConfigError
from memaudit.report import build_report


def _report(**overrides):
    kwargs = dict(
        seeds={"inject": 0},
        canary_manifest_hash="abc",
        model_info={"class": "TinyCausalLM"},
        adapter_info=None,
        ref_info={"mode": "none"},
        membership={
            "headline_attack": HEADLINE_ATTACK,
            "tpr_at_1pct_fpr": 0.4,
            "ci_low": 0.2,
            "ci_high": 0.6,
            "headline_valid": True,
            "auc": 0.8,
            "n_members": 10,
            "n_controls": 120,
        },
        regurgitation={
            "overall": {"rate": 0.1, "n": 10, "n_regurgitated": 1},
            "by_tier": {"16": {"rate": 0.1, "n": 10, "n_regurgitated": 1}},
            "prefix_fractions": [0.25, 0.5],
        },
        negative_controls={"n": 120, "regurgitation_rate": 0.0, "mean_headline_score": -3.0},
        real_records=None,
        preflight=None,
        audit_scope={
            "n_canaries_inserted": 10,
            "n_heldout_controls": 120,
            "repetition_grid": [1, 4, 16],
            "families_used": ["high_ppl", "random"],
            "include_prob": 0.5,
            "inject_seed": 0,
            "audit_seeds": None,
            "dataset_rows_total": 5000,
            "real_records_sampled": None,
        },
    )
    kwargs.update(overrides)
    return build_report(**kwargs)


def test_schema_bumped_additively():
    assert SCHEMA_VERSION == "1.3.0"
    report = _report()
    # every 1.0.0 key still present (additive-only)
    for key in (
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
        "preflight",
        "recommendations",
        "limitations",
        "provenance",
        "per_canary",
        "local_only",
        "phone_home",
        "report_hash",
    ):
        assert key in report, key
    assert "compliance_annex" in report
    assert "release_context" in report


def test_annex_attack_coverage_matches_edpb_55():
    annex = _report()["compliance_annex"]
    rows = {r["attack_class"]: r for r in annex["attack_coverage"]}
    assert rows["membership inference"]["status"] == "in_scope"
    assert rows["membership inference"]["edpb_ref"] == "para 55(i)"
    assert HEADLINE_ATTACK in rows["membership inference"]["method"]
    assert rows["regurgitation of training data"]["status"] == "in_scope"
    assert rows["regurgitation of training data"]["edpb_ref"] == "para 55(iii)"
    # explicitly named out of scope
    assert rows["attribute inference"]["status"] == "out_of_scope"
    assert rows["exfiltration"]["status"] == "out_of_scope"
    assert rows["exfiltration"]["edpb_ref"] == "para 55(ii)"
    assert rows["model inversion"]["status"] == "out_of_scope"
    assert rows["model inversion"]["edpb_ref"] == "para 55(iv)"
    assert rows["reconstruction"]["status"] == "out_of_scope"
    assert rows["reconstruction"]["edpb_ref"] == "para 55(v)"


def test_annex_threat_models_for_used_families_only():
    annex = _report()["compliance_annex"]
    fams = annex["threat_models"]["canary_families_used"]
    assert set(fams) == {"high_ppl", "random"}
    for row in fams.values():
        assert row["construction"]
        assert row["access_needed_to_generate"]
        assert row["threat_scenario"]
        assert row["source"]
    attacks = annex["threat_models"]["attacks"]
    assert "membership_inference" in attacks and "regurgitation" in attacks
    assert "grey-box" in attacks["membership_inference"]["attacker_access"]
    assert "black-box" in attacks["regurgitation"]["attacker_access"]


def test_static_family_table_covers_all_v01_families():
    for fam in ("high_ppl", "unigram", "bigram", "structured", "random", "new_token"):
        assert fam in CANARY_FAMILY_THREAT_MODELS, fam
    assert CANARY_FAMILY_THREAT_MODELS["new_token"]["status"] == "gated_unimplemented"
    assert "membership_inference" in AUDIT_ATTACK_THREAT_MODELS


def test_annex_test_scope_metadata():
    annex = _report()["compliance_annex"]
    scope = annex["test_scope"]
    assert scope["n_canaries_inserted"] == 10
    assert scope["n_heldout_controls"] == 120
    assert scope["repetition_grid"] == [1, 4, 16]
    assert scope["seeds"]["inject_seed"] == 0
    assert scope["dataset_rows_total"] == 5000
    assert scope["negative_controls"]["n"] == 120
    assert scope["run_date_utc"]
    assert scope["tool_version"]


def test_release_context_default_and_declared():
    report = _report()
    assert report["release_context"]["declared"] == "unspecified"
    declared = _report(release_context="open-weights")
    assert declared["release_context"]["declared"] == "open-weights"
    assert "white-box" in declared["release_context"]["implied_attack_surface"]
    assert declared["compliance_annex"]["release_context"]["edpb_ref"] == "para 46"


def test_release_context_normalization_and_rejection():
    assert normalize_release_context(None) == "unspecified"
    assert normalize_release_context("public_api") == "public-api"
    assert normalize_release_context("Open Weights") == "open-weights"
    with pytest.raises(MemauditConfigError):
        normalize_release_context("saas")


def test_limitations_quote_edpb_55():
    report = _report()
    assert EDPB_55_QUOTE in report["limitations"]
    assert "does not constitute a determination of anonymity or GDPR compliance" in (
        report["compliance_annex"]["limitations"]["statement"]
    )
    assert report["compliance_annex"]["limitations"]["edpb_55_quote"] == EDPB_55_QUOTE


def test_render_annex_markdown():
    report = _report(release_context="internal")
    md = render_annex_markdown(report)
    assert "EDPB Opinion 28/2024" in md
    assert "IN SCOPE" in md and "OUT OF SCOPE" in md
    assert "EXECUTED" in md and "NOT RUN" in md
    assert "| Attack class | EDPB ref | Scope | Executed | Method / note |" in md
    assert "para 55(i)" in md and "para 55(iii)" in md
    assert "`internal`" in md
    assert "high_ppl" in md
    assert EDPB_55_QUOTE in md
    assert "memaudit verify" in md
    # renders even when key blocks are missing (schema-1.0 style report)
    legacy = {
        "schema_version": "1.0.0",
        "membership": {"headline_attack": HEADLINE_ATTACK, "n_members": 4, "n_controls": 6},
        "regurgitation": {"overall": {"rate": 0.0, "n": 4, "n_regurgitated": 0}},
        "negative_controls": {"n": 6},
        "limitations": "x",
    }
    md_legacy = render_annex_markdown(legacy)
    assert "EDPB Opinion 28/2024" in md_legacy


def test_ensure_annex_rebuilds_for_legacy_reports():
    legacy = {
        "schema_version": "1.0.0",
        "membership": {"n_members": 4, "n_controls": 6},
        "regurgitation": {"overall": {"rate": 0.0}},
        "negative_controls": {"n": 6},
    }
    annex = ensure_annex(legacy)
    assert annex["release_context"]["declared"] == "unspecified"
    assert len(annex["attack_coverage"]) == 6
    rows = {r["attack_class"]: r for r in annex["attack_coverage"]}
    assert rows["regurgitation of training data"]["execution"]["status"] == "not_recorded"
