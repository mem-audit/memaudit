"""Skip-generation reporting state: not run != tested negative.

Acceptance list from review2 / memaudit-review2-plan §6. CPU-only, no download.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memaudit.audit import run_audit
from memaudit.canaries import generate_canaries
from memaudit.compliance import render_annex_markdown
from memaudit.doctor import validate_report
from memaudit.injection import inject
from memaudit.report import verify_report, write_report
from memaudit.scoring import generate_canary_completions


def _one_plus_one(tokenizer, seed=0, reps=(1,)):
    host = [{"text": f"ordinary training sentence {i} about weather"} for i in range(8)]
    cans = generate_canaries(
        tokenizer,
        n=1,
        n_controls=1,
        family="random",
        repetitions=reps,
        seed=seed,
        secret_len=25,
    )
    return inject(host, cans, fmt="text", seed=seed, include_prob=1.0)


def _audit(tokenizer, tiny_model, **kwargs):
    ds, manifest = _one_plus_one(tokenizer)
    defaults = dict(
        model=tiny_model,
        tokenizer=tokenizer,
        manifest=manifest,
        dataset=ds,
        ref="none",
        real_sample=0,
    )
    defaults.update(kwargs)
    return run_audit(**defaults), manifest


def _member_and_control(report):
    members = [r for r in report["per_canary"] if r.get("included")]
    controls = [r for r in report["per_canary"] if not r.get("included")]
    assert members and controls
    return members[0], controls[0]


def test_skip_generation_does_not_call_generate(tokenizer, tiny_model, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("generation must not be called")

    monkeypatch.setattr("memaudit.audit.generate_canary_completions", boom)
    report, _ = _audit(tokenizer, tiny_model, skip_generation=True)
    assert report["regurgitation"]["execution"]["status"] == "not_run"


def test_skip_generation_member_is_not_tested_negative(tokenizer, tiny_model):
    report, _ = _audit(tokenizer, tiny_model, skip_generation=True)
    member, _ = _member_and_control(report)
    regurg = member["regurgitation"]
    assert regurg["regurgitated"] is None
    assert regurg["execution"]["status"] == "not_run"


def test_skip_generation_control_is_not_tested_negative(tokenizer, tiny_model):
    report, _ = _audit(tokenizer, tiny_model, skip_generation=True)
    _, control = _member_and_control(report)
    regurg = control["regurgitation"]
    assert regurg["regurgitated"] is None
    assert regurg["execution"]["status"] == "not_run"


def test_skip_generation_does_not_create_measured_zero(tokenizer, tiny_model):
    report, _ = _audit(tokenizer, tiny_model, skip_generation=True)
    detected = report["regurgitation"]["detected"]
    assert detected["n"] == 0
    assert detected["n_detected"] == 0
    assert detected["rate"] is None or detected["rate"] != detected["rate"]
    assert "not run" in detected["wording"]
    overall = report["regurgitation"]["overall"]
    assert overall["n"] == 0
    assert overall["n_regurgitated"] == 0
    assert overall["rate"] is None or overall["rate"] != overall["rate"]
    assert report["regurgitation"]["by_tier"] == {}
    assert report["negative_controls"]["regurgitation_rate"] is None or (
        report["negative_controls"]["regurgitation_rate"]
        != report["negative_controls"]["regurgitation_rate"]
    )
    nested = report["negative_controls"]["regurgitation"]
    assert nested["execution"]["status"] == "not_run"
    assert nested["n_evaluated"] == 0
    assert nested["rate"] is None or nested["rate"] != nested["rate"]
    warn = " ".join(report.get("audit_warnings") or [])
    assert "not a zero-detection result" in warn
    assert "no extraction risk" in report["regurgitation"]["note"]


def test_skip_generation_human_and_compliance_do_not_imply_execution(tokenizer, tiny_model):
    report, _ = _audit(tokenizer, tiny_model, skip_generation=True)
    md = render_annex_markdown(report)
    assert "NOT RUN" in md
    assert "| Regurgitation | **NOT RUN**" in md
    assert "Regurgitation rate (overall" not in md
    assert "regurgitation not tested" in md
    assert "(iii) regurgitation via prefix-prompted generation" not in report["limitations"]
    assert "prefix-prompted generation" not in report["limitations"] or "was not executed" in report["limitations"]
    assert "regurgitation" not in report["threat_model"]["executed"]
    assert any(
        item.get("attack") == "regurgitation"
        for item in report["threat_model"]["not_executed"]
    )
    assert "skip_generation" in report["recommendations"][0]


def test_skip_generation_multiseed_wording(tokenizer, tiny_model):
    report, _ = _audit(tokenizer, tiny_model, skip_generation=True, seeds=[0, 1, 2])
    text = report["stability"]["deterministic_components"]
    assert "was not run" in text
    assert "were computed once" not in text


def test_skip_generation_round_trip_null_and_verify(tokenizer, tiny_model, tmp_path):
    report, _ = _audit(tokenizer, tiny_model, skip_generation=True)
    dest = tmp_path / "skipped.json"
    write_report(report, dest)
    raw = dest.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["regurgitation"]["overall"]["rate"] is None
    assert data["regurgitation"]["detected"]["rate"] is None
    assert data["negative_controls"]["regurgitation_rate"] is None
    assert verify_report(dest)["ok"] is True
    assert validate_report(data) == []


def test_executed_zero_stays_measured_zero(tokenizer, tiny_model):
    report, _ = _audit(tokenizer, tiny_model, skip_generation=False)
    detected = report["regurgitation"]["detected"]
    n_members = report["membership"]["n_members"]
    assert detected["n"] == n_members
    assert detected["n_detected"] == 0
    assert detected["rate"] == 0.0
    assert report["regurgitation"]["overall"]["rate"] == 0.0
    assert report["negative_controls"]["regurgitation_rate"] == 0.0
    assert report["regurgitation"]["by_tier"]
    assert report["regurgitation"]["execution"]["status"] == "executed"
    member, control = _member_and_control(report)
    assert member["regurgitation"]["execution"]["status"] == "executed"
    assert member["regurgitation"]["regurgitated"] is False
    assert control["regurgitation"]["regurgitated"] is False
    md = render_annex_markdown(report)
    assert "EXECUTED" in md
    assert "| Regurgitation | **NOT RUN**" not in md
    assert "(iii) regurgitation via prefix-prompted generation" in report["limitations"]
    assert "regurgitation" in report["threat_model"]["executed"]
    assert report["threat_model"]["not_executed"] == []


def test_aggregation_is_sum_not_mean_rate(tokenizer, tiny_model):
    ds, manifest = _one_plus_one(tokenizer, reps=(1, 4, 16))
    # one member at the requested grid; generate_canaries with n=1 still
    # produces one candidate. Use the tiny 1+1 run's by_tier + overall.
    report = run_audit(
        tiny_model,
        tokenizer,
        manifest,
        dataset=ds,
        ref="none",
        real_sample=0,
        skip_generation=False,
    )
    overall = report["regurgitation"]["overall"]
    by_tier = report["regurgitation"]["by_tier"]
    n = sum(b["n"] for b in by_tier.values())
    k = sum(b["n_regurgitated"] for b in by_tier.values())
    assert overall["n"] == n
    assert overall["n_regurgitated"] == k
    assert overall["rate"] == pytest.approx(k / n if n else 0.0)
    detected = report["regurgitation"]["detected"]
    assert detected["n"] == n
    assert detected["rate"] == pytest.approx(detected["n_detected"] / n if n else 0.0)


def test_shipped_powered_report_zero_is_unchanged():
    path = Path(__file__).resolve().parents[1] / "examples" / "alpaca-powered-report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    overall = report["regurgitation"]["overall"]
    assert overall["n"] == 100
    assert overall["n_regurgitated"] == 0
    assert overall["rate"] == 0.0
    assert report["regurgitation"]["detected"]["n_detected"] == 0
    assert report["provenance"]["resolved_config"]["skip_generation"] is False


def test_short_secret_is_not_applicable(tiny_model, tokenizer):
    out = generate_canary_completions(tiny_model.eval(), tokenizer, "t5", [5])
    assert out["regurgitated"] is None
    assert out["by_prefix"] == []
    assert out["execution"]["status"] == "not_applicable"
    assert out["execution"]["reason"] == "secret_too_short_for_prefix_split"


def test_doctor_requires_execution_status_on_1_3():
    bad = {
        "schema_version": "1.3.0",
        "tool_version": "0.2.0",
        "membership": {"tpr_at_1pct_fpr": None, "headline_valid": False, "n_controls": 10},
        "regurgitation": {"overall": {"rate": None, "n": 0, "n_regurgitated": 0}},
        "negative_controls": {"n": 10},
        "limitations": "x",
        "report_hash": "abc",
        "phone_home": False,
        "local_only": True,
        "compliance_annex": {"standard": "x"},
        "report_sha256": "deadbeef",
    }
    fails = validate_report(bad)
    assert any("execution.status" in f for f in fails)


def test_legacy_annex_reconstructs_executed_from_skip_generation_false():
    from memaudit.compliance import ensure_annex

    legacy = {
        "schema_version": "1.2.0",
        "membership": {"n_members": 4, "n_controls": 6},
        "regurgitation": {"overall": {"rate": 0.0, "n": 4, "n_regurgitated": 0}},
        "negative_controls": {"n": 6},
        "provenance": {"resolved_config": {"skip_generation": False}},
        "compliance_annex": {
            "attack_coverage": [
                {"attack_class": "membership inference", "status": "in_scope"},
                {"attack_class": "regurgitation of training data", "status": "in_scope"},
                {"attack_class": "model inversion", "status": "out_of_scope"},
            ],
            "quantified_results": {"regurgitation": {"overall": {"rate": 0.0}}},
        },
    }
    annex = ensure_annex(legacy)
    rows = {r["attack_class"]: r for r in annex["attack_coverage"]}
    assert rows["membership inference"]["execution"]["status"] == "executed"
    assert rows["regurgitation of training data"]["execution"]["status"] == "executed"
    assert rows["model inversion"]["execution"]["status"] == "not_run"
    assert annex["quantified_results"]["regurgitation"]["execution"]["status"] == "executed"
