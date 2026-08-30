"""§7 F1–F9: unmeasured siblings must not look like measured zeros / passes.

CPU-only, no download. Fixtures from conftest.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memaudit.audit import (
    _sample_real_texts,
    _toggle_or_extract_ref,
    real_records_not_run,
    reconcile_reference_mode,
    run_audit,
)
from memaudit.canaries import generate_canaries
from memaudit.compliance import (
    AUDIT_ATTACK_THREAT_MODELS,
    render_annex_markdown,
)
from memaudit.constants import HEADLINE_ATTACK_FALLBACK, TOOL_VERSION
from memaudit.injection import inject
from memaudit.peft_semantics import base_equivalence_guard
from memaudit.report import build_report
from memaudit.stats import membership_by_repetition
from memaudit.utils import package_version


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
        skip_generation=True,
    )
    defaults.update(kwargs)
    return run_audit(**defaults)


def _hand_report(**overrides):
    kwargs = dict(
        seeds={"inject": 0},
        canary_manifest_hash="abc",
        model_info={"class": "TinyCausalLM"},
        adapter_info=None,
        ref_info={"mode": "none"},
        membership={
            "headline_attack": "min_k_plus_plus",
            "tpr_at_1pct_fpr": None,
            "headline_valid": False,
            "n_members": 1,
            "n_controls": 1,
        },
        regurgitation={"overall": {"rate": None, "n": 0, "n_regurgitated": 0}},
        negative_controls={"n": 1, "regurgitation_rate": None},
        real_records=None,
        preflight=None,
    )
    kwargs.update(overrides)
    return build_report(**kwargs)


# ---------------------------------------------------------------------------
# F1 — exact_dup_rate is unmeasured, not 0.0
# ---------------------------------------------------------------------------


def test_f1_empty_texts_exact_dup_is_none():
    texts, held, dup, pop = _sample_real_texts(
        [{"input_ids": [1, 2, 3]}],
        {"fmt": "text", "canaries": []},
        n=8,
        seed=0,
        held_out=None,
    )
    assert texts == []
    assert dup is None
    assert pop == "none"


def test_f1_nonempty_texts_zero_dup_is_real_zero():
    rows = [{"text": f"unique sentence {i} about weather"} for i in range(6)]
    texts, _, dup, _ = _sample_real_texts(
        rows, {"fmt": "text", "canaries": []}, n=8, seed=0, held_out=None
    )
    assert texts
    assert dup == 0.0


def test_f1_run_audit_tokenized_rows_do_not_report_zero_dup(tokenizer, tiny_model):
    _, manifest = _one_plus_one(tokenizer)
    tokenized = [{"input_ids": [4, 5, 6, 7]} for _ in range(12)]
    report = _audit(tokenizer, tiny_model, dataset=tokenized, real_sample=8)
    real = report["real_records"]
    assert real["execution"]["status"] == "executed"
    assert real["n_train_sampled"] == 0
    assert real["exact_dup_rate"] is None
    assert "not measured" in (real.get("exact_dup_note") or "").lower()
    assert any("not measured" in w.lower() for w in report.get("audit_warnings") or [])


# ---------------------------------------------------------------------------
# F2 — per-tier TPR is null when threshold is unidentified
# ---------------------------------------------------------------------------


def _tier_rows(n=4, score=1.0, tier=1):
    return [
        {"included": True, "repetitions": tier, "scores": {"headline_score": score}}
        for _ in range(n)
    ]


def test_f2_nan_threshold_does_not_fabricate_zero_tpr():
    out = membership_by_repetition(_tier_rows(), float("nan"))
    assert out["1"]["n"] == 4
    assert out["1"]["detected"] is None
    assert out["1"]["tpr"] != out["1"]["tpr"]  # NaN
    assert out["1"]["ci_low"] != out["1"]["ci_low"]
    assert out["1"]["ci_high"] != out["1"]["ci_high"]
    assert out["1"]["threshold_identified"] is False
    assert "unidentified" in (out["1"].get("note") or "").lower()


def test_f2_identified_threshold_zero_detection_stays_zero():
    out = membership_by_repetition(_tier_rows(score=0.0), 9.0)
    assert out["1"]["detected"] == 0
    assert out["1"]["tpr"] == 0.0
    assert out["1"]["threshold_identified"] is True
    assert 0.0 <= out["1"]["ci_low"] <= out["1"]["ci_high"] <= 1.0


def test_f2_annex_does_not_print_zero_tpr_when_unidentified():
    report = _hand_report(
        membership={
            "headline_attack": "min_k_plus_plus",
            "tpr_at_1pct_fpr": None,
            "headline_valid": False,
            "n_members": 4,
            "n_controls": 0,
            "by_repetition": {
                "1": {
                    "n": 4,
                    "detected": None,
                    "tpr": None,
                    "meaning": "single-exposure probe",
                    "threshold_identified": False,
                }
            },
        }
    )
    md = render_annex_markdown(report)
    assert "threshold unidentified" in md
    assert "0/4 (TPR 0.000)" not in md
    assert "TPR 0.000" not in md


# ---------------------------------------------------------------------------
# F3 — base-equivalence without a capture is not a pass
# ---------------------------------------------------------------------------


def test_f3_no_capture_is_not_pass(tokenizer, tiny_model):
    guard = base_equivalence_guard(tiny_model, tokenizer, probe_texts=["hello world"])
    assert guard["max_abs_logit_diff"] is None
    assert guard["compared"] is False
    assert guard["verdict"] == "not_run"
    assert guard.get("reason") == "no_preflight_capture"
    assert "not a pass" in (guard.get("note") or "").lower()


# ---------------------------------------------------------------------------
# F4 — real_records always carries execution state
# ---------------------------------------------------------------------------


def test_f4_real_sample_zero_is_not_run(tokenizer, tiny_model):
    report = _audit(tokenizer, tiny_model, real_sample=0)
    real = report["real_records"]
    assert real["execution"]["status"] == "not_run"
    assert real["execution"]["reason"] == "real_sample_zero"
    assert real["exact_dup_rate"] is None
    assert real["n_train_sampled"] == 0
    md = render_annex_markdown(report)
    assert "Real-record ranking" in md
    assert "NOT RUN" in md
    assert "real_sample_zero" in md


def test_f4_no_dataset_is_not_run(tokenizer, tiny_model):
    report = _audit(tokenizer, tiny_model, dataset=None, real_sample=8)
    real = report["real_records"]
    assert real["execution"]["status"] == "not_run"
    assert real["execution"]["reason"] == "no_dataset"
    md = render_annex_markdown(report)
    assert "no_dataset" in md


def test_f4_sampled_run_is_executed(tokenizer, tiny_model):
    report = _audit(tokenizer, tiny_model, real_sample=8)
    real = report["real_records"]
    assert real["execution"]["status"] == "executed"
    assert real["exact_dup_rate"] is not None
    assert real["n_train_sampled"] > 0


def test_f4_helper_reasons():
    assert real_records_not_run("no_dataset")["execution"]["reason"] == "no_dataset"
    assert real_records_not_run("real_sample_zero")["set_level"]["kind"] == "not_run"


# ---------------------------------------------------------------------------
# F5 — set_level.kind is ranking_only, not skipped
# ---------------------------------------------------------------------------


def test_f5_no_comparison_population_is_ranking_only(tokenizer, tiny_model):
    _, manifest = _one_plus_one(tokenizer)
    # One extractable row → comparison_population=none (< 4 texts, no held_out)
    report = _audit(
        tokenizer,
        tiny_model,
        dataset=[{"text": "a single ordinary weather sentence"}],
        real_sample=8,
    )
    sl = report["real_records"]["set_level"]
    assert sl["kind"] == "ranking_only"
    assert sl["kind"] != "skipped"
    assert sl["inferential"] is False


# ---------------------------------------------------------------------------
# F6 — preflight reaches the annex
# ---------------------------------------------------------------------------


def test_f6_absent_preflight_is_visible_in_annex(tokenizer, tiny_model):
    report = _audit(tokenizer, tiny_model)
    pre = report["compliance_annex"]["preflight"]
    assert pre["ran"] is False
    assert pre["execution"]["status"] == "not_run"
    assert pre["verification_unknown"] is True
    md = render_annex_markdown(report)
    assert "## Preflight" in md
    assert "NOT RUN" in md
    assert "not a pass" in md.lower()


def test_f6_hand_built_report_preflight_not_recorded():
    report = _hand_report()
    pre = report["compliance_annex"]["preflight"]
    assert pre["execution"]["status"] == "not_recorded"
    md = render_annex_markdown(report)
    assert "NOT RECORDED" in md


# ---------------------------------------------------------------------------
# F7 — reference.mode follows the fallback after disable_adapter failure
# ---------------------------------------------------------------------------


def test_f7_toggle_exception_is_recorded():
    class Boom:
        peft_config = {"default": object()}

        def disable_adapter(self):
            raise RuntimeError("toggle exploded")

    failures: list[str] = []

    def extract(_model, cache_tag=""):
        return SimpleNamespace(tag=cache_tag)

    ft, ref, how = _toggle_or_extract_ref(
        Boom(),
        extract,
        ref_model=None,
        toggle_safe=True,
        toggle_failures=failures,
    )
    assert how == "target_only"
    assert ref is None
    assert failures and "toggle exploded" in failures[0]


def test_f7_reconcile_updates_stale_disable_adapter_mode():
    meta, headline = reconcile_reference_mode(
        {"mode": "disable_adapter"},
        [{"ref_source": "target_only"}],
        ["RuntimeError: boom"],
        has_ref_model=False,
    )
    assert meta["mode"] == "target_only"
    assert meta["downgraded_from"] == "disable_adapter"
    assert headline == HEADLINE_ATTACK_FALLBACK
    assert meta.get("toggle_error")


def test_f7_run_audit_updates_mode_and_warns(tokenizer, tiny_model, monkeypatch):
    class BoomPeft:
        def __init__(self, inner):
            self._inner = inner
            self.peft_config = {
                "default": SimpleNamespace(peft_type="LORA", bias="none", r=8)
            }
            self.config = inner.config

        def disable_adapter(self):
            raise RuntimeError("toggle exploded")

        def __call__(self, *args, **kwargs):
            return self._inner(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(
        "memaudit.audit.base_equivalence_guard",
        lambda *a, **k: {
            "adapter_active": True,
            "restored": True,
            "max_abs_logit_diff": 0.0,
            "compared": True,
            "verdict": "pass",
            "atol_warn": 1e-5,
            "n_probes": 1,
            "probe_texts": ["x"],
        },
    )
    wrapped = BoomPeft(tiny_model)
    report = _audit(tokenizer, wrapped, ref="auto")
    assert report["reference"]["mode"] == "target_only"
    assert report["reference"].get("downgraded_from") == "disable_adapter"
    assert any("disable_adapter() failed" in w for w in report.get("audit_warnings") or [])


# ---------------------------------------------------------------------------
# F8 — reports stamp TOOL_VERSION, not stale dist metadata
# ---------------------------------------------------------------------------


def test_f8_report_uses_in_tree_constant(monkeypatch):
    monkeypatch.setattr("memaudit.utils.package_version", lambda: "0.1.0")
    report = _hand_report()
    assert report["tool_version"] == TOOL_VERSION
    assert report["tool_version"] == "0.2.0"
    assert report["compliance_annex"]["test_scope"]["tool_version"] == TOOL_VERSION


def test_f8_package_version_matches_constant():
    assert package_version() == TOOL_VERSION


# ---------------------------------------------------------------------------
# F9 — para 55(i) names the compound class
# ---------------------------------------------------------------------------


def test_f9_membership_edpb_class_includes_attribute():
    label = AUDIT_ATTACK_THREAT_MODELS["membership_inference"]["edpb_class"]
    assert label == "para 55(i) attribute and membership inference"
    assert "attribute and membership" in label
