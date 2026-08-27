"""Buyer-risk regressions: defaults, --ref auto, injection module name."""

from __future__ import annotations

import importlib
import warnings

import pytest

from memaudit.canaries import generate_canaries
from memaudit.constants import DEFAULT_N_CONTROLS, MIN_CONTROLS_FOR_TPR_AT_1PCT
from memaudit.exceptions import MemauditConfigError
from memaudit.injection import inject


def test_default_n_controls_meets_tpr_floor():
    assert DEFAULT_N_CONTROLS >= MIN_CONTROLS_FOR_TPR_AT_1PCT


def test_low_n_controls_warns(tokenizer):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        generate_canaries(tokenizer, n=2, n_controls=4, family="random", seed=0, secret_len=25)
    assert any("n_controls" in str(w.message) and "floor" in str(w.message) for w in caught)


def test_ref_auto_refuses_full_ft_without_base(tokenizer, tiny_model):
    from memaudit.audit import run_audit

    host = [{"text": f"doc {i}"} for i in range(4)]
    cans = generate_canaries(tokenizer, n=2, n_controls=2, family="random", seed=0, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=0, include_prob=1.0)
    with pytest.raises(MemauditConfigError, match="ref auto"):
        run_audit(tiny_model, tokenizer, manifest, dataset=ds, ref="auto", real_sample=0, skip_generation=True)


def test_ref_none_is_explicit_target_only(tokenizer, tiny_model):
    from memaudit.audit import run_audit

    host = [{"text": f"doc {i}"} for i in range(4)]
    cans = generate_canaries(tokenizer, n=2, n_controls=2, family="random", seed=0, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=0, include_prob=1.0)
    report = run_audit(
        tiny_model, tokenizer, manifest, dataset=ds, ref="none", real_sample=0, skip_generation=True
    )
    assert report["membership"]["headline_attack"] == "min_k_plus_plus"
    assert any("target-only" in w for w in (report.get("audit_warnings") or []))


def test_injection_module_is_not_shadowed():
    import memaudit
    import memaudit.injection as inj_mod

    assert callable(memaudit.inject)
    assert inj_mod.inject is memaudit.inject
    assert importlib.import_module("memaudit.injection").__name__ == "memaudit.injection"
