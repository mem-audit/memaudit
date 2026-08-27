"""Multi-seed mode: stability block semantics (audit-procedure variance)."""

from __future__ import annotations

import pytest

from memaudit.audit import run_audit
from memaudit.canaries import generate_canaries
from memaudit.cli import _parse_seeds, build_parser
from memaudit.exceptions import MemauditConfigError, MemauditError
from memaudit.injection import inject


def _setup(tokenizer):
    host = [{"text": f"ordinary sentence {i} about weather and trains"} for i in range(16)]
    cans = generate_canaries(tokenizer, n=4, n_controls=8, family="random", seed=0, secret_len=25)
    return inject(host, cans, fmt="text", seed=0, include_prob=1.0)


def test_single_seed_default_has_no_stability(tokenizer, tiny_model, tmp_path):
    ds, manifest = _setup(tokenizer)
    report = run_audit(
        model=tiny_model,
        tokenizer=tokenizer,
        manifest=manifest,
        dataset=ds,
        ref="none",
        real_sample=0,
        skip_generation=True,
    )
    assert report["stability"] is None
    assert report["seeds"]["audit_seeds"] is None


def test_multiseed_adds_stability_block(tokenizer, tiny_model, tmp_path):
    ds, manifest = _setup(tokenizer)
    report = run_audit(
        model=tiny_model,
        tokenizer=tokenizer,
        manifest=manifest,
        dataset=ds,
        ref="none",
        real_sample=8,
        skip_generation=True,
        seeds=[0, 1, 2],
    )
    st = report["stability"]
    assert st is not None
    assert st["kind"] == "audit_procedure_variance"
    # labeled honestly: audit-procedure variance, not training variance
    assert "not" in st["label"].lower() and "training" in st["label"].lower()
    assert st["audit_seeds"] == [0, 1, 2]
    var = st["variance"]
    for key in ("tpr_mean", "tpr_min", "tpr_max", "tpr_std", "per_seed"):
        assert key in var, key
    assert len(var["per_seed"]) == 3
    for row in var["per_seed"]:
        assert set(row) >= {"seed", "tpr", "threshold", "n_detected", "auc", "headline_valid"}
        assert 0.0 <= row["tpr"] <= 1.0
    assert var["tpr_min"] <= var["tpr_mean"] <= var["tpr_max"]
    # real records re-sampled per seed
    assert st["real_records_per_seed"] is not None
    assert len(st["real_records_per_seed"]) == 3
    assert [r["seed"] for r in st["real_records_per_seed"]] == [0, 1, 2]
    # top-level seeds record the audit seeds
    assert report["seeds"]["audit_seeds"] == [0, 1, 2]
    # annex picks up the variance summary
    assert report["compliance_annex"]["quantified_results"]["stability"] is not None


def test_multiseed_is_deterministic(tokenizer, tiny_model):
    ds, manifest = _setup(tokenizer)
    kwargs = dict(
        model=tiny_model,
        tokenizer=tokenizer,
        manifest=manifest,
        dataset=None,
        ref="none",
        real_sample=0,
        skip_generation=True,
        seeds=[7, 8],
    )
    a = run_audit(**kwargs)["stability"]["variance"]
    b = run_audit(**kwargs)["stability"]["variance"]
    assert a["per_seed"] == b["per_seed"]
    assert a["tpr_mean"] == b["tpr_mean"]


def test_empty_seed_list_rejected(tokenizer, tiny_model):
    ds, manifest = _setup(tokenizer)
    with pytest.raises(MemauditConfigError):
        run_audit(
            model=tiny_model,
            tokenizer=tokenizer,
            manifest=manifest,
            ref="none",
            real_sample=0,
            skip_generation=True,
            seeds=[],
        )


def test_cli_seeds_parsing():
    assert _parse_seeds(None) is None
    assert _parse_seeds("") is None
    assert _parse_seeds("0,1,2") == [0, 1, 2]
    assert _parse_seeds("3") == [3]
    with pytest.raises(MemauditError):
        _parse_seeds("a,b")
    parser = build_parser()
    ns = parser.parse_args(
        ["audit", "--model", "./out", "--canary-set", "./m.json", "--seeds", "0,1,2",
         "--release-context", "internal"]
    )
    assert ns.seeds == "0,1,2"
    assert ns.release_context == "internal"
