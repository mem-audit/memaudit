"""CPU micro-integration: generate ? inject ? score both verdicts on a tiny LM."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memaudit.audit import run_audit
from memaudit.callback import MemorizationAuditCallback
from memaudit.canaries import generate_canaries
from memaudit.injection import inject
from memaudit.preflight import run_preflight


@pytest.mark.smoke
def test_cpu_audit_two_verdicts(tokenizer, tiny_model, tmp_path):
    host = [{"text": f"ordinary training sentence number {i} about weather"} for i in range(16)]
    cans = generate_canaries(tokenizer, n=4, n_controls=6, family="random", seed=0, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=0, include_prob=1.0)
    assert manifest["n_inserted_canaries"] >= 1

    report = run_audit(
        model=tiny_model,
        tokenizer=tokenizer,
        manifest=manifest,
        dataset=ds,
        ref="none",
        real_sample=8,
        output_path=tmp_path / "memaudit-report.json",
        skip_generation=False,
    )
    assert (tmp_path / "memaudit-report.json").is_file()
    assert "membership" in report and "regurgitation" in report
    assert report["membership"]["n_members"] >= 1
    assert report["membership"]["n_controls"] >= 1
    assert "tpr_at_1pct_fpr" in report["membership"]
    assert "ci_low" in report["membership"]
    assert report["regurgitation"]["overall"]["n"] >= 1
    assert report["phone_home"] is False
    assert report["per_canary"]
    # both tiers actually ran
    assert any("regurgitation" in row for row in report["per_canary"])
    assert any("scores" in row and "masked_nll" in row["scores"] for row in report["per_canary"])


@pytest.mark.smoke
def test_callback_preflight_and_end(tokenizer, tiny_model, tmp_path):
    host = [{"text": f"row {i}"} for i in range(10)]
    cans = generate_canaries(tokenizer, n=3, n_controls=3, family="random", seed=3, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=3, include_prob=1.0)

    args = SimpleNamespace(
        output_dir=str(tmp_path),
        max_length=1024,
        packing_strategy="bfd",
        packing=False,
        completion_only_loss=False,
        assistant_only_loss=False,
        learning_rate=2e-5,
        num_train_epochs=1,
    )
    trainer = SimpleNamespace(
        model=tiny_model,
        train_dataset=ds,
        args=args,
        processing_class=tokenizer,
        tokenizer=None,
        accelerator=None,
    )
    cb = MemorizationAuditCallback(trainer=trainer, manifest=manifest, real_sample=4, ref="none")
    control = SimpleNamespace()
    state = SimpleNamespace(is_world_process_zero=True)
    cb.on_train_begin(args, state, control, model=tiny_model, processing_class=tokenizer)
    assert cb.preflight is not None
    assert (tmp_path / "memaudit-manifest.json").is_file()
    cb.on_train_end(args, state, control, model=tiny_model, processing_class=tokenizer)
    assert (tmp_path / "memaudit-report.json").is_file()
    assert cb.report["membership"]["headline_attack"]


def test_preflight_passes_on_injected(tokenizer, tiny_model):
    host = [{"text": f"z{i}"} for i in range(6)]
    cans = generate_canaries(tokenizer, n=2, n_controls=2, family="random", seed=0, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=0, include_prob=1.0)
    out = run_preflight(
        model=tiny_model,
        trainer=None,
        manifest=manifest,
        tokenizer=tokenizer,
        args=SimpleNamespace(max_length=512, packing_strategy="bfd"),
        train_dataset=ds,
        raise_fatal=True,
    )
    assert out["survival"]["n_found"] == out["survival"]["n_inserted"]
