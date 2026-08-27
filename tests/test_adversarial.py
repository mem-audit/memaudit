"""Adversarial tests: the tool must refuse to silently lie."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from memaudit.audit import run_audit, validate_manifest_for_audit
from memaudit.callback import MemorizationAuditCallback
from memaudit.canaries import generate_canaries
from memaudit.exceptions import MemauditAuditError, MemauditConfigError, MemauditPreflightError
from memaudit.injection import assert_secret_on_trainable_side, canary_record, inject, sniff_format
from memaudit.preflight import is_sharded_backend, run_preflight
from memaudit.scoring import combine_ft_ref, score_sequence
from memaudit.stats import tpr_at_fpr
from memaudit.types import canaries_from_obj
from memaudit.utils import write_json


def test_json_report_never_emits_nan(tmp_path):
    path = write_json(tmp_path / "r.json", {"tpr": float("nan"), "inf": float("inf"), "ok": 0.5})
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    obj = json.loads(raw)
    assert obj["tpr"] is None
    assert obj["inf"] is None
    assert obj["ok"] == 0.5


def test_tpr_headline_invalid_with_too_few_controls():
    out = tpr_at_fpr([3.0, 2.5, 2.0], [0.1, 0.0, -1.0], fpr=0.01)
    assert out["headline_valid"] is False
    assert out["tpr"] == 1.0  # exploratory number still computed
    assert "unidentified" in (out["warning"] or "")


def test_tpr_headline_valid_with_100_controls():
    members = [10.0] * 8
    controls = list(range(100))  # 0..99; members all above
    out = tpr_at_fpr(members, controls, fpr=0.01)
    assert out["headline_valid"] is True
    assert out["n_controls"] == 100
    assert 0.0 <= out["tpr"] <= 1.0


def test_combine_ft_ref_zero_nll_is_not_missing():
    ft = {"masked_nll": 0.0, "min_k": 0.0, "min_k_plus_plus": 0.1, "mean_logprob": 0.0, "n_scored_tokens": 4}
    ref = {"masked_nll": 0.0, "min_k": -1.0, "min_k_plus_plus": -1.0, "mean_logprob": -1.0, "n_scored_tokens": 4}
    out = combine_ft_ref(ft, ref)
    assert out["loss_ratio"] == 0.0
    assert out["loss_diff"] == 0.0
    assert out["headline_attack_used"] == "base_calibrated_min_k_plus_plus"


def test_inject_refuses_format_mismatch(tokenizer):
    host = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}]
    cans = generate_canaries(tokenizer, n=1, n_controls=1, family="random", seed=0, secret_len=25)
    with pytest.raises(MemauditConfigError, match="does not match"):
        inject(host, cans, fmt="text", seed=0)


def test_inject_refuses_sharegpt_from_value():
    with pytest.raises(MemauditConfigError, match="ShareGPT"):
        sniff_format([{"from": "human", "value": "hi"}])


def test_inject_refuses_prompt_side_secret(tokenizer, monkeypatch):
    def bad_record(canary, fmt):  # noqa: ARG001
        return {"prompt": canary.secret, "completion": "thanks"}

    import sys

    monkeypatch.setattr(sys.modules["memaudit.injection"], "canary_record", bad_record)
    host = [{"prompt": "Q?", "completion": "A."}]
    cans = generate_canaries(tokenizer, n=1, n_controls=1, family="random", seed=0, secret_len=25)
    with pytest.raises(MemauditConfigError, match="prompt"):
        inject(host, cans, fmt="prompt_completion", seed=0, include_prob=1.0)


def test_assert_secret_refuses_user_turn(tokenizer):
    cans = generate_canaries(tokenizer, n=1, n_controls=0, family="random", seed=0, secret_len=25)
    bad = {
        "messages": [
            {"role": "user", "content": cans[0].secret},
            {"role": "assistant", "content": "ok"},
        ]
    }
    with pytest.raises(MemauditConfigError, match="user"):
        assert_secret_on_trainable_side(bad, "messages", cans[0].secret)


def test_canary_record_never_puts_secret_in_prompt_or_user(tokenizer):
    cans = generate_canaries(tokenizer, n=2, n_controls=0, family="random", seed=1, secret_len=25)
    for c in cans:
        pc = canary_record(c, "prompt_completion")
        assert c.secret not in pc["prompt"]
        assert c.secret in pc["completion"]
        chat = canary_record(c, "messages")
        assert c.secret not in chat["messages"][0]["content"]
        assert c.secret in chat["messages"][1]["content"]


def test_include_prob_zero_and_one(tokenizer):
    host = [{"text": f"doc {i}"} for i in range(4)]
    cans = generate_canaries(tokenizer, n=4, n_controls=4, family="random", seed=0, secret_len=25)
    _, none = inject(host, cans, fmt="text", seed=0, include_prob=0.0)
    assert none["n_inserted_canaries"] == 0
    assert all(not c["included"] for c in none["canaries"])
    _, all_in = inject(host, cans, fmt="text", seed=0, include_prob=1.0)
    assert all_in["n_inserted_canaries"] == 4
    assert all(c["included"] for c in all_in["canaries"] if c["role"] != "control")
    assert all(not c["included"] for c in all_in["canaries"] if c["role"] == "control")


def test_controls_never_counted_as_members(tokenizer, tiny_model):
    host = [{"text": f"host {i} lorem"} for i in range(8)]
    cans = generate_canaries(tokenizer, n=3, n_controls=5, family="random", seed=0, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=0, include_prob=1.0)
    report = run_audit(
        tiny_model, tokenizer, manifest, dataset=ds, ref="none", real_sample=0, skip_generation=True
    )
    member_ids = {c["id"] for c in manifest["canaries"] if c["included"]}
    control_ids = {c["id"] for c in manifest["canaries"] if not c["included"]}
    for row in report["per_canary"]:
        if row["id"] in control_ids:
            assert row["included"] is False
        if row["id"] in member_ids:
            assert row["included"] is True
    assert report["membership"]["n_members"] == len(member_ids)
    assert report["membership"]["n_controls"] == len(control_ids)
    assert report["negative_controls"]["n"] == len(control_ids)


def test_run_audit_refuses_raw_canary_dump(tokenizer, tiny_model):
    cans = generate_canaries(tokenizer, n=2, n_controls=2, family="random", seed=0, secret_len=25)
    raw = {"canaries": [c.to_dict() for c in cans]}
    with pytest.raises(MemauditAuditError, match="included"):
        validate_manifest_for_audit(raw)
    with pytest.raises(MemauditAuditError):
        run_audit(tiny_model, tokenizer, raw, ref="none", real_sample=0, skip_generation=True)


def test_run_audit_refuses_all_excluded(tokenizer, tiny_model):
    host = [{"text": "only host"}]
    cans = generate_canaries(tokenizer, n=2, n_controls=2, family="random", seed=0, secret_len=25)
    _, manifest = inject(host, cans, fmt="text", seed=0, include_prob=0.0)
    with pytest.raises(MemauditAuditError, match="included"):
        run_audit(tiny_model, tokenizer, manifest, ref="none", real_sample=0, skip_generation=True)


def test_empty_dataset_inject_raises(tokenizer):
    cans = generate_canaries(tokenizer, n=1, n_controls=1, family="random", seed=0, secret_len=25)
    with pytest.raises(MemauditConfigError, match="empty"):
        inject([], cans, fmt="text", seed=0)


def test_canaries_from_obj_is_memaudit_error():
    with pytest.raises(MemauditConfigError):
        canaries_from_obj("not-a-manifest")


def test_callback_raises_without_tokenizer(tokenizer, tiny_model, tmp_path):
    host = [{"text": f"row {i}"} for i in range(6)]
    cans = generate_canaries(tokenizer, n=2, n_controls=2, family="random", seed=0, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=0, include_prob=1.0)
    args = SimpleNamespace(output_dir=str(tmp_path), max_length=1024, packing_strategy="bfd", packing=False)
    trainer = SimpleNamespace(
        model=tiny_model,
        train_dataset=ds,
        args=args,
        processing_class=None,
        tokenizer=None,
        accelerator=None,
    )
    cb = MemorizationAuditCallback(trainer=trainer, manifest=manifest, real_sample=0)
    control = SimpleNamespace()
    state = SimpleNamespace(is_world_process_zero=True)
    cb.on_train_begin(args, state, control, model=tiny_model, processing_class=tokenizer)
    trainer.processing_class = None
    trainer.tokenizer = None
    with pytest.raises(MemauditConfigError, match="tokenizer"):
        cb.on_train_end(args, state, control, model=tiny_model)


def test_zero3_defers_not_crash():
    acc = SimpleNamespace(distributed_type="DEEPSPEED", deepspeed_plugin=SimpleNamespace(zero_stage=3))
    trainer = SimpleNamespace(accelerator=acc)
    defer, reason = is_sharded_backend(trainer)
    assert defer is True
    assert "3" in reason or "DEEP" in reason.upper() or "stage" in reason


def test_fsdp_defers():
    acc = SimpleNamespace(distributed_type="FSDP")
    trainer = SimpleNamespace(accelerator=acc)
    defer, reason = is_sharded_backend(trainer)
    assert defer is True
    assert "FSDP" in reason


def test_preflight_fatal_when_record_exceeds_max_length(tokenizer, tiny_model):
    host = [{"text": "short"}]
    cans = generate_canaries(tokenizer, n=2, n_controls=2, family="random", seed=0, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=0, include_prob=1.0)
    with pytest.raises(MemauditPreflightError, match="max_length"):
        run_preflight(
            model=tiny_model,
            trainer=None,
            manifest=manifest,
            tokenizer=tokenizer,
            args=SimpleNamespace(max_length=8, packing_strategy="bfd"),
            train_dataset=ds,
            raise_fatal=True,
        )


def test_secret_span_scores_diverge_from_full_span(tiny_model):
    import torch

    torch.manual_seed(1)
    # Construct ids where prefix and secret regions differ in predictability
    ids = list(range(4, 40))
    full = score_sequence(tiny_model.eval(), ids, span=(0, len(ids)))
    secret = score_sequence(tiny_model.eval(), ids, span=(24, 40))
    assert full["n_scored_tokens"] != secret["n_scored_tokens"]
    # Same model, different spans — NLL is not required to differ on random
    # init, but the scored token sets must differ so a full-span bug is caught.
    assert full["n_scored_tokens"] == len(ids) - 1  # skip first
    # span (24, 40) clipped to seq_len=36 -> positions 24..35
    assert secret["n_scored_tokens"] == 12


def test_new_token_still_gated(tokenizer):
    with pytest.raises(MemauditConfigError, match="never resizes"):
        generate_canaries(tokenizer, n=1, n_controls=1, family="new_token")
