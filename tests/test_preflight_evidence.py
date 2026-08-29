"""WS1 fixtures: collator labels, evidence levels, scan coverage, fatal policy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memaudit.canaries import generate_canaries
from memaudit.exceptions import MemauditPreflightError
from memaudit.injection import TEXT_PREFIX, inject
from memaudit.preflight import inspect_training_args, run_preflight, survival_scan
from memaudit.preflight_labels import effective_labels
from memaudit.utils import encode_ids, example_text


class _MaskCollator:
    """TRL-shaped collator: labels materialize here, not on the row."""

    def __init__(self, completion_only_loss: bool = False, padding_free: bool = False) -> None:
        self.completion_only_loss = completion_only_loss
        self.padding_free = padding_free
        self.calls = 0

    def __call__(self, features):
        self.calls += 1
        row = features[0]
        ids = list(row["input_ids"])
        labels = list(row["labels"]) if "labels" in row else list(ids)
        if self.completion_only_loss and "completion_mask" in row:
            mask = list(row["completion_mask"])
            labels = [-100 if m == 0 else lab for lab, m in zip(labels, mask)]
        if "assistant_masks" in row:
            mask = list(row["assistant_masks"])
            labels = [-100 if m == 0 else lab for lab, m in zip(labels, mask)]
        if self.padding_free:
            seq = row.get("seq_lengths")
            if seq:
                pos = 0
                for slen in seq:
                    if pos < len(labels):
                        labels[pos] = -100
                    pos += int(slen)
            elif labels:
                labels[0] = -100
        return {"input_ids": [ids], "labels": [labels]}


def _included(manifest: dict) -> list[dict]:
    return [c for c in manifest["canaries"] if c.get("included")]


def _tokenized_canary_row(canary: dict, tokenizer, *, completion_mask=None, assistant_masks=None, extra=None):
    ids = list(canary["secret_token_ids"])
    row = {"input_ids": ids}
    if completion_mask is not None:
        row["completion_mask"] = completion_mask
    if assistant_masks is not None:
        row["assistant_masks"] = assistant_masks
    if extra:
        row.update(extra)
    return row


def _run(
    tiny_model,
    tokenizer,
    manifest,
    dataset,
    *,
    args=None,
    trainer=None,
    raise_fatal=True,
    scan_limit=50_000,
):
    args = args or SimpleNamespace(max_length=2048, packing=False, packing_strategy="bfd")
    return run_preflight(
        model=tiny_model,
        trainer=trainer,
        manifest=manifest,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset,
        raise_fatal=raise_fatal,
        scan_limit=scan_limit,
    )


def test_collator_labels_preferred_over_missing_column(tokenizer, tiny_model):
    cans = generate_canaries(tokenizer, n=2, n_controls=0, family="random", seed=1, secret_len=25)
    _, manifest = inject([{"text": "host"}], cans, fmt="text", seed=1, include_prob=1.0)
    rows = []
    for c in _included(manifest):
        ids = list(c["secret_token_ids"])
        rows.append({"input_ids": ids, "completion_mask": [1] * len(ids)})
    collator = _MaskCollator(completion_only_loss=True)
    trainer = SimpleNamespace(
        data_collator=collator,
        completion_only_loss=True,
        args=SimpleNamespace(completion_only_loss=True, max_length=2048),
    )
    out = _run(tiny_model, tokenizer, manifest, rows, trainer=trainer)
    assert collator.calls >= 1
    assert out["ran"] is True
    for ev in out["survival"]["per_canary"]:
        assert ev["labels_source"] == "collator_call"
        assert ev["directly_supervised"] is True
        assert ev["evidence_level"] == "directly_supervised"


def test_fixture_a_truncated_secret_absent_is_fatal(tokenizer, tiny_model):
    cans = generate_canaries(tokenizer, n=2, n_controls=0, family="random", seed=2, secret_len=25)
    _, manifest = inject([{"text": "host"}], cans, fmt="text", seed=2, include_prob=1.0)
    # Prepared rows are prompt-only fragments — secret never appears.
    rows = [{"input_ids": [10, 11, 12, 13]} for _ in range(4)]
    with pytest.raises(MemauditPreflightError, match="complete inspection|inject"):
        _run(tiny_model, tokenizer, manifest, rows)


def test_fixture_a_secret_present_but_completion_mask_zero_is_fatal(tokenizer, tiny_model):
    cans = generate_canaries(tokenizer, n=2, n_controls=0, family="random", seed=3, secret_len=25)
    _, manifest = inject([{"text": "host"}], cans, fmt="text", seed=3, include_prob=1.0)
    rows = []
    for c in _included(manifest):
        ids = list(c["secret_token_ids"])
        rows.append({"input_ids": ids, "completion_mask": [0] * len(ids)})
    collator = _MaskCollator(completion_only_loss=True)
    args = SimpleNamespace(
        max_length=2048,
        completion_only_loss=True,
        packing=False,
        packing_strategy="bfd",
    )
    trainer = SimpleNamespace(data_collator=collator, completion_only_loss=True, args=args)
    with pytest.raises(MemauditPreflightError, match="-100"):
        _run(tiny_model, tokenizer, manifest, rows, args=args, trainer=trainer)


def test_fixture_b1_all_zero_assistant_masks_is_fatal(tokenizer, tiny_model):
    cans = generate_canaries(tokenizer, n=2, n_controls=0, family="random", seed=4, secret_len=25)
    _, manifest = inject([{"text": "host"}], cans, fmt="text", seed=4, include_prob=1.0)
    rows = []
    for c in _included(manifest):
        ids = list(c["secret_token_ids"])
        rows.append({"input_ids": ids, "assistant_masks": [0] * len(ids)})
    args = SimpleNamespace(
        max_length=2048,
        assistant_only_loss=True,
        packing=False,
        packing_strategy="bfd",
    )
    trainer = SimpleNamespace(
        data_collator=_MaskCollator(),
        args=args,
        completion_only_loss=False,
    )
    with pytest.raises(MemauditPreflightError, match="-100"):
        _run(tiny_model, tokenizer, manifest, rows, args=args, trainer=trainer)


def test_fixture_b2_mask_column_absent_is_warning_not_fatal(tokenizer, tiny_model):
    cans = generate_canaries(tokenizer, n=2, n_controls=0, family="random", seed=5, secret_len=25)
    _, manifest = inject([{"text": "host"}], cans, fmt="text", seed=5, include_prob=1.0)
    rows = [_tokenized_canary_row(c, tokenizer) for c in _included(manifest)]
    args = SimpleNamespace(
        max_length=2048,
        assistant_only_loss=True,
        packing=False,
        packing_strategy="bfd",
    )
    # Full-sequence collator: supervision broadens (no mask applied).
    trainer = SimpleNamespace(data_collator=_MaskCollator(), args=args)
    out = _run(tiny_model, tokenizer, manifest, rows, args=args, trainer=trainer)
    assert out["survival"]["n_mask_column_missing"] >= 1
    assert any("full-sequence" in w for w in out["warnings"])
    assert not out["fatal"]


def test_fixture_c_packed_prompt_only_chunk_is_fatal(tokenizer, tiny_model):
    cans = generate_canaries(tokenizer, n=2, n_controls=0, family="random", seed=6, secret_len=25)
    _, manifest = inject([{"text": "host"}], cans, fmt="text", seed=6, include_prob=1.0)
    rows = [
        {"input_ids": [8, 9, 10, 11], "completion_mask": [0, 0, 0, 0], "seq_lengths": [4]}
        for _ in range(3)
    ]
    args = SimpleNamespace(
        max_length=16,
        completion_only_loss=True,
        packing=True,
        packing_strategy="bfd",
        padding_free=True,
    )
    trainer = SimpleNamespace(
        data_collator=_MaskCollator(completion_only_loss=True, padding_free=True),
        completion_only_loss=True,
        args=args,
        padding_free=True,
    )
    with pytest.raises(MemauditPreflightError, match="complete inspection|not present|-100"):
        _run(tiny_model, tokenizer, manifest, rows, args=args, trainer=trainer)


def test_fixture_c_secret_in_packed_row_but_mask_zero_is_fatal(tokenizer, tiny_model):
    cans = generate_canaries(tokenizer, n=1, n_controls=0, family="random", seed=7, secret_len=25)
    _, manifest = inject([{"text": "host"}], cans, fmt="text", seed=7, include_prob=1.0)
    c = _included(manifest)[0]
    ids = list(c["secret_token_ids"])
    rows = [{"input_ids": ids, "completion_mask": [0] * len(ids), "seq_lengths": [len(ids)]}]
    args = SimpleNamespace(
        max_length=2048,
        completion_only_loss=True,
        packing=True,
        packing_strategy="bfd",
        padding_free=True,
    )
    trainer = SimpleNamespace(
        data_collator=_MaskCollator(completion_only_loss=True, padding_free=True),
        completion_only_loss=True,
        args=args,
        padding_free=True,
    )
    with pytest.raises(MemauditPreflightError, match="-100"):
        _run(tiny_model, tokenizer, manifest, rows, args=args, trainer=trainer)


def test_fixture_c_straddle_is_warning_not_silent_missing(tokenizer, tiny_model):
    cans = generate_canaries(tokenizer, n=1, n_controls=0, family="random", seed=8, secret_len=25)
    _, manifest = inject([{"text": "host"}], cans, fmt="text", seed=8, include_prob=1.0)
    c = _included(manifest)[0]
    ids = list(c["secret_token_ids"])
    mid = len(ids) // 2
    rows = [
        {"input_ids": ids[:mid], "completion_mask": [1] * mid},
        {"input_ids": ids[mid:], "completion_mask": [1] * (len(ids) - mid)},
    ]
    args = SimpleNamespace(
        max_length=2048,
        completion_only_loss=True,
        packing=True,
        packing_strategy="wrapped",
    )
    trainer = SimpleNamespace(
        data_collator=_MaskCollator(completion_only_loss=True),
        completion_only_loss=True,
        args=args,
    )
    out = _run(tiny_model, tokenizer, manifest, rows, args=args, trainer=trainer)
    ev = out["survival"]["per_canary"][0]
    assert ev["record_observed"] is True
    assert ev["split_across_packed_rows"] is True
    assert ev["directly_supervised"] is False
    assert any("split across packed rows" in w for w in out["warnings"])


def test_fixture_c_padding_free_first_token_is_partial(tokenizer, tiny_model):
    cans = generate_canaries(tokenizer, n=1, n_controls=0, family="random", seed=9, secret_len=25)
    _, manifest = inject([{"text": "host"}], cans, fmt="text", seed=9, include_prob=1.0)
    c = _included(manifest)[0]
    ids = list(c["secret_token_ids"])
    rows = [{"input_ids": ids, "completion_mask": [1] * len(ids), "seq_lengths": [len(ids)]}]
    args = SimpleNamespace(
        max_length=2048,
        completion_only_loss=True,
        packing=True,
        packing_strategy="bfd",
        padding_free=True,
    )
    trainer = SimpleNamespace(
        data_collator=_MaskCollator(completion_only_loss=True, padding_free=True),
        completion_only_loss=True,
        padding_free=True,
        args=args,
    )
    out = _run(tiny_model, tokenizer, manifest, rows, args=args, trainer=trainer)
    ev = out["survival"]["per_canary"][0]
    assert ev["loss_mask_checked"] is True
    assert ev["directly_supervised"] is False
    assert ev["supervised_token_fraction"] is not None
    assert 0.0 < ev["supervised_token_fraction"] < 1.0
    assert any("partially supervised" in w for w in out["warnings"])


def test_row_vanished_after_complete_scan_is_fatal(tokenizer, tiny_model):
    cans = generate_canaries(tokenizer, n=2, n_controls=0, family="random", seed=10, secret_len=25)
    _, manifest = inject([{"text": "host"}], cans, fmt="text", seed=10, include_prob=1.0)
    kept, dropped = _included(manifest)
    rows = [{"input_ids": list(kept["secret_token_ids"]), "completion_mask": [1] * len(kept["secret_token_ids"])}]
    args = SimpleNamespace(max_length=2048, completion_only_loss=True, packing=False)
    trainer = SimpleNamespace(
        data_collator=_MaskCollator(completion_only_loss=True),
        completion_only_loss=True,
        args=args,
    )
    with pytest.raises(MemauditPreflightError, match="complete inspection|not present"):
        _run(tiny_model, tokenizer, manifest, rows, args=args, trainer=trainer)


def test_scan_window_miss_is_not_attributed_to_truncation(tokenizer, tiny_model):
    cans = generate_canaries(tokenizer, n=2, n_controls=0, family="random", seed=11, secret_len=25)
    _, manifest = inject([{"text": "host"}], cans, fmt="text", seed=11, include_prob=1.0)
    inserted = _included(manifest)
    dataset = (
        [{"text": f"{TEXT_PREFIX}{inserted[0]['secret']}"}]
        + [{"text": "zzzz no secret here"} for _ in range(20)]
        + [{"text": f"{TEXT_PREFIX}{inserted[1]['secret']}"}]
    )
    scan = survival_scan(dataset, manifest, tokenizer=tokenizer, limit=8)
    assert scan["scan_complete"] is False
    assert scan["rows_scanned"] == 8
    assert scan["rows_total"] == len(dataset)
    assert scan["n_found"] == 1
    assert scan["n_missing"] == 1
    missing = [e for e in scan["per_canary"] if not e["record_observed"]]
    assert missing[0]["miss_reason"] == "scan_window"

    out = _run(
        tiny_model,
        tokenizer,
        manifest,
        dataset,
        raise_fatal=True,
        scan_limit=8,
    )
    assert any("scan window" in w for w in out["warnings"])
    assert not any("Truncation or a formatting_func may have dropped" in w for w in out["warnings"])
    assert not any("complete inspection" in f for f in out["fatal"])


def test_verification_unknown_when_no_label_source(tokenizer, tiny_model):
    cans = generate_canaries(tokenizer, n=2, n_controls=0, family="random", seed=12, secret_len=25)
    ds, manifest = inject([{"text": "host"}], cans, fmt="text", seed=12, include_prob=1.0)
    out = _run(tiny_model, tokenizer, manifest, ds, trainer=None)
    evs = out["survival"]["per_canary"]
    assert evs
    assert all(e["record_observed"] for e in evs)
    assert all(e["verification_unknown"] for e in evs)
    assert all(e["directly_supervised"] is False for e in evs)
    assert any("verification_unknown" in w or "could not have their loss mask" in w for w in out["warnings"])


def test_inspect_training_args_effective_completion_only_loss():
    args = SimpleNamespace(completion_only_loss=None, packing=False, packing_strategy="bfd")
    trainer = SimpleNamespace(completion_only_loss=True, data_collator=SimpleNamespace(completion_only_loss=True))
    info = inspect_training_args(args, trainer=trainer, dataset=[{"prompt": "p", "completion": "c"}])
    assert info["completion_only_loss"] is None
    assert info["completion_only_loss_effective"] is True
    sniffed = inspect_training_args(
        SimpleNamespace(completion_only_loss=None, packing=False),
        trainer=None,
        dataset=[{"prompt": "p", "completion": "c"}],
    )
    assert sniffed["completion_only_loss_effective"] is True
    skip = inspect_training_args(
        SimpleNamespace(dataset_kwargs={"skip_prepare_dataset": True}, packing=False),
    )
    assert skip["skip_prepare_dataset"] is True
    requeue = inspect_training_args(SimpleNamespace(packing=True, packing_strategy="bfd-requeue"))
    assert requeue["packing_strategy"] == "bfd-requeue"
    assert requeue["padding_free_forced_by_bfd"] is False


def test_max_length_counts_special_token_overhead(tokenizer, tiny_model):
    from memaudit.injection import canary_record
    from memaudit.preflight import _canary_record_n_tokens

    cans = generate_canaries(tokenizer, n=1, n_controls=0, family="random", seed=13, secret_len=25)
    rec = canary_record(cans[0], "text")
    blob = example_text(rec, "text")
    n_raw = len(encode_ids(tokenizer, blob, add_special_tokens=False))
    n_full = _canary_record_n_tokens(cans[0].to_dict(), "text", tokenizer)
    assert n_full == n_raw + 2  # FakeTokenizer BOS+EOS


def test_effective_labels_mask_fallback_without_collator():
    row = {"input_ids": [5, 6, 7, 8], "completion_mask": [0, 0, 1, 1]}
    result = effective_labels(row, trainer=None, config={"completion_only_loss_effective": True})
    assert result is not None
    assert result.source == "mask_column"
    assert result.labels == [-100, -100, 7, 8]


def test_report_provenance_and_comparison_split(tokenizer, tiny_model, tmp_path):
    from memaudit.audit import run_audit

    host = [{"text": f"ordinary training sentence number {i} about weather"} for i in range(16)]
    cans = generate_canaries(tokenizer, n=2, n_controls=2, family="high_ppl", seed=14, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=14, include_prob=1.0)
    report = run_audit(
        tiny_model,
        tokenizer,
        manifest,
        dataset=ds,
        ref="none",
        real_sample=8,
        skip_generation=True,
        output_path=tmp_path / "r.json",
    )
    assert report["audit_scope"]["requested_family"] == "high_ppl"
    assert report["audit_scope"]["actual_generator"] == "uniform_vocab"
    assert report["preflight"]["ran"] is False
    row = next(r for r in report["per_canary"] if r["included"])
    assert row["requested_family"] == "high_ppl"
    assert row["actual_generator"] == "uniform_vocab"
    assert row["verification_unknown"] is True
    real = report["real_records"]
    assert real["comparison_population"] == "training_split"
    assert "n_comparison_split" in real
    assert "n_held_out" not in real
    sl = real.get("set_level") or {}
    assert sl.get("inferential") is False
    assert sl.get("kind") == "descriptive_ranking_only"
    assert sl.get("p_value") is None
    assert "training_split" in sl.get("note", "")
    assert "member-vs-nonmember" in sl.get("note", "")
    assert "no genuine held-out" in sl.get("note", "").lower()
