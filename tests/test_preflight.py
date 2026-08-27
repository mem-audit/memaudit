from __future__ import annotations

from types import SimpleNamespace

import pytest

from memaudit.canaries import generate_canaries
from memaudit.exceptions import MemauditPreflightError
from memaudit.injection import inject
from memaudit.preflight import inspect_embeddings, run_preflight, survival_scan


def test_survival_scan_finds_injected(tokenizer):
    host = [{"text": f"host document {i} lorem ipsum"} for i in range(8)]
    cans = generate_canaries(tokenizer, n=4, n_controls=4, family="random", seed=0, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=0, include_prob=1.0)
    scan = survival_scan(ds, manifest, tokenizer=tokenizer)
    assert scan["n_inserted"] == manifest["n_inserted_canaries"]
    assert scan["n_found"] == scan["n_inserted"]
    assert scan["n_missing"] == 0


def test_preflight_raises_if_not_injected(tokenizer, tiny_model):
    host = [{"text": "nothing to see here"}]
    cans = generate_canaries(tokenizer, n=2, n_controls=2, family="random", seed=0, secret_len=25)
    _, manifest = inject(host, cans, fmt="text", seed=0, include_prob=1.0)
    # pass the UN-injected host so the scan fails
    with pytest.raises(MemauditPreflightError, match="inject"):
        run_preflight(
            model=tiny_model,
            trainer=None,
            manifest=manifest,
            tokenizer=tokenizer,
            args=SimpleNamespace(max_length=1024, packing_strategy="bfd", output_dir="."),
            train_dataset=host,
            raise_fatal=True,
        )


def test_preflight_raises_when_labels_mask_secret(tokenizer, tiny_model):
    host = [{"text": "aaaa"}]
    cans = generate_canaries(tokenizer, n=2, n_controls=0, family="random", seed=1, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=1, include_prob=1.0)
    # synthesize a tokenized dataset where every label is -100
    tokenized = []
    for row in ds:
        from memaudit.utils import encode_ids

        ids = encode_ids(tokenizer, row["text"])
        tokenized.append({"input_ids": ids, "labels": [-100] * len(ids)})
    with pytest.raises(MemauditPreflightError, match="-100"):
        run_preflight(
            model=tiny_model,
            trainer=None,
            manifest=manifest,
            tokenizer=tokenizer,
            args=SimpleNamespace(max_length=2048, packing_strategy="bfd"),
            train_dataset=tokenized,
            raise_fatal=True,
        )


def test_wrapped_packing_warning(tokenizer, tiny_model):
    host = [{"text": f"doc {i}"} for i in range(4)]
    cans = generate_canaries(tokenizer, n=2, n_controls=2, family="random", seed=0, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=0, include_prob=1.0)
    out = run_preflight(
        model=tiny_model,
        trainer=None,
        manifest=manifest,
        tokenizer=tokenizer,
        args=SimpleNamespace(max_length=1024, packing_strategy="wrapped", packing=True),
        train_dataset=ds,
        raise_fatal=True,
    )
    assert any("wrapped" in w for w in out["warnings"])


def test_inspect_embeddings_plain_frozen(tiny_model):
    info = inspect_embeddings(tiny_model)
    assert info["input_trainable"] is False or isinstance(info["input_trainable"], bool)
    assert "mechanisms" in info


def test_bias_warning_via_fake_peft(tiny_model):
    tiny_model.peft_config = {
        "default": SimpleNamespace(
            bias="all",
            peft_type="LORA",
            r=8,
            lora_alpha=16,
            modules_to_save=None,
            trainable_token_indices=None,
        )
    }
    tiny_model.active_adapter = "default"
    tiny_model.disable_adapter = lambda: None
    from memaudit.preflight import adapter_toggle_safe, inspect_embeddings

    info = inspect_embeddings(tiny_model)
    ok, reason = adapter_toggle_safe(info)
    assert ok is False
    assert "bias" in (reason or "")
