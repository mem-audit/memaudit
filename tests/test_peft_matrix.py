"""WS4 PEFT configuration-semantics matrix. Live tiny models + stub QLoRA path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from memaudit.exceptions import MemauditPreflightError
from memaudit.peft_semantics import (
    base_equivalence_guard,
    detect_quantization,
    quantization_ref_mismatch,
    trainable_token_canary_findings,
)
from memaudit.preflight import adapter_toggle_safe, inspect_embeddings, run_preflight


peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")


def _tiny_llama(*, tie: bool = True, vocab: int = 256):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        vocab_size=vocab,
        max_position_embeddings=64,
        tie_word_embeddings=tie,
        rms_norm_eps=1e-5,
    )
    return LlamaForCausalLM(cfg)


def _tiny_gpt2(vocab: int = 128):
    from transformers import GPT2Config, GPT2LMHeadModel

    return GPT2LMHeadModel(
        GPT2Config(n_embd=32, n_layer=2, n_head=2, n_positions=64, vocab_size=vocab)
    )


def _wrap(model, **kwargs):
    from peft import LoraConfig, get_peft_model

    cfg = dict(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    cfg.update(kwargs)
    return get_peft_model(model, LoraConfig(**cfg))


def _logits(model, ids):
    model.eval()
    with torch.inference_mode():
        return model(input_ids=ids).logits.detach().clone()


def _train_steps(model, ids, steps: int = 3, lr: float = 0.5):
    model.train()
    opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
    model.eval()


def _new_token_canary(secret_ids: list[int], *, cid: str = "c-new_token-0000") -> dict:
    secret = " ".join(f"t{i}" for i in secret_ids)
    return {
        "id": cid,
        "family": "new_token",
        "secret": secret,
        "secret_token_ids": list(secret_ids),
        "prefix": "",
        "prefix_token_ids": [],
        "secret_span": [0, len(secret_ids)],
        "repetitions": 1,
        "role": "member",
        "included": True,
        "requested_family": "new_token",
        "actual_generator": "new_token",
    }


def _manifest_with(canaries: list[dict]) -> dict:
    return {
        "canaries": canaries,
        "fmt": "text",
        "n_inserted_canaries": sum(1 for c in canaries if c.get("included")),
        "seed": 0,
    }


# ---------------------------------------------------------------------------
# Row 1 — plain LoRA
# ---------------------------------------------------------------------------


@pytest.mark.peft
def test_plain_lora_toggle_exact():
    model = _tiny_llama(tie=True)
    ids = torch.randint(0, 256, (1, 8))
    pre = _logits(model, ids)
    wrapped = _wrap(model)
    info = inspect_embeddings(wrapped)
    ok, reason = adapter_toggle_safe(info)
    assert ok is True, reason
    assert "plain_frozen_embedding" in (info.get("input_mechanisms") or info["mechanisms"])
    _train_steps(wrapped, ids)
    enabled = _logits(wrapped, ids)
    with wrapped.disable_adapter():
        disabled = _logits(wrapped, ids)
    assert not torch.equal(enabled, disabled)
    assert torch.equal(disabled, pre)


# ---------------------------------------------------------------------------
# Row 2 — modules_to_save
# ---------------------------------------------------------------------------


@pytest.mark.peft
def test_modules_to_save_lm_head_reported():
    untied = _wrap(_tiny_llama(tie=False), modules_to_save=["lm_head"])
    info = inspect_embeddings(untied)
    assert info["modules_to_save"] == ["lm_head"]
    assert "ModulesToSaveWrapper" in (info.get("output_mechanisms") or [])
    assert info["output_trainable"] is True
    assert "ModulesToSaveWrapper" in info["mechanisms"]

    tied_base = _tiny_llama(tie=True)
    assert tied_base.get_input_embeddings().weight.data_ptr() == tied_base.get_output_embeddings().weight.data_ptr()
    tied = _wrap(tied_base, modules_to_save=["lm_head"])
    tied_info = inspect_embeddings(tied)
    tying = tied_info.get("weight_tying") or {}
    assert tying.get("shared") is False
    assert tying.get("tie_broken_by_wrap") is True
    assert tied.get_input_embeddings().weight.data_ptr() != tied.get_output_embeddings().weight.data_ptr()


# ---------------------------------------------------------------------------
# Row 3 — trainable_token_indices ∩ canary ids
# ---------------------------------------------------------------------------


@pytest.mark.peft
def test_trainable_tokens_index_canary_mismatch(tokenizer):
    """Known-bad: new_token canary on frozen rows must fail preflight."""
    model = _wrap(_tiny_llama(tie=True), trainable_token_indices=[3, 4])
    info = inspect_embeddings(model)
    assert "TrainableTokensWrapper" in (info.get("input_mechanisms") or info["mechanisms"])
    assert info["trainable_token_index_set"] == [3, 4]
    assert info["trainable"] is True  # boolean would have passed the old gate

    frozen_ids = [10, 11, 12, 13, 14, 15, 16, 17]
    canary = _new_token_canary(frozen_ids)
    secret = canary["secret"]
    ds = [{"text": f"host document {secret} tail"}]
    manifest = _manifest_with([canary])
    with pytest.raises(MemauditPreflightError, match="trainable_token_indices|frozen|outside"):
        run_preflight(
            model=model,
            trainer=None,
            manifest=manifest,
            tokenizer=tokenizer,
            args=SimpleNamespace(max_length=2048, packing=False, packing_strategy="bfd"),
            train_dataset=ds,
            raise_fatal=True,
        )


def test_trainable_token_intersection_helper():
    manifest = _manifest_with([_new_token_canary([10, 11])])
    fatal, _warn, rows = trainable_token_canary_findings(manifest, [3, 4])
    assert fatal
    assert rows[0]["all_frozen"] is True
    ok_manifest = _manifest_with([_new_token_canary([3, 4])])
    fatal2, _, rows2 = trainable_token_canary_findings(ok_manifest, [3, 4])
    assert not fatal2
    assert rows2[0]["all_trained"] is True


# ---------------------------------------------------------------------------
# Row 4 — tied + ensure_weight_tying
# ---------------------------------------------------------------------------


@pytest.mark.peft
def test_tied_ensure_weight_tying_shared_storage():
    base = _tiny_llama(tie=True)
    ids = torch.randint(0, 256, (1, 8))
    pre = _logits(base, ids)
    wrapped = _wrap(
        base,
        modules_to_save=["embed_tokens"],
        ensure_weight_tying=True,
    )
    info = inspect_embeddings(wrapped)
    assert info["ensure_weight_tying"] is True
    assert "ModulesToSaveWrapper" in (info.get("input_mechanisms") or [])
    assert "ModulesToSaveWrapper" in (info.get("output_mechanisms") or [])
    tying = info.get("weight_tying") or {}
    assert tying.get("shared") is True
    assert tying.get("tie_broken_by_wrap") is False
    assert tying.get("ensure_weight_tying_engaged") is True
    names = info.get("embedding_layer_names") or {}
    assert names.get("conventional") is True
    with wrapped.disable_adapter():
        disabled = _logits(wrapped, ids)
    assert torch.equal(disabled, pre)


# ---------------------------------------------------------------------------
# Row 5 — untied embedding + lm_head targeted
# ---------------------------------------------------------------------------


@pytest.mark.peft
def test_untied_embedding_lora_detection():
    wrapped = _wrap(
        _tiny_llama(tie=False),
        target_modules=["embed_tokens", "lm_head"],
    )
    info = inspect_embeddings(wrapped)
    assert "lora.Embedding" in (info.get("input_mechanisms") or [])
    assert "lora.Embedding" in info["mechanisms"]
    assert "lora.Linear" in (info.get("output_mechanisms") or [])
    assert info["input_trainable"] is True
    assert info["output_trainable"] is True
    emb = wrapped.get_input_embeddings()
    assert hasattr(emb, "lora_embedding_A")
    # hardened path must not depend on the literal "lora.Embedding" substring
    assert "lora.Embedding" not in str(type(emb))


# ---------------------------------------------------------------------------
# Row 6 — QLoRA detection without bitsandbytes
# ---------------------------------------------------------------------------


class Linear4bit(nn.Module):
    """Stub named like bitsandbytes; no real 4-bit load."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 4))

    def forward(self, x):  # noqa: ANN001
        return x


class _QuantStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = Linear4bit()
        self.config = SimpleNamespace(
            quantization_config={"load_in_4bit": True},
            tie_word_embeddings=False,
        )
        self.is_loaded_in_4bit = True

    def get_input_embeddings(self):
        return None

    def get_output_embeddings(self):
        return None


def test_quantized_target_fp_ref_warning(tiny_model):
    stub = _QuantStub()
    q = detect_quantization(stub)
    assert q["quantized"] is True
    assert any("4bit" in k.lower() or k == "Linear4bit" for k in q["kinds"])
    info = inspect_embeddings(stub)
    assert info["quantized"] is True
    assert info.get("embedding_verification") == "verification_unknown"
    mismatch, msg = quantization_ref_mismatch(info, tiny_model)
    assert mismatch is True
    assert msg and "full-precision" in msg
    # matching quantized ref is not a mismatch
    ok, _ = quantization_ref_mismatch(info, stub)
    assert ok is False


# ---------------------------------------------------------------------------
# Row 7 — MoE / target_parameters
# ---------------------------------------------------------------------------


class _PlainExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feed_forward = nn.Module()
        self.feed_forward.experts = nn.Module()
        self.feed_forward.experts.gate_up_proj = nn.Parameter(torch.randn(3, 8, 8))
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def forward(self, x):  # noqa: ANN001
        return x


@pytest.mark.peft
def test_target_parameters_moe_toggle():
    from peft import LoraConfig, get_peft_model

    plain = _PlainExperts()
    wrapped = get_peft_model(
        plain,
        LoraConfig(
            r=4,
            lora_alpha=8,
            bias="none",
            target_modules=[],
            target_parameters=["feed_forward.experts.gate_up_proj"],
        ),
    )
    info = inspect_embeddings(wrapped)
    assert info["target_parameters"] == ["feed_forward.experts.gate_up_proj"]
    assert info.get("embedding_verification") == "verification_unknown"
    assert "plain_frozen_embedding" not in info["mechanisms"]
    ok, _ = adapter_toggle_safe(info)
    assert ok is True  # toggle still allowed; not an unexamined all-clear
    assert info.get("embedding_verification") != "ok"
    # toggle context exists
    with wrapped.disable_adapter():
        pass


# ---------------------------------------------------------------------------
# WS4.a — GPT-2 nonstandard naming
# ---------------------------------------------------------------------------


@pytest.mark.peft
def test_gpt2_wte_nonstandard_downgrade():
    model = _tiny_gpt2()
    info = inspect_embeddings(model)
    names = info.get("embedding_layer_names") or {}
    assert names.get("input") == "transformer.wte"
    assert names.get("conventional") is False
    assert "plain_frozen_embedding" not in (info.get("mechanisms") or [])
    assert "plain_frozen_embedding" not in (info.get("input_mechanisms") or [])
    assert info.get("embedding_verification") == "verification_unknown"


# ---------------------------------------------------------------------------
# WS4.b — base-equivalence guard (cheap, in-process)
# ---------------------------------------------------------------------------


@pytest.mark.peft
def test_base_equivalence_guard_plain_lora(tokenizer):
    model = _tiny_llama(tie=True)
    ids = torch.randint(0, 256, (1, 8))
    wrapped = _wrap(model)
    _train_steps(wrapped, ids)
    guard = base_equivalence_guard(wrapped, tokenizer, probe_texts=["hello world", "abc 123"])
    assert guard["restored"] is True
    assert guard["adapter_active"] is True
    assert guard["verdict"] == "pass"
    ok, _ = adapter_toggle_safe(inspect_embeddings(wrapped))
    assert ok is True
