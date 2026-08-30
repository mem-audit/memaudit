"""Live QLoRA validation against a real bitsandbytes 4-bit base.

test_peft_matrix.py row 6 covers quantization detection with a named stub only
(dev machines without bitsandbytes). These tests execute the real thing: a
config-built tiny Llama is saved, reloaded through
``BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")``,
LoRA-wrapped, and driven through quantization detection, the full-precision-ref
mismatch guard, the base-equivalence guard, and a full ``run_audit()`` on
Linear4bit forwards.

Opt in:  pytest -m integration tests/test_qlora_live.py
Requires a platform where bitsandbytes imports and 4-bit CPU inference works
(multi-backend bitsandbytes >= 0.45 on Linux; not Apple Silicon natively).
"""

from __future__ import annotations

import platform
from types import SimpleNamespace

import pytest
import torch

from memaudit.audit import run_audit
from memaudit.canaries import generate_canaries
from memaudit.constants import HEADLINE_ATTACK, HEADLINE_ATTACK_FALLBACK
from memaudit.injection import inject
from memaudit.peft_semantics import (
    base_equivalence_guard,
    capture_disabled_logits,
    detect_quantization,
    quantization_ref_mismatch,
    unusual_peft_triggers,
)
from memaudit.preflight import adapter_toggle_safe, inspect_embeddings

pytestmark = [pytest.mark.integration, pytest.mark.peft]

bnb = pytest.importorskip("bitsandbytes")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")


def _tiny_llama_config(vocab: int = 256):
    from transformers import LlamaConfig

    return LlamaConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        vocab_size=vocab,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        rms_norm_eps=1e-5,
    )


def _bnb_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
    )


def _load_4bit(ckpt: str):
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(ckpt, quantization_config=_bnb_config())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"bitsandbytes {bnb.__version__} cannot execute a 4-bit load on "
            f"{platform.system()}/{platform.machine()}: {exc!r}"
        )


@pytest.fixture(scope="module")
def qlora(tmp_path_factory):
    """Full-precision base + the same checkpoint reloaded 4-bit and LoRA-wrapped."""
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaForCausalLM

    torch.manual_seed(0)
    fp_base = LlamaForCausalLM(_tiny_llama_config())
    fp_base.eval()
    ckpt = str(tmp_path_factory.mktemp("tiny-llama-4bit"))
    fp_base.save_pretrained(ckpt)

    q_base = _load_4bit(ckpt)
    model = get_peft_model(
        q_base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        ),
    )
    # lora_B is zero-init; perturb it so the adapter is active without a backward pass
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0.0, 0.5)
    model.eval()
    return SimpleNamespace(fp_base=fp_base, model=model, ckpt=ckpt)


def _inject_manifest(tokenizer):
    host = [{"text": f"ordinary training sentence number {i} about weather"} for i in range(12)]
    cans = generate_canaries(tokenizer, n=2, n_controls=3, family="random", seed=0, secret_len=25)
    return inject(host, cans, fmt="text", seed=0, include_prob=1.0)


# ---------------------------------------------------------------------------
# 1. quantization detection on a real 4-bit model
# ---------------------------------------------------------------------------


def test_quantization_detection_live(qlora):
    det = detect_quantization(qlora.model)
    assert det["quantized"] is True
    assert "is_loaded_in_4bit" in det["kinds"]
    assert "quantization_config" in det["kinds"]
    assert "Linear4bit" in det["kinds"]
    # quant_method value, not the enum repr "QuantizationMethod.BITS_AND_BYTES"
    assert "bitsandbytes" in det["kinds"]

    info = inspect_embeddings(qlora.model)
    assert info["quantized"] is True
    assert info["quantization_kinds"] == det["kinds"]
    # embeddings stay unquantized and conventionally named on a QLoRA Llama
    assert info["embedding_verification"] == "ok"
    assert "quantized_base" in unusual_peft_triggers(info)
    ok, reason = adapter_toggle_safe(info)
    assert ok is True, reason

    assert detect_quantization(qlora.fp_base) == {"quantized": False, "kinds": []}


# ---------------------------------------------------------------------------
# 2. full-precision --ref vs quantized target: warn + downgrade, never silent
# ---------------------------------------------------------------------------


def test_fp_ref_mismatch_warns_and_downgrades(qlora, tokenizer):
    info = inspect_embeddings(qlora.model)
    mismatch, msg = quantization_ref_mismatch(info, qlora.fp_base)
    assert mismatch is True
    assert msg and "full-precision" in msg
    matching, _ = quantization_ref_mismatch(info, qlora.model)
    assert matching is False

    ds, manifest = _inject_manifest(tokenizer)
    report = run_audit(
        model=qlora.model,
        tokenizer=tokenizer,
        manifest=manifest,
        dataset=ds,
        ref=qlora.fp_base,
        real_sample=0,
        skip_generation=True,
    )
    ref_block = report["reference"]
    assert ref_block["quantization_mismatch"] is True
    assert ref_block["downgraded_from"] == "provided_object"
    # exact toggle is available, so ref-mode downgrades to disable_adapter
    assert ref_block["mode"] == "disable_adapter"
    assert report["membership"]["headline_attack"] == HEADLINE_ATTACK
    assert any("full-precision" in w for w in report["audit_warnings"])


def test_fp_ref_on_bare_quantized_target_goes_target_only(qlora, tokenizer):
    """No adapter toggle available: the fp ref must be refused, not silently used."""
    bare = _load_4bit(qlora.ckpt)
    _ds, manifest = _inject_manifest(tokenizer)
    report = run_audit(
        model=bare,
        tokenizer=tokenizer,
        manifest=manifest,
        dataset=None,
        ref=qlora.fp_base,
        real_sample=0,
        skip_generation=True,
    )
    assert report["reference"]["mode"] == "target_only"
    assert report["reference"]["quantization_mismatch"] is True
    assert report["membership"]["headline_attack"] == HEADLINE_ATTACK_FALLBACK
    assert any("Quantization mismatch" in w for w in report["audit_warnings"])


# ---------------------------------------------------------------------------
# 3. base-equivalence guard on the quantized base
# ---------------------------------------------------------------------------


def test_base_equivalence_guard_quantized_toggle_bit_exact(qlora, tokenizer):
    guard = base_equivalence_guard(qlora.model, tokenizer, probe_texts=["hello world", "abc 123"])
    assert guard["adapter_active"] is True
    assert guard["restored"] is True
    assert guard["compared"] is False
    assert guard["verdict"] == "not_run"

    capture = capture_disabled_logits(qlora.model, tokenizer)
    assert capture is not None
    replay = base_equivalence_guard(qlora.model, tokenizer, captured=capture)
    # in-process disable_adapter() on the quantized base is bit-exact
    assert replay["max_abs_logit_diff"] == 0.0
    assert replay["verdict"] == "pass"


def test_base_equivalence_guard_refuses_quantized_vs_fp(qlora, tokenizer):
    capture = capture_disabled_logits(qlora.model, tokenizer)
    assert capture is not None
    fp_logits = []
    with torch.inference_mode():
        for ids in capture["probe_ids"]:
            t = torch.tensor([list(ids)], dtype=torch.long)
            fp_logits.append(qlora.fp_base(input_ids=t).logits.float().tolist())
    fp_capture = {
        "probe_texts": capture["probe_texts"],
        "probe_ids": capture["probe_ids"],
        "logits": fp_logits,
    }
    guard = base_equivalence_guard(qlora.model, tokenizer, captured=fp_capture)
    # nf4 quantization error is orders of magnitude above the 1e-5 WARN headroom
    assert guard["max_abs_logit_diff"] > guard["atol_warn"]
    assert guard["verdict"] == "fail"
    assert guard["restored"] is True


# ---------------------------------------------------------------------------
# 4. end-to-end audit on Linear4bit forwards (toggle ref-mode)
# ---------------------------------------------------------------------------


def test_run_audit_end_to_end_on_4bit(qlora, tokenizer, tmp_path):
    ds, manifest = _inject_manifest(tokenizer)
    report = run_audit(
        model=qlora.model,
        tokenizer=tokenizer,
        manifest=manifest,
        dataset=ds,
        ref="auto",
        real_sample=4,
        output_path=tmp_path / "memaudit-report.json",
        skip_generation=False,
    )
    assert (tmp_path / "memaudit-report.json").is_file()
    assert report["reference"]["mode"] == "disable_adapter"
    assert report["preflight"]["base_equivalence"]["verdict"] == "not_run"
    assert report["membership"]["headline_attack"] == HEADLINE_ATTACK
    assert report["membership"]["n_members"] == 2
    assert report["membership"]["n_controls"] == 3
    assert report["adapter"]["quantized"] is True
    assert report["regurgitation"]["overall"]["n"] == 2
    for row in report["per_canary"]:
        assert row["scores"]["masked_nll"] == row["scores"]["masked_nll"]  # finite, not NaN
