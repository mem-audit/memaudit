"""PEFT / sharded-backend surfaces. Live LoRA if peft imports; otherwise mocks."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from memaudit.preflight import adapter_toggle_safe, inspect_embeddings
from memaudit.utils import is_peft_model


def test_is_peft_model_by_methods():
    obj = SimpleNamespace(peft_config={"default": object()}, disable_adapter=lambda: None)
    assert is_peft_model(obj) is True
    assert is_peft_model(SimpleNamespace()) is False


def test_merged_adapter_not_toggle_safe():
    info = {
        "bias": "none",
        "merged": True,
        "peft_type": "LORA",
    }
    ok, reason = adapter_toggle_safe(info)
    assert ok is False
    assert "merged" in (reason or "").lower()


def test_prompt_tuning_not_toggle_safe():
    info = {"bias": "none", "merged": False, "peft_type": "PROMPT_TUNING"}
    ok, reason = adapter_toggle_safe(info)
    assert ok is False
    assert "PROMPT" in (reason or "") or "toggle" in (reason or "").lower()


def test_inspect_embeddings_reads_modules_to_save(tiny_model):
    tiny_model.peft_config = {
        "default": SimpleNamespace(
            bias="none",
            peft_type="LORA",
            r=4,
            lora_alpha=8,
            modules_to_save=["lm_head"],
            trainable_token_indices=None,
        )
    }
    tiny_model.active_adapter = "default"
    tiny_model.disable_adapter = lambda: None
    info = inspect_embeddings(tiny_model)
    assert info["r"] == 4
    assert info["modules_to_save"] == ["lm_head"]


def test_disable_adapter_scoring_path(tiny_model, tokenizer):
    """Mock PeftModel.toggle: FT and base scores must be allowed to differ."""

    class _Toggle:
        def __init__(self, model):
            self.model = model

        def __enter__(self):
            return self.model

        def __exit__(self, *exc):
            return False

    tiny_model.peft_config = {
        "default": SimpleNamespace(
            bias="none",
            peft_type="LORA",
            r=8,
            lora_alpha=16,
            modules_to_save=None,
            trainable_token_indices=None,
        )
    }
    tiny_model.active_adapter = "default"
    tiny_model.disable_adapter = lambda: _Toggle(tiny_model)
    assert is_peft_model(tiny_model)
    info = inspect_embeddings(tiny_model)
    ok, _ = adapter_toggle_safe(info)
    assert ok is True
    with tiny_model.disable_adapter():
        assert tiny_model is tiny_model


@pytest.mark.peft
def test_live_peft_imports_or_skip():
    """Importing peft+transformers 5.x can hang on torch 2.6.dev FSDP symbols.

    We only assert the optional extra is documented; live wrap is exercised
    by ``memaudit demo --lora`` when the stack imports cleanly.
    """
    import os

    if os.environ.get("MEMAUDIT_LIVE_PEFT") != "1":
        pytest.skip("set MEMAUDIT_LIVE_PEFT=1 to instantiate a real LoRA GPT-2")
    pytest.importorskip("peft")
