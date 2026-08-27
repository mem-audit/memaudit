"""Live-model pre-flight. Inspect ``trainer.model``, not the user-supplied config.

TRL mutates ``peft_config`` (trainable_token_indices / modules_to_save) after
chat-template token adds. Callback-time injection is impossible - this module
only *verifies* that pre-train ``inject()`` survived preprocessing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memaudit.constants import MIN_CONTROLS_FOR_TPR_AT_1PCT
from memaudit.exceptions import MemauditPreflightError
from memaudit.utils import encode_ids, example_text, find_subsequence, is_peft_model


def _peft_cfg(model: Any) -> Any | None:
    cfgs = getattr(model, "peft_config", None)
    if not cfgs:
        return None
    if isinstance(cfgs, dict):
        adapter = getattr(model, "active_adapter", None)
        if adapter is not None and adapter in cfgs:
            return cfgs[adapter]
        if cfgs:
            return next(iter(cfgs.values()))
    return cfgs


def _wrapper_names(module: Any) -> set[str]:
    names: set[str] = set()
    for cls in type(module).__mro__:
        names.add(cls.__name__)
    return names


def inspect_embeddings(model: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "trainable": False,
        "input_trainable": False,
        "output_trainable": False,
        "mechanisms": [],
        "tie_word_embeddings": None,
        "modules_to_save": None,
        "trainable_token_indices": None,
        "bias": None,
        "peft_type": None,
        "r": None,
        "lora_alpha": None,
        "merged": None,
        "status_enabled": None,
    }
    if model is None:
        return info
    cfg = getattr(model, "config", None)
    if cfg is not None:
        info["tie_word_embeddings"] = getattr(cfg, "tie_word_embeddings", None)

    peft_cfg = _peft_cfg(model)
    if peft_cfg is not None:
        info["bias"] = getattr(peft_cfg, "bias", None)
        info["peft_type"] = str(getattr(peft_cfg, "peft_type", None))
        info["r"] = getattr(peft_cfg, "r", None)
        info["lora_alpha"] = getattr(peft_cfg, "lora_alpha", None)
        info["modules_to_save"] = list(getattr(peft_cfg, "modules_to_save", None) or []) or None
        info["trainable_token_indices"] = getattr(peft_cfg, "trainable_token_indices", None)

    emb = getattr(model, "get_input_embeddings", lambda: None)()
    out = getattr(model, "get_output_embeddings", lambda: None)()
    if emb is not None:
        names = _wrapper_names(emb)
        if "ModulesToSaveWrapper" in names:
            info["mechanisms"].append("ModulesToSaveWrapper")
        if "TrainableTokensWrapper" in names:
            info["mechanisms"].append("TrainableTokensWrapper")
        if "BaseTunerLayer" in names or "lora.Embedding" in str(type(emb)) or "LoraLayer" in names:
            if "Embedding" in type(emb).__name__ or hasattr(emb, "lora_embedding_A"):
                info["mechanisms"].append("lora.Embedding")
        try:
            info["input_trainable"] = any(bool(p.requires_grad) for p in emb.parameters())
        except Exception:
            info["input_trainable"] = False
        if not info["mechanisms"] and not info["input_trainable"]:
            info["mechanisms"].append("plain_frozen_embedding")
    if out is not None:
        try:
            info["output_trainable"] = any(bool(p.requires_grad) for p in out.parameters())
        except Exception:
            info["output_trainable"] = False
    info["trainable"] = bool(info["input_trainable"] or info["output_trainable"])

    if is_peft_model(model):
        # Do not import peft here: a broken transformers/torch pair can hang
        # on FSDP symbols. Use peft only if the user already loaded it.
        import sys

        peft_mod = sys.modules.get("peft")
        get_status = getattr(peft_mod, "get_model_status", None) if peft_mod is not None else None
        if callable(get_status):
            try:
                status = get_status(model)
                info["merged"] = getattr(status, "merged", None)
                info["status_enabled"] = getattr(status, "enabled", None)
            except Exception:
                info["merged"] = getattr(model, "merged", None)
        else:
            info["merged"] = getattr(model, "merged", None)
    return info


def inspect_training_args(args: Any | None) -> dict[str, Any]:
    if args is None:
        return {}
    packing = getattr(args, "packing_strategy", None)
    packing_flag = getattr(args, "packing", None)
    return {
        "packing_strategy": packing,
        "packing": packing_flag,
        "max_length": getattr(args, "max_length", None) or getattr(args, "max_seq_length", None),
        "completion_only_loss": getattr(args, "completion_only_loss", None),
        "assistant_only_loss": getattr(args, "assistant_only_loss", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "num_train_epochs": getattr(args, "num_train_epochs", None),
        "output_dir": getattr(args, "output_dir", None),
        "deepspeed": getattr(args, "deepspeed", None),
        "fsdp": getattr(args, "fsdp", None) or getattr(args, "fsdp_config", None),
    }


def is_sharded_backend(trainer: Any) -> tuple[bool, str]:
    """Return (should_defer_in_callback, reason). ZeRO-3 / FSDP only."""
    if trainer is None:
        return False, ""
    acc = getattr(trainer, "accelerator", None)
    if acc is None:
        return False, ""
    dt = str(getattr(acc, "distributed_type", "") or "").upper()
    if "FSDP" in dt:
        return True, f"accelerator.distributed_type={dt}"
    if "DEEPSPEED" in dt or "DEEP_SPEED" in dt:
        plugin = getattr(getattr(acc, "state", None), "deepspeed_plugin", None) or getattr(
            acc, "deepspeed_plugin", None
        )
        stage = None
        if plugin is not None:
            stage = getattr(plugin, "zero_stage", None)
            if stage is None:
                cfg = getattr(plugin, "deepspeed_config", None) or getattr(plugin, "config", None)
                if isinstance(cfg, dict):
                    stage = (cfg.get("zero_optimization") or {}).get("stage")
        if stage is None or int(stage) >= 3:
            return True, f"DeepSpeed ZeRO stage={stage} (treat missing as 3)"
    return False, ""


def adapter_toggle_safe(emb_info: dict[str, Any]) -> tuple[bool, str | None]:
    bias = emb_info.get("bias")
    if bias not in (None, "none"):
        return False, (
            f"peft bias={bias!r}: disable_adapter() does not restore a true base model. "
            "Falling back to a separately loaded reference (or target-only scores)."
        )
    if emb_info.get("merged"):
        return False, (
            "Adapter appears merged (merge_and_unload / merge_adapter). "
            "disable_adapter() is gone - pass a base checkpoint as --ref."
        )
    ptype = str(emb_info.get("peft_type") or "").upper()
    if ptype and "LORA" not in ptype and ptype not in {"NONE", "NONE"}:
        # prompt-learning / IA3 etc. - toggle path is different
        if "PROMPT" in ptype or "PREFIX" in ptype or "P_TUNING" in ptype or "ADALORA" not in ptype:
            if "LORA" not in ptype:
                return False, (
                    f"peft_type={ptype}: adapter-toggle scoring is scoped to LoRA-family in v0.1."
                )
    return True, None


def _dataset_rows(dataset: Any, limit: int = 50_000) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    if dataset is None:
        return rows
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        try:
            n = min(len(dataset), limit)
            for i in range(n):
                row = dataset[i]
                if isinstance(row, Mapping):
                    rows.append(row)
            return rows
        except Exception:
            pass
    from memaudit.utils import iter_examples

    return list(iter_examples(dataset, limit=limit))


def _row_ids_and_labels(row: Mapping[str, Any], tokenizer: Any | None) -> tuple[list[int] | None, list[int] | None]:
    ids = row.get("input_ids")
    labels = row.get("labels")
    if ids is not None:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = [int(x) for x in ids]
        if labels is not None:
            if hasattr(labels, "tolist"):
                labels = labels.tolist()
            labels = [int(x) for x in labels]
        return ids, labels
    return None, None


def survival_scan(
    dataset: Any,
    manifest: dict[str, Any],
    tokenizer: Any | None = None,
    limit: int = 50_000,
) -> dict[str, Any]:
    """Prove inserted canary secrets still appear (token-level, string fallback)."""
    fmt = manifest.get("fmt") or "text"
    inserted = [c for c in manifest.get("canaries") or [] if c.get("included")]
    rows = _dataset_rows(dataset, limit=limit)
    found = 0
    masked_out = 0
    missing_ids: list[str] = []
    token_hits = 0
    string_hits = 0

    tokenized_rows: list[tuple[list[int], list[int] | None]] = []
    string_blobs: list[str] = []
    for row in rows:
        ids, labels = _row_ids_and_labels(row, tokenizer)
        if ids is not None:
            tokenized_rows.append((ids, labels))
        string_blobs.append(example_text(row, fmt if "text" in row or "messages" in row or "prompt" in row else "text"))

    for canary in inserted:
        cid = canary.get("id", "?")
        secret = canary.get("secret") or ""
        needle = list(canary.get("secret_token_ids") or [])
        hit = False
        labels_all_ignore = False
        for ids, labels in tokenized_rows:
            loc = find_subsequence(ids, needle) if needle else None
            if loc:
                hit = True
                token_hits += 1
                if labels is not None:
                    sl = labels[loc[0] : loc[1]]
                    if sl and all(int(x) == -100 for x in sl):
                        labels_all_ignore = True
                break
        if not hit and secret:
            for blob in string_blobs:
                if secret and secret in blob:
                    hit = True
                    string_hits += 1
                    break
        if hit:
            found += 1
            if labels_all_ignore:
                masked_out += 1
        else:
            missing_ids.append(cid)

    n = len(inserted)
    return {
        "n_inserted": n,
        "n_found": found,
        "n_missing": n - found,
        "n_fully_masked": masked_out,
        "token_level_hits": token_hits,
        "string_level_hits": string_hits,
        "missing_ids": missing_ids[:20],
        "rows_scanned": len(rows),
    }


def run_preflight(
    *,
    model: Any,
    trainer: Any | None,
    manifest: dict[str, Any],
    tokenizer: Any | None,
    args: Any | None = None,
    train_dataset: Any | None = None,
    raise_fatal: bool = True,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    warnings: list[str] = []
    fatal: list[str] = []

    emb = inspect_embeddings(model)
    train = inspect_training_args(args if args is not None else getattr(trainer, "args", None))

    toggle_ok, toggle_reason = adapter_toggle_safe(emb)
    if toggle_reason:
        warnings.append(toggle_reason)
        findings.append({"code": "adapter_toggle", "level": "warning", "message": toggle_reason})

    if train.get("packing_strategy") == "wrapped" or (
        train.get("packing") and train.get("packing_strategy") == "wrapped"
    ):
        msg = (
            "packing_strategy='wrapped' splits records mid-sequence and trains with "
            "cross-document attention. Audit scores will be biased low. Prefer bfd."
        )
        warnings.append(msg)
        findings.append({"code": "packing_wrapped", "level": "warning", "message": msg})

    if train.get("packing"):
        warnings.append(
            "Packing is on: first token of each packed document has labels=-100. "
            "Scoring skips the first record token. Non-FlashAttention packing may "
            "cross-contaminate documents (TRL warning) - treat scores as slightly biased."
        )

    max_len = train.get("max_length")
    if max_len:
        too_long = []
        fmt = manifest.get("fmt") or "text"
        for c in manifest.get("canaries") or []:
            if not c.get("included"):
                continue
            n_tok = len(c.get("secret_token_ids") or [])
            if tokenizer is not None:
                try:
                    from memaudit.injection import canary_record
                    from memaudit.types import Canary

                    rec = canary_record(Canary.from_dict(c), fmt)
                    blob = example_text(rec, fmt)
                    n_tok = len(encode_ids(tokenizer, blob, add_special_tokens=False))
                except Exception:
                    pass
            if n_tok and n_tok > int(max_len):
                too_long.append(c.get("id"))
        if too_long:
            msg = (
                f"{len(too_long)} canary *records* (prefix + secret) exceed "
                f"max_length={max_len} and will be truncated (keep_start). "
                "This silently zeros those canaries."
            )
            fatal.append(msg)

    n_ctrl = sum(1 for c in (manifest.get("canaries") or []) if not c.get("included"))
    n_mem = sum(1 for c in (manifest.get("canaries") or []) if c.get("included"))
    if n_mem > 0 and n_ctrl < MIN_CONTROLS_FOR_TPR_AT_1PCT:
        msg = (
            f"Only {n_ctrl} held-out controls; TPR@1% FPR requires "
            f">={MIN_CONTROLS_FOR_TPR_AT_1PCT}. The report will refuse that "
            "headline rather than invent a precise detection rate."
        )
        warnings.append(msg)
        findings.append({"code": "tpr_underpowered", "level": "warning", "message": msg})

    fmt = manifest.get("fmt")
    if train.get("completion_only_loss") and fmt == "text":
        warnings.append(
            "completion_only_loss is set but canaries were injected as 'text' records. "
            "If the live dataset is prompt+completion, re-inject with fmt='prompt_completion'."
        )
    if train.get("assistant_only_loss") and fmt not in {None, "messages"}:
        warnings.append(
            "assistant_only_loss=True but canaries are not in messages format. "
            "User-turn / prompt secrets would be labeled -100."
        )

    dataset = train_dataset
    if dataset is None and trainer is not None:
        dataset = getattr(trainer, "train_dataset", None)
    scan = survival_scan(dataset, manifest, tokenizer=tokenizer)
    if scan["n_inserted"] == 0:
        fatal.append(
            "No canaries were marked included in the manifest. inject() coin-flips may "
            "have excluded every candidate, or you passed controls only. The audit "
            "would be silently empty."
        )
    elif scan["n_found"] == 0:
        fatal.append(
            f"0 of {scan['n_inserted']} inserted canary secrets were found in the training "
            "dataset. Canary injection cannot happen inside the Trainer callback - the "
            "dataloader is built (and TRL tokenizes/packs) before any hook fires. Call "
            "memaudit.inject() on the RAW dataset BEFORE constructing Trainer/SFTTrainer."
        )
    elif scan["n_missing"] > 0:
        warnings.append(
            f"{scan['n_missing']} inserted canaries were not found in the scanned dataset "
            f"(missing ids: {scan['missing_ids']}). Truncation or a formatting_func may "
            "have dropped them."
        )
    if scan["n_fully_masked"] > 0:
        fatal.append(
            f"{scan['n_fully_masked']} canaries appear in input_ids but every secret token "
            "has labels=-100 (completion_only_loss / assistant_only_loss / packing). "
            "The secret must live in the completion / assistant turn / text body - never "
            "the prompt or user turn. The audit would be silently zeroed."
        )

    # new-token family leftover
    families = {c.get("family") for c in manifest.get("canaries") or []}
    if "new_token" in families and not emb.get("trainable"):
        fatal.append(
            "Manifest contains new_token canaries but embeddings are not trainable. "
            "That family is unimplemented/gated in v0.1 and would measure noise."
        )

    result = {
        "embeddings": emb,
        "training": {
            **train,
            "epochs": train.get("num_train_epochs"),
            "learning_rate": train.get("learning_rate"),
        },
        "adapter_toggle_safe": toggle_ok,
        "survival": scan,
        "findings": findings,
        "warnings": warnings,
        "fatal": fatal,
    }
    if fatal and raise_fatal:
        raise MemauditPreflightError(" ".join(fatal))
    return result


def longest_canary_tokens(manifest: dict[str, Any]) -> int:
    m = 0
    for c in manifest.get("canaries") or []:
        m = max(m, len(c.get("secret_token_ids") or []))
    return m


def inserted_canaries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in (manifest.get("canaries") or []) if c.get("included")]


def control_canaries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in (manifest.get("canaries") or []) if not c.get("included")]
