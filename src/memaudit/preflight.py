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
from memaudit.preflight_labels import (
    AlignmentHit,
    CanaryEvidence,
    align_in_ids,
    align_straddle,
    context_needles,
    effective_labels,
    finalize_evidence,
    mask_column_missing,
    row_has_text_keys,
    row_input_ids,
)
from memaudit.peft_semantics import (
    capture_disabled_logits,
    inspect_peft_embeddings,
    trainable_token_canary_findings,
)
from memaudit.utils import (
    dataset_length,
    decode_ids,
    encode_ids,
    example_text,
    is_peft_model,
)


def inspect_embeddings(model: Any) -> dict[str, Any]:
    """Inspect live embedding / adapter state. See ``peft_semantics`` for the matrix."""
    return inspect_peft_embeddings(model)


def _first_row(dataset: Any) -> Mapping[str, Any] | None:
    if dataset is None:
        return None
    try:
        row = dataset[0]
        if isinstance(row, Mapping):
            return row
    except Exception:
        return None
    return None


def _sniff_completion_only_loss(dataset: Any) -> bool | None:
    row = _first_row(dataset)
    if row is None:
        return None
    return "prompt" in row and "completion" in row


def _dataset_kwargs(args: Any | None) -> dict[str, Any]:
    if args is None:
        return {}
    raw = getattr(args, "dataset_kwargs", None)
    if isinstance(raw, Mapping):
        return dict(raw)
    if raw is None:
        return {}
    out: dict[str, Any] = {}
    for key in ("skip_prepare_dataset",):
        if hasattr(raw, key):
            out[key] = getattr(raw, key)
    return out


def inspect_training_args(
    args: Any | None,
    trainer: Any | None = None,
    dataset: Any | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Read live trainer/args/dataset. Record effective values; never invent them."""
    if args is None and trainer is None:
        return {}
    if args is None:
        args = getattr(trainer, "args", None)
    packing = getattr(args, "packing_strategy", None) if args is not None else None
    packing_flag = getattr(args, "packing", None) if args is not None else None
    col_args = getattr(args, "completion_only_loss", None) if args is not None else None
    asst_args = getattr(args, "assistant_only_loss", None) if args is not None else None

    effective_col = None
    if trainer is not None:
        effective_col = getattr(trainer, "completion_only_loss", None)
        if effective_col is None:
            collator = getattr(trainer, "data_collator", None)
            effective_col = getattr(collator, "completion_only_loss", None)
    if effective_col is None and col_args is not None:
        effective_col = col_args
    if effective_col is None:
        sniffed = _sniff_completion_only_loss(dataset)
        if sniffed is not None:
            effective_col = sniffed

    padding_free = getattr(args, "padding_free", None) if args is not None else None
    padding_free_forced = False
    if trainer is not None and getattr(trainer, "padding_free", None) is not None:
        padding_free = getattr(trainer, "padding_free")
    if packing_flag and packing in (None, "bfd") and not padding_free:
        padding_free = True
        padding_free_forced = True

    skip_prepare = bool(_dataset_kwargs(args).get("skip_prepare_dataset"))
    use_liger = getattr(args, "use_liger_kernel", None) if args is not None else None

    first = _first_row(dataset)
    columns = list(first.keys()) if first is not None else None
    pretokenized = bool(first is not None and "input_ids" in first)

    tok = tokenizer
    if tok is None and trainer is not None:
        tok = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    tmpl = getattr(tok, "chat_template", None) if tok is not None else None
    has_generation = None
    if isinstance(tmpl, str) and tmpl:
        has_generation = "{% generation %}" in tmpl

    return {
        "packing_strategy": packing,
        "packing": packing_flag,
        "max_length": (
            getattr(args, "max_length", None) or getattr(args, "max_seq_length", None)
            if args is not None
            else None
        ),
        "completion_only_loss": col_args,
        "completion_only_loss_effective": effective_col,
        "assistant_only_loss": asst_args,
        "chat_template_has_generation_markers": has_generation,
        "skip_prepare_dataset": skip_prepare,
        "padding_free": padding_free,
        "padding_free_forced_by_bfd": padding_free_forced,
        "use_liger_kernel": use_liger,
        "pretokenized": pretokenized,
        "dataset_columns": columns,
        "learning_rate": getattr(args, "learning_rate", None) if args is not None else None,
        "num_train_epochs": getattr(args, "num_train_epochs", None) if args is not None else None,
        "output_dir": getattr(args, "output_dir", None) if args is not None else None,
        "deepspeed": getattr(args, "deepspeed", None) if args is not None else None,
        "fsdp": (
            getattr(args, "fsdp", None) or getattr(args, "fsdp_config", None)
            if args is not None
            else None
        ),
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


def _prepare_scan_rows(
    rows: list[Mapping[str, Any]],
    tokenizer: Any | None,
    fmt: str,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        ids = row_input_ids(row, tokenizer, fmt)
        decoded = None
        if ids is not None and tokenizer is not None:
            try:
                decoded = decode_ids(tokenizer, ids, skip_special_tokens=True)
            except Exception:
                decoded = None
        use_fmt = fmt if row_has_text_keys(row) else "text"
        blob = example_text(row, use_fmt)
        prepared.append({"row": row, "ids": ids, "decoded": decoded, "blob": blob})
    return prepared


def _best_hit(hits: list[AlignmentHit]) -> AlignmentHit:
    rank = {"exact": 0, "covering": 1, "straddle": 2, "string": 3}
    return min(hits, key=lambda h: (rank.get(h.alignment, 9), 0 if h.span else 1))


def _scan_one_canary(
    canary: Mapping[str, Any],
    prepared: list[dict[str, Any]],
    *,
    trainer: Any | None,
    config: Mapping[str, Any],
    tokenizer: Any | None,
    fmt: str,
    scan_complete: bool,
    rows_total_known: bool,
) -> CanaryEvidence:
    cid = str(canary.get("id") or "?")
    secret = str(canary.get("secret") or "")
    needles = context_needles(canary, tokenizer, fmt)
    hits: list[AlignmentHit] = []
    for idx, prep in enumerate(prepared):
        ids = prep["ids"]
        aligned = align_in_ids(ids, needles, secret, tokenizer, prep["decoded"])
        if aligned:
            span, how = aligned
            hits.append(
                AlignmentHit(
                    row=prep["row"],
                    ids=ids,
                    span=span,
                    alignment=how,
                    row_index=idx,
                )
            )
            continue
        blob = prep["blob"] or ""
        decoded = prep["decoded"] or ""
        if secret and (secret in blob or secret in decoded):
            cover = None
            if ids is not None and tokenizer is not None:
                from memaudit.preflight_labels import char_cover_span

                cover = char_cover_span(ids, secret, tokenizer)
            if cover:
                hits.append(
                    AlignmentHit(
                        row=prep["row"],
                        ids=ids,
                        span=cover,
                        alignment="covering",
                        row_index=idx,
                    )
                )
            else:
                stream_missing = ids is not None and not (
                    secret and decoded and secret in decoded
                )
                hits.append(
                    AlignmentHit(
                        row=prep["row"],
                        ids=ids,
                        span=None,
                        alignment="string",
                        row_index=idx,
                        token_stream_missing=stream_missing,
                    )
                )

    if not hits:
        for idx in range(len(prepared) - 1):
            left, right = prepared[idx], prepared[idx + 1]
            if not left["ids"] or not right["ids"]:
                continue
            loc = align_straddle(left["ids"], right["ids"], needles)
            if loc:
                hits.append(
                    AlignmentHit(
                        row=left["row"],
                        ids=list(left["ids"]) + list(right["ids"]),
                        span=loc,
                        alignment="straddle",
                        straddle=True,
                        partner_row=right["row"],
                        row_index=idx,
                    )
                )
                break

    ev = CanaryEvidence(id=cid)
    hit = _best_hit(hits) if hits else None
    labels_result = None
    if hit is not None:
        labels_result = effective_labels(hit.row, trainer, config)
        if labels_result is None and hit.partner_row is not None:
            labels_result = effective_labels(hit.partner_row, trainer, config)
        if mask_column_missing(hit.row, config, fmt):
            ev.reasons.append(
                "configured masking is not present in the prepared data; "
                "effective supervision is full-sequence; audit semantics "
                "differ from the recorded protocol"
            )
    custom_loss = bool(trainer is not None and getattr(trainer, "compute_loss_func", None))
    return finalize_evidence(
        ev,
        hit,
        labels_result,
        custom_loss=custom_loss,
        scan_complete=scan_complete,
        rows_total_known=rows_total_known,
    )


def survival_scan(
    dataset: Any,
    manifest: dict[str, Any],
    tokenizer: Any | None = None,
    limit: int = 50_000,
    trainer: Any | None = None,
    args: Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Scan inserted canaries and record per-canary evidence levels + coverage."""
    fmt = manifest.get("fmt") or "text"
    inserted = [c for c in manifest.get("canaries") or [] if c.get("included")]
    rows = _dataset_rows(dataset, limit=limit)
    rows_total = dataset_length(dataset)
    scan_complete = rows_total is not None and len(rows) >= int(rows_total)
    train_cfg = dict(config) if config is not None else inspect_training_args(
        args if args is not None else getattr(trainer, "args", None),
        trainer=trainer,
        dataset=dataset,
        tokenizer=tokenizer,
    )
    prepared = _prepare_scan_rows(rows, tokenizer, fmt)

    per_canary: list[dict[str, Any]] = []
    found = 0
    masked_out = 0
    token_hits = 0
    string_hits = 0
    missing_ids: list[str] = []
    n_directly = 0
    n_mask_verified = 0
    n_unknown = 0
    n_deleted = 0
    mask_missing_ids: list[str] = []
    stream_missing_ids: list[str] = []

    for canary in inserted:
        ev = _scan_one_canary(
            canary,
            prepared,
            trainer=trainer,
            config=train_cfg,
            tokenizer=tokenizer,
            fmt=fmt,
            scan_complete=scan_complete,
            rows_total_known=rows_total is not None,
        )
        if ev.record_observed:
            found += 1
            if ev.alignment in {"exact", "covering"}:
                token_hits += 1
            elif ev.alignment == "string":
                string_hits += 1
            if ev.loss_mask_checked and ev.supervised_token_fraction == 0.0:
                masked_out += 1
            if any("configured masking is not present" in r for r in ev.reasons):
                mask_missing_ids.append(ev.id)
            if ev.token_stream_missing:
                stream_missing_ids.append(ev.id)
        else:
            missing_ids.append(ev.id)
            if ev.miss_reason == "not_found_after_complete_scan":
                n_deleted += 1
        if ev.directly_supervised:
            n_directly += 1
        if ev.loss_mask_checked:
            n_mask_verified += 1
        if ev.verification_unknown:
            n_unknown += 1
        per_canary.append(ev.to_dict())

    n = len(inserted)
    return {
        "n_inserted": n,
        "n_found": found,
        "n_missing": n - found,
        "n_fully_masked": masked_out,
        "n_directly_supervised": n_directly,
        "n_loss_mask_verified": n_mask_verified,
        "loss_mask_verified": n_mask_verified,
        "directly_supervised": n_directly,
        "n_verification_unknown": n_unknown,
        "n_row_deleted": n_deleted,
        "n_mask_column_missing": len(set(mask_missing_ids)),
        "mask_column_missing_ids": sorted(set(mask_missing_ids))[:20],
        "n_token_stream_missing": len(set(stream_missing_ids)),
        "token_stream_missing_ids": sorted(set(stream_missing_ids))[:20],
        "token_level_hits": token_hits,
        "string_level_hits": string_hits,
        "missing_ids": missing_ids[:20],
        "rows_scanned": len(rows),
        "rows_total": rows_total,
        "scan_complete": scan_complete,
        "per_canary": per_canary,
    }


def _canary_record_n_tokens(
    canary: Mapping[str, Any],
    fmt: str,
    tokenizer: Any,
) -> int:
    """Token count including chat-template / BOS-EOS overhead (not raw-blob only)."""
    from memaudit.injection import canary_record
    from memaudit.types import Canary

    rec = canary_record(Canary.from_dict(dict(canary)), fmt)
    if fmt == "messages":
        apply = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply):
            try:
                ids = apply(rec["messages"], tokenize=True, add_generation_prompt=False)
                # transformers >=5 returns a BatchEncoding dict by default
                # (return_dict flipped); len(dict) would undercount to ~2.
                if isinstance(ids, Mapping) or hasattr(ids, "keys"):
                    ids = ids["input_ids"]
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                if ids and isinstance(ids[0], (list, tuple)):
                    ids = ids[0]
                return len(ids)
            except Exception:
                pass
    blob = example_text(rec, fmt)
    return len(encode_ids(tokenizer, blob, add_special_tokens=True))


def absent_preflight(*, path: str = "post_hoc_cli") -> dict[str, Any]:
    """Honest block when preflight never ran (post-hoc CLI). Not a pass."""
    note = (
        "Preflight did not run. The Trainer callback runs these checks at "
        "on_train_begin; this path did not. Supervision and survival evidence "
        "is verification_unknown — this is not a pass."
    )
    return {
        "ran": False,
        "path": path,
        "verification_unknown": True,
        "note": note,
        "embeddings": {},
        "training": {},
        "adapter_toggle_safe": None,
        "survival": {
            "rows_total": None,
            "rows_scanned": None,
            "scan_complete": None,
            "loss_mask_verified": None,
            "directly_supervised": None,
        },
        "rows_total": None,
        "rows_scanned": None,
        "scan_complete": None,
        "loss_mask_verified": None,
        "directly_supervised": None,
        "findings": [],
        "warnings": [note],
        "fatal": [],
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
    scan_limit: int = 50_000,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    warnings: list[str] = []
    fatal: list[str] = []

    dataset = train_dataset
    if dataset is None and trainer is not None:
        dataset = getattr(trainer, "train_dataset", None)

    emb = inspect_embeddings(model)
    train = inspect_training_args(
        args if args is not None else getattr(trainer, "args", None),
        trainer=trainer,
        dataset=dataset,
        tokenizer=tokenizer,
    )

    toggle_ok, toggle_reason = adapter_toggle_safe(emb)
    if toggle_reason:
        warnings.append(toggle_reason)
        findings.append({"code": "adapter_toggle", "level": "warning", "message": toggle_reason})

    packing_strategy = train.get("packing_strategy")
    if packing_strategy == "wrapped" or (train.get("packing") and packing_strategy == "wrapped"):
        msg = (
            "packing_strategy='wrapped' splits records mid-sequence and trains with "
            "cross-document attention. Audit scores will be biased low. Prefer bfd."
        )
        warnings.append(msg)
        findings.append({"code": "packing_wrapped", "level": "warning", "message": msg})
    if packing_strategy in {"bfd-requeue", "bfd_split"}:
        msg = (
            f"packing_strategy={packing_strategy!r}: overflow is re-queued as detached "
            "fragments, so a secret may survive but be detached from its prefix. "
            "This strategy is not treated as wrapped and may not force padding-free."
        )
        warnings.append(msg)
        findings.append({"code": "packing_overflow_requeue", "level": "warning", "message": msg})

    if train.get("packing"):
        warnings.append(
            "Packing is on: first token of each packed document has labels=-100 "
            "when padding-free is on (bfd forces this). "
            "Scoring skips the first record token. Non-FlashAttention packing may "
            "cross-contaminate documents (TRL warning) - treat scores as slightly biased."
        )

    if train.get("skip_prepare_dataset"):
        msg = (
            "skip_prepare_dataset=True: TRL did not prepare the dataset. "
            "Supervision is owned by the supplied collator. Pre-flight will invoke "
            "the collator when possible; otherwise evidence is verification_unknown."
        )
        warnings.append(msg)
        findings.append({"code": "skip_prepare_dataset", "level": "warning", "message": msg})

    if train.get("use_liger_kernel"):
        warnings.append(
            "use_liger_kernel is on: user-supplied labels columns may be dropped "
            "by TRL select_columns. Mask columns are inspected when present."
        )

    if train.get("assistant_only_loss") and train.get("chat_template_has_generation_markers") is False:
        warnings.append(
            "assistant_only_loss is on but the chat template has no {% generation %} "
            "markers; assistant masks may be absent (verification_unknown if we "
            "cannot read labels)."
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
                    n_tok = _canary_record_n_tokens(c, fmt, tokenizer)
                except Exception:
                    pass
            if n_tok and n_tok > int(max_len):
                too_long.append(c.get("id"))
        if too_long:
            msg = (
                f"{len(too_long)} canary *records* (prefix + secret, including "
                f"chat-template / special-token overhead) exceed max_length={max_len} "
                "and will be truncated (keep_start). This silently zeros those canaries."
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
    effective_col = train.get("completion_only_loss_effective")
    if (train.get("completion_only_loss") or effective_col) and fmt == "text":
        warnings.append(
            "completion_only_loss is set but canaries were injected as 'text' records. "
            "If the live dataset is prompt+completion, re-inject with fmt='prompt_completion'."
        )
    if train.get("assistant_only_loss") and fmt not in {None, "messages"}:
        warnings.append(
            "assistant_only_loss=True but canaries are not in messages format. "
            "User-turn / prompt secrets would be labeled -100."
        )

    scan = survival_scan(
        dataset,
        manifest,
        tokenizer=tokenizer,
        trainer=trainer,
        args=args if args is not None else getattr(trainer, "args", None),
        config=train,
        limit=scan_limit,
    )
    if scan["n_inserted"] == 0:
        fatal.append(
            "No canaries were marked included in the manifest. inject() coin-flips may "
            "have excluded every candidate, or you passed controls only. The audit "
            "would be silently empty."
        )
    elif scan["n_found"] == 0:
        if scan.get("scan_complete"):
            fatal.append(
                f"0 of {scan['n_inserted']} inserted canary secrets were found after a "
                "complete inspection of the prepared dataset (possible TRL fully-masked "
                "row filter, truncation, or formatting_func — not a scan-window miss). "
                "Canary injection cannot happen inside the Trainer callback - the "
                "dataloader is built (and TRL tokenizes/packs) before any hook fires. Call "
                "memaudit.inject() on the RAW dataset BEFORE constructing Trainer/SFTTrainer."
            )
        else:
            fatal.append(
                f"0 of {scan['n_inserted']} inserted canary secrets were found in the "
                f"scanned window ({scan.get('rows_scanned')} of {scan.get('rows_total')} "
                "rows). Canary injection cannot happen inside the Trainer callback - the "
                "dataloader is built (and TRL tokenizes/packs) before any hook fires. Call "
                "memaudit.inject() on the RAW dataset BEFORE constructing Trainer/SFTTrainer."
            )
    elif scan["n_missing"] > 0:
        if scan.get("scan_complete"):
            fatal.append(
                f"{scan['n_missing']} inserted canaries are not present after a complete "
                f"inspection of the prepared dataset (missing ids: {scan['missing_ids']}). "
                "This is a fatal supervision loss (row deleted by a prep filter, or the "
                "secret was truncated away) — not scan-window noise."
            )
        else:
            warnings.append(
                f"{scan['n_missing']} inserted canaries were not observed within the "
                f"scan window (scanned {scan.get('rows_scanned')} of "
                f"{scan.get('rows_total')} rows; missing ids: {scan['missing_ids']}). "
                "They may still appear later in the dataset. Truncation or a "
                "formatting_func is only a cause after a complete scan."
            )
    if scan["n_fully_masked"] > 0:
        fatal.append(
            f"{scan['n_fully_masked']} canaries appear in input_ids but every secret token "
            "has labels=-100 (completion_only_loss / assistant_only_loss / packing). "
            "The canary protocol expected this secret span to be directly supervised; "
            "pre-flight did not verify that condition. This invalidates the "
            "supervised-memorization probe — it does not mean the tokens are "
            "untrainable or that information cannot be memorized."
        )
    if scan.get("n_token_stream_missing"):
        stream_msg = (
            f"{scan['n_token_stream_missing']} canaries appear only in raw text "
            "columns of the prepared dataset "
            f"(ids: {scan.get('token_stream_missing_ids')}); their secret tokens "
            "are absent from the tokenized training stream (input_ids). "
            "Truncation or re-tokenization removed the supervised span - the "
            "model never trains on those tokens."
        )
        if scan.get("scan_complete"):
            fatal.append(
                stream_msg
                + " This is a fatal supervision loss, verified after a complete "
                "inspection of the prepared dataset."
            )
        else:
            warnings.append(
                stream_msg
                + " The scan window is incomplete, so a surviving copy may exist "
                "in unscanned rows."
            )
    if scan.get("n_mask_column_missing"):
        msg = (
            f"{scan['n_mask_column_missing']} canary-bearing rows have no mask column "
            "while the training config says masking is on "
            f"(ids: {scan.get('mask_column_missing_ids')}). Effective supervision is "
            "full-sequence; audit semantics differ from the recorded protocol."
        )
        warnings.append(msg)
        findings.append({"code": "mask_column_absent", "level": "warning", "message": msg})

    unsupervised_ids = [
        ev["id"]
        for ev in (scan.get("per_canary") or [])
        if ev.get("loss_mask_checked")
        and ev.get("supervised_token_fraction") == 0.0
        and ev.get("record_observed")
    ]
    if unsupervised_ids and scan["n_fully_masked"] == 0:
        fatal.append(
            f"{len(unsupervised_ids)} canaries have a secret span that is not directly "
            "supervised (labels=-100 on every secret token). The canary protocol "
            "expected this span to be directly supervised; this invalidates the "
            "supervised-memorization probe."
        )

    unknown_checked = [
        ev["id"]
        for ev in (scan.get("per_canary") or [])
        if ev.get("verification_unknown") and ev.get("record_observed")
    ]
    if unknown_checked:
        warnings.append(
            f"{len(unknown_checked)} observed canaries could not have their loss mask "
            "verified (verification_unknown). This is not a silent pass."
        )

    partial = [
        ev
        for ev in (scan.get("per_canary") or [])
        if ev.get("loss_mask_checked")
        and ev.get("supervised_token_fraction") is not None
        and 0.0 < float(ev["supervised_token_fraction"]) < 1.0
    ]
    if partial:
        warnings.append(
            f"{len(partial)} canaries are only partially supervised "
            "(e.g. padding-free first-token -100). See per-canary "
            "supervised_token_fraction."
        )
    straddled = [ev["id"] for ev in (scan.get("per_canary") or []) if ev.get("split_across_packed_rows")]
    if straddled:
        warnings.append(
            f"{len(straddled)} canaries have a secret split across packed rows "
            f"(ids: {straddled[:10]}). Membership scores measure two fragments, "
            "not one contiguous supervised span."
        )

    # new-token family leftover + trainable_token_indices ∩ secret ids
    families = {c.get("family") for c in manifest.get("canaries") or []}
    trained_idx = emb.get("trainable_token_index_set")
    tok_fatal, tok_warn, tok_rows = trainable_token_canary_findings(manifest, trained_idx)
    fatal.extend(tok_fatal)
    warnings.extend(tok_warn)
    if tok_rows:
        emb["canary_token_coverage"] = tok_rows
        findings.append(
            {
                "code": "trainable_token_indices",
                "level": "fatal" if tok_fatal else "info",
                "message": "Intersected canary secret_token_ids with trainable_token_indices",
                "n_frozen_canaries": sum(1 for r in tok_rows if r.get("all_frozen")),
            }
        )
    if "new_token" in families and not emb.get("trainable") and not tok_fatal:
        fatal.append(
            "Manifest contains new_token canaries but embeddings are not trainable. "
            "That family is unimplemented/gated in v0.1 and would measure noise."
        )

    tying = emb.get("weight_tying") or {}
    if tying.get("tie_broken_by_wrap"):
        msg = (
            "tie_word_embeddings=True but wrapping one side of the tied pair "
            "broke storage sharing (weight data_ptr diverged). Embedding-row "
            "and output-row updates are no longer the same tensor."
        )
        warnings.append(msg)
        findings.append({"code": "tie_broken_by_wrap", "level": "warning", "message": msg})
    if emb.get("ensure_weight_tying") and tying.get("ensure_weight_tying_engaged") is False:
        msg = (
            "ensure_weight_tying is set but PEFT's name-gated tying did not "
            "engage (names outside embed_tokens/lm_head, or copies do not share "
            "storage). Embedding claims are verification_unknown."
        )
        warnings.append(msg)
        findings.append({"code": "ensure_weight_tying_inert", "level": "warning", "message": msg})

    if emb.get("target_parameters"):
        msg = (
            f"target_parameters={emb['target_parameters']!r} is experimental "
            "(MoE / nn.Parameter LoRA). This is not an unexamined all-clear: "
            "embedding/adapter verification is verification_unknown until the "
            "audit-time base-equivalence guard can reason about the toggle."
        )
        warnings.append(msg)
        findings.append({"code": "target_parameters", "level": "warning", "message": msg})

    if emb.get("quantized"):
        msg = (
            "Quantized base detected "
            f"({emb.get('quantization_kinds')}). disable_adapter() is the "
            "matching reference; a separately loaded full-precision --ref "
            "would silently bias base-calibrated scores."
        )
        warnings.append(msg)
        findings.append({"code": "quantized_base", "level": "warning", "message": msg})

    if emb.get("embedding_verification") == "verification_unknown":
        for reason in emb.get("embedding_verification_reasons") or []:
            if reason not in warnings:
                warnings.append(reason)
        findings.append(
            {
                "code": "embedding_verification_unknown",
                "level": "warning",
                "message": "; ".join(emb.get("embedding_verification_reasons") or ["unverified"]),
            }
        )

    capture = None
    if toggle_ok and is_peft_model(model) and tokenizer is not None:
        capture = capture_disabled_logits(model, tokenizer, manifest=manifest)

    embedding_unknown = emb.get("embedding_verification") == "verification_unknown"
    result = {
        "ran": True,
        "path": "callback",
        "verification_unknown": bool(scan.get("n_verification_unknown") or embedding_unknown),
        "embeddings": emb,
        "training": {
            **train,
            "epochs": train.get("num_train_epochs"),
            "learning_rate": train.get("learning_rate"),
        },
        "adapter_toggle_safe": toggle_ok,
        "survival": scan,
        "rows_total": scan.get("rows_total"),
        "rows_scanned": scan.get("rows_scanned"),
        "scan_complete": scan.get("scan_complete"),
        "loss_mask_verified": scan.get("n_loss_mask_verified"),
        "directly_supervised": scan.get("n_directly_supervised"),
        "findings": findings,
        "warnings": warnings,
        "fatal": fatal,
        "per_canary": scan.get("per_canary") or [],
        "base_equivalence_capture": capture,
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
