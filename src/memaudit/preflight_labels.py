"""Collator-based label verification and per-canary evidence levels.

Labels often exist only inside the data collator (TRL 0.29.x). This module
invokes the live collator when possible, falls back to mask columns, and
never silently upgrades a canary to ``directly_supervised``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from memaudit.injection import ASSISTANT_PREFIX, TEXT_PREFIX
from memaudit.utils import decode_ids, encode_ids, example_text, find_subsequence

LABEL_IGNORE = -100

# Monotone ladder. verification_unknown is a cap, not a rung.
EVIDENCE_NOT_OBSERVED = "not_observed"
EVIDENCE_RECORD_OBSERVED = "record_observed"
EVIDENCE_SECRET_TOKEN_ALIGNED = "secret_token_aligned"
EVIDENCE_LOSS_MASK_CHECKED = "loss_mask_checked"
EVIDENCE_DIRECTLY_SUPERVISED = "directly_supervised"


def as_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, int):
        return [int(value)]
    if not value:
        return []
    if isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(x) for x in value]


def row_has_text_keys(row: Mapping[str, Any]) -> bool:
    return any(k in row for k in ("text", "messages", "prompt", "completion"))


def row_input_ids(
    row: Mapping[str, Any],
    tokenizer: Any | None,
    fmt: str,
) -> list[int] | None:
    ids = as_int_list(row.get("input_ids"))
    if ids is not None:
        return ids
    if tokenizer is None:
        return None
    use_fmt = fmt if row_has_text_keys(row) else "text"
    blob = example_text(row, use_fmt)
    if not blob:
        return None
    return encode_ids(tokenizer, blob, add_special_tokens=False)


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def context_needles(
    canary: Mapping[str, Any],
    tokenizer: Any | None,
    fmt: str,
) -> list[list[int]]:
    """Needles for the alignment ladder: standalone ids, then junction strings."""
    needles: list[list[int]] = []
    standalone = as_int_list(canary.get("secret_token_ids")) or []
    if standalone:
        needles.append(standalone)
    if tokenizer is None:
        return needles
    secret = str(canary.get("secret") or "")
    if not secret:
        return needles
    junctions = [TEXT_PREFIX, ASSISTANT_PREFIX]
    if fmt == "text":
        junctions = [TEXT_PREFIX]
    elif fmt in {"prompt_completion", "messages"}:
        junctions = [ASSISTANT_PREFIX, TEXT_PREFIX]
    seen: set[tuple[int, ...]] = {tuple(standalone)} if standalone else set()
    for ctx in junctions:
        try:
            full = encode_ids(tokenizer, ctx + secret, add_special_tokens=False)
            ctx_ids = encode_ids(tokenizer, ctx, add_special_tokens=False)
        except Exception:
            continue
        needle = full[_common_prefix_len(ctx_ids, full) :]
        key = tuple(needle)
        if needle and key not in seen:
            seen.add(key)
            needles.append(needle)
    return needles


def char_cover_span(
    ids: Sequence[int],
    secret: str,
    tokenizer: Any,
) -> tuple[int, int] | None:
    """Covering token span for ``secret`` via cumulative decode. O(n) decodes."""
    if not secret or tokenizer is None or not ids:
        return None
    prefixes: list[str] = [""]
    for i in range(1, len(ids) + 1):
        prefixes.append(decode_ids(tokenizer, ids[:i], skip_special_tokens=True))
    full = prefixes[-1]
    pos = full.find(secret)
    if pos < 0:
        return None
    end_char = pos + len(secret)
    start: int | None = None
    stop: int | None = None
    for i in range(1, len(prefixes)):
        prev_len = len(prefixes[i - 1])
        curr_len = len(prefixes[i])
        if curr_len > pos and prev_len < end_char:
            if start is None:
                start = i - 1
            stop = i
    if start is None or stop is None:
        return None
    return (start, stop)


@dataclass
class AlignmentHit:
    row: Mapping[str, Any]
    ids: list[int] | None
    span: tuple[int, int] | None
    alignment: str
    straddle: bool = False
    partner_row: Mapping[str, Any] | None = None
    row_index: int | None = None


def align_in_ids(
    ids: Sequence[int] | None,
    needles: Sequence[Sequence[int]],
    secret: str,
    tokenizer: Any | None,
    decoded: str | None,
) -> tuple[tuple[int, int], str] | None:
    if not ids:
        return None
    for needle in needles:
        loc = find_subsequence(ids, needle)
        if loc:
            return loc, "exact"
    if secret and decoded and secret in decoded and tokenizer is not None:
        cover = char_cover_span(ids, secret, tokenizer)
        if cover:
            return cover, "covering"
    return None


def align_straddle(
    left_ids: Sequence[int],
    right_ids: Sequence[int],
    needles: Sequence[Sequence[int]],
) -> tuple[int, int] | None:
    """Return the span in the concatenated ids if a needle straddles the join."""
    if not left_ids or not right_ids:
        return None
    concat = list(left_ids) + list(right_ids)
    join = len(left_ids)
    for needle in needles:
        loc = find_subsequence(concat, needle)
        if loc and loc[0] < join < loc[1]:
            return loc
    return None


@dataclass
class LabelsResult:
    labels: list[int]
    source: str  # dataset_column | collator_call | reimplemented | mask_column


def _labels_from_collator(collator: Any, row: Mapping[str, Any]) -> list[int] | None:
    if collator is None or not callable(collator):
        return None
    try:
        batch = collator([dict(row)])
    except Exception:
        return None
    if not isinstance(batch, Mapping) or "labels" not in batch:
        return None
    labels = as_int_list(batch["labels"])
    if labels is None:
        return None
    ids = as_int_list(row.get("input_ids"))
    if ids is not None and len(labels) > len(ids):
        labels = labels[: len(ids)]
    return labels


def reimplement_collator_labels(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[int] | None:
    """Replicate TRL DataCollatorForLanguageModeling mask rules (fallback)."""
    ids = as_int_list(row.get("input_ids"))
    if ids is None:
        return None
    raw = as_int_list(row.get("labels"))
    labels = list(raw) if raw is not None else list(ids)
    if len(labels) < len(ids):
        labels = labels + [LABEL_IGNORE] * (len(ids) - len(labels))
    labels = labels[: len(ids)]

    completion_mask = as_int_list(row.get("completion_mask"))
    if config.get("completion_only_loss_effective") and completion_mask is not None:
        labels = [
            LABEL_IGNORE if (i < len(completion_mask) and completion_mask[i] == 0) else lab
            for i, lab in enumerate(labels)
        ]
    assistant_masks = as_int_list(row.get("assistant_masks"))
    if assistant_masks is not None:
        labels = [
            LABEL_IGNORE if (i < len(assistant_masks) and assistant_masks[i] == 0) else lab
            for i, lab in enumerate(labels)
        ]

    padding_free = bool(config.get("padding_free"))
    seq_lengths = as_int_list(row.get("seq_lengths"))
    if padding_free and seq_lengths:
        pos = 0
        for slen in seq_lengths:
            if 0 <= pos < len(labels):
                labels[pos] = LABEL_IGNORE
            pos += slen
    elif padding_free and labels:
        labels[0] = LABEL_IGNORE
    return labels


def labels_from_masks(row: Mapping[str, Any], config: Mapping[str, Any]) -> list[int] | None:
    ids = as_int_list(row.get("input_ids"))
    if ids is None:
        return None
    completion_mask = as_int_list(row.get("completion_mask"))
    assistant_masks = as_int_list(row.get("assistant_masks"))
    if completion_mask is None and assistant_masks is None:
        return None
    labels = list(ids)
    if config.get("completion_only_loss_effective") and completion_mask is not None:
        labels = [
            LABEL_IGNORE if (i < len(completion_mask) and completion_mask[i] == 0) else lab
            for i, lab in enumerate(labels)
        ]
    if assistant_masks is not None:
        labels = [
            LABEL_IGNORE if (i < len(assistant_masks) and assistant_masks[i] == 0) else lab
            for i, lab in enumerate(labels)
        ]
    return labels


def effective_labels(
    row: Mapping[str, Any],
    trainer: Any | None,
    config: Mapping[str, Any],
) -> LabelsResult | None:
    """Ground-truth labels: dataset column, live collator, masks, then reimplement."""
    col = as_int_list(row.get("labels"))
    if col is not None:
        return LabelsResult(col, "dataset_column")

    collator = getattr(trainer, "data_collator", None) if trainer is not None else None
    from_collator = _labels_from_collator(collator, row)
    if from_collator is not None:
        return LabelsResult(from_collator, "collator_call")

    from_masks = labels_from_masks(row, config)
    if from_masks is not None:
        return LabelsResult(from_masks, "mask_column")

    if collator is not None or config.get("completion_only_loss_effective") or config.get(
        "assistant_only_loss"
    ):
        rebuilt = reimplement_collator_labels(row, config)
        if rebuilt is not None and (
            "completion_mask" in row or "assistant_masks" in row or "seq_lengths" in row
        ):
            return LabelsResult(rebuilt, "reimplemented")
    return None


def expected_mask_column(config: Mapping[str, Any], fmt: str | None) -> str | None:
    if config.get("assistant_only_loss"):
        return "assistant_masks"
    if config.get("completion_only_loss_effective"):
        return "completion_mask"
    if config.get("completion_only_loss") and fmt in {"prompt_completion", "prompt+completion"}:
        return "completion_mask"
    return None


def mask_column_missing(row: Mapping[str, Any], config: Mapping[str, Any], fmt: str | None) -> bool:
    """Fixture B2: config says masking is on, but the prepared row has no mask or labels."""
    expected = expected_mask_column(config, fmt)
    if not expected:
        return False
    if "labels" in row:
        return False
    return expected not in row


def span_supervised_fraction(labels: Sequence[int], span: tuple[int, int]) -> float:
    start, stop = span
    sl = labels[start:stop]
    if not sl:
        return 0.0
    kept = sum(1 for x in sl if int(x) != LABEL_IGNORE)
    return kept / len(sl)


def document_contains_span(seq_lengths: Sequence[int] | None, span: tuple[int, int]) -> bool:
    if not seq_lengths:
        return True
    start, stop = span
    pos = 0
    for slen in seq_lengths:
        end = pos + slen
        if start >= pos and stop <= end:
            return True
        pos = end
    return False


@dataclass
class CanaryEvidence:
    id: str
    evidence_level: str = EVIDENCE_NOT_OBSERVED
    record_observed: bool = False
    secret_token_aligned: bool = False
    loss_mask_checked: bool = False
    directly_supervised: bool = False
    verification_unknown: bool = False
    alignment: str | None = None
    labels_source: str | None = None
    supervised_token_fraction: float | None = None
    split_across_packed_rows: bool = False
    miss_reason: str | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "evidence_level": self.evidence_level,
            "record_observed": self.record_observed,
            "secret_token_aligned": self.secret_token_aligned,
            "loss_mask_checked": self.loss_mask_checked,
            "directly_supervised": self.directly_supervised,
            "verification_unknown": self.verification_unknown,
            "alignment": self.alignment,
            "labels_source": self.labels_source,
            "supervised_token_fraction": self.supervised_token_fraction,
            "split_across_packed_rows": self.split_across_packed_rows,
            "miss_reason": self.miss_reason,
            "reasons": list(self.reasons),
        }


def _cap_unknown(ev: CanaryEvidence, reason: str) -> CanaryEvidence:
    ev.verification_unknown = True
    ev.reasons.append(reason)
    return ev


def finalize_evidence(
    ev: CanaryEvidence,
    hit: AlignmentHit | None,
    labels_result: LabelsResult | None,
    *,
    custom_loss: bool,
    scan_complete: bool,
    rows_total_known: bool,
) -> CanaryEvidence:
    if hit is None:
        ev.evidence_level = EVIDENCE_NOT_OBSERVED
        if scan_complete:
            ev.miss_reason = "not_found_after_complete_scan"
            ev.reasons.append(
                "not observed after a complete inspection of the prepared dataset "
                "(row filter, truncation, or formatting_func — not a scan-window miss)"
            )
        elif rows_total_known:
            ev.miss_reason = "scan_window"
            ev.verification_unknown = True
            ev.reasons.append("not observed within the bounded scan window")
        else:
            ev.miss_reason = "scan_window"
            ev.verification_unknown = True
            ev.reasons.append("not observed; dataset length unknown so the scan may be incomplete")
        return ev

    ev.record_observed = True
    ev.evidence_level = EVIDENCE_RECORD_OBSERVED
    ev.split_across_packed_rows = bool(hit.straddle)
    if hit.alignment in {"exact", "covering"}:
        ev.secret_token_aligned = True
        ev.alignment = hit.alignment
        ev.evidence_level = EVIDENCE_SECRET_TOKEN_ALIGNED
    elif hit.alignment == "straddle":
        ev.alignment = "straddle"
        ev.reasons.append(
            "secret split across packed rows (wrapped packing); not one contiguous supervised span"
        )
    elif hit.alignment == "string":
        ev.alignment = "string"
        ev.reasons.append("observed in decoded/record text; token span not aligned")

    if labels_result is None:
        return _cap_unknown(
            ev,
            "no label source at on_train_begin (no labels column, collator did not "
            "yield labels, and no completion_mask/assistant_masks to inspect)",
        )

    ev.labels_source = labels_result.source
    ev.loss_mask_checked = True
    ev.evidence_level = EVIDENCE_LOSS_MASK_CHECKED

    if custom_loss:
        ev.verification_unknown = True
        ev.reasons.append(
            "custom compute_loss_func is set; labels were checked but "
            "directly_supervised is not claimed beyond labels != -100"
        )
        return ev

    if hit.straddle:
        ev.reasons.append(
            "secret is not a single contiguous span inside one packed document"
        )
        return ev

    if hit.span is None or hit.ids is None:
        ev.verification_unknown = True
        ev.reasons.append("loss mask readable but secret token span was not aligned")
        return ev

    seq_lengths = as_int_list(hit.row.get("seq_lengths"))
    if not document_contains_span(seq_lengths, hit.span):
        ev.split_across_packed_rows = True
        ev.reasons.append(
            "secret is not a single contiguous span inside one packed document"
        )
        return ev

    frac = span_supervised_fraction(labels_result.labels, hit.span)
    ev.supervised_token_fraction = frac

    if frac >= 1.0:
        ev.directly_supervised = True
        ev.evidence_level = EVIDENCE_DIRECTLY_SUPERVISED
    elif frac <= 0.0:
        ev.reasons.append(
            "secret span is fully masked (labels=-100). The canary protocol "
            "expected this span to be directly supervised; pre-flight did not "
            "verify that. This invalidates the supervised-memorization probe "
            "— it does not mean the tokens cannot be memorized."
        )
    else:
        ev.reasons.append(
            f"partial supervision ({frac:.0%} of the covering secret span has labels != -100)"
        )
    return ev
